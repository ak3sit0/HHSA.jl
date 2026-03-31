# ─────────────────────────────────────────────────────────────────────────────
# holo_spectrum.jl  –  Hilbert-based instantaneous parameters + two-layer HHSA
#
# Reference: Huang et al. (2016) Phil. Trans. R. Soc. A 374, 20150196
#            "On Holo-Hilbert spectral analysis: a full informational spectral
#             representation for nonlinear and non-stationary data"
# ─────────────────────────────────────────────────────────────────────────────

# ── Instantaneous amplitude and frequency ───────────────────────────────────
"""
    instantaneous_params(signal, fs) → (amplitude, frequency)

Return the instantaneous amplitude and frequency (in Hz) of `signal`
via the Hilbert analytic signal.

`DSP.hilbert` returns the analytic signal z = signal + i·H[signal].
"""
function instantaneous_params(signal::Vector{Float64}, fs::Float64)
    z   = DSP.hilbert(signal)            # analytic signal
    amp = abs.(z)                         # instantaneous amplitude  A(t)
    φ   = angle.(z)                       # instantaneous phase      θ(t)
    # Unwrap phase then differentiate; divide by 2π to get Hz
    dφ  = diff(DSP.unwrap(φ))
    # Pad to keep the same length (repeat last value)
    push!(dφ, dφ[end])
    freq = dφ .* (fs / (2π))
    # Clamp negative frequencies (can arise at flat/endpoint regions)
    freq = max.(freq, 0.0)
    return amp, freq
end

# ── Result container ─────────────────────────────────────────────────────────
"""
    HoloResult

Holds all intermediate and final products of a two-layer HHSA.

Fields
──────
- `imfs`        : Stage-1 IMFs  (one vector per IMF, last is residue)
- `carrier_amp` : A_j(t) – instantaneous amplitude of each Stage-1 IMF
- `carrier_freq`: ω_j(t) – instantaneous carrier frequency (Hz)
- `mod_imfs`    : Stage-2 IMFs of each A_j(t); mod_imfs[j][k]
- `mod_amp`     : instantaneous amplitude of each Stage-2 IMF
- `mod_freq`    : Ω_{j,k}(t) – instantaneous modulation frequency (Hz)
- `fs`          : sample rate used
"""
struct HoloResult
    imfs        ::Vector{Vector{Float64}}
    carrier_amp ::Vector{Vector{Float64}}
    carrier_freq::Vector{Vector{Float64}}
    mod_imfs    ::Vector{Vector{Vector{Float64}}}
    mod_amp     ::Vector{Vector{Vector{Float64}}}
    mod_freq    ::Vector{Vector{Vector{Float64}}}
    fs          ::Float64
end

# ── Two-layer HHSA ────────────────────────────────────────────────────────────
"""
    hhsa(signal, fs; emd_kw...) → HoloResult

Perform Holo-Hilbert Spectral Analysis on `signal` sampled at `fs` Hz.

Algorithm
─────────
1. **Stage 1** – EMD of `signal` → IMFs  {c_j(t)}
2.  For each c_j: Hilbert transform → amplitude A_j(t), carrier freq ω_j(t)
3. **Stage 2 (the Holo step)** – EMD of A_j(t) → modulation IMFs {IA_{j,k}(t)}
4.  For each IA_{j,k}: Hilbert transform → modulation freq Ω_{j,k}(t)

The output can be used to build the full Holo spectrum
(carrier freq × modulation freq energy map) via `holo_spectrum`.
"""
function hhsa(signal::Vector{Float64}, fs::Float64; emd_kw...)
    # ── Stage 1 ──────────────────────────────────────────────────────────────
    imfs = emd(signal; emd_kw...)

    carrier_amp  = Vector{Vector{Float64}}()
    carrier_freq = Vector{Vector{Float64}}()
    mod_imfs     = Vector{Vector{Vector{Float64}}}()
    mod_amp      = Vector{Vector{Vector{Float64}}}()
    mod_freq     = Vector{Vector{Vector{Float64}}}()

    for imf in imfs
        # Stage-1 instantaneous params
        amp_j, freq_j = instantaneous_params(imf, fs)
        push!(carrier_amp,  amp_j)
        push!(carrier_freq, freq_j)

        # ── Stage 2 (Holo step) ───────────────────────────────────────────
        # EMD the amplitude envelope of this IMF
        amp_imfs_j = emd(amp_j; emd_kw...)

        mod_imfs_j = Vector{Vector{Float64}}()
        mod_amp_j  = Vector{Vector{Float64}}()
        mod_freq_j = Vector{Vector{Float64}}()

        for amp_imf in amp_imfs_j
            a_jk, f_jk = instantaneous_params(amp_imf, fs)
            push!(mod_imfs_j, amp_imf)
            push!(mod_amp_j,  a_jk)
            push!(mod_freq_j, f_jk)
        end

        push!(mod_imfs, mod_imfs_j)
        push!(mod_amp,  mod_amp_j)
        push!(mod_freq, mod_freq_j)
    end

    return HoloResult(imfs, carrier_amp, carrier_freq,
                      mod_imfs, mod_amp, mod_freq, fs)
end

# ── Holo Spectrum (2-D energy map) ───────────────────────────────────────────
"""
    holo_spectrum(result; n_carrier, n_mod, f_carrier_max, f_mod_max)
        → (carrier_axis, mod_axis, holo_matrix)

Bin the energy A²(t) from each Stage-2 IMF into a 2-D histogram over
(carrier frequency ω, modulation frequency Ω).

Returns the frequency axes and the energy matrix (mod_freq × carrier_freq).
"""
function holo_spectrum(result::HoloResult;
                       n_carrier   ::Int     = 128,
                       n_mod       ::Int     = 64,
                       f_carrier_max::Float64 = result.fs / 2,
                       f_mod_max    ::Float64 = result.fs / 4)

    carrier_axis = range(0.0, f_carrier_max, length = n_carrier)
    mod_axis     = range(0.0, f_mod_max,     length = n_mod)
    Δω = step(carrier_axis)
    ΔΩ = step(mod_axis)

    H = zeros(n_mod, n_carrier)   # rows = Ω (mod), cols = ω (carrier)

    n_imfs = length(result.imfs)
    for j in 1:n_imfs
        ω_j = result.carrier_freq[j]     # length N
        for k in eachindex(result.mod_imfs[j])
            Ω_jk = result.mod_freq[j][k]  # length N
            A_jk = result.mod_amp[j][k]   # length N
            for t in eachindex(ω_j)
                # ── Physical constraint (Huang et al. 2016, Section 3, Fig. 6c) ──
                # Ω is derived from the slowly-varying amplitude envelope, so it
                # must always satisfy  ω > Ω.  Points that violate this are
                # physically invalid and are excluded from the spectrum.
                ω_j[t] > Ω_jk[t] || continue

                ci = clamp(round(Int, ω_j[t] / Δω) + 1, 1, n_carrier)
                mi = clamp(round(Int, Ω_jk[t] / ΔΩ) + 1, 1, n_mod)
                H[mi, ci] += A_jk[t]^2
            end
        end
    end

    return collect(carrier_axis), collect(mod_axis), H
end

# ── Plotting helpers ─────────────────────────────────────────────────────────
"""
    plot_imfs(result; fs, title)

Plot the original signal reconstruction, Stage-1 IMFs, and residue.
"""
function plot_imfs(result::HoloResult; fs::Float64 = result.fs,
                   title::String = "Stage-1 IMFs")
    n_imfs = length(result.imfs)
    n      = length(result.imfs[1])
    t      = (0:n-1) ./ fs

    plots = []
    for (i, imf) in enumerate(result.imfs)
        label = i < n_imfs ? "IMF $i" : "Residue"
        push!(plots, Plots.plot(t, imf; label, ylabel = label,
                                linewidth = 1.5, legend = :topright))
    end
    return Plots.plot(plots...; layout = (n_imfs, 1),
                      suptitle = title, size = (800, 150 * n_imfs))
end

"""
    plot_holo_spectrum(result; kwargs...)

2-D heatmap of carrier frequency (x-axis) vs. modulation frequency (y-axis),
coloured by cumulative energy  Σ A²(t).
"""
function plot_holo_spectrum(result::HoloResult;
                            n_carrier    ::Int     = 128,
                            n_mod        ::Int     = 64,
                            f_carrier_max::Float64 = result.fs / 2,
                            f_mod_max    ::Float64 = result.fs / 4,
                            title        ::String  = "Holo-Hilbert Spectrum")
    ω_ax, Ω_ax, H = holo_spectrum(result;
                                  n_carrier, n_mod,
                                  f_carrier_max, f_mod_max)

    return Plots.heatmap(ω_ax, Ω_ax, H;
                         xlabel    = "Carrier frequency ω (Hz)",
                         ylabel    = "Modulation frequency Ω (Hz)",
                         title,
                         color     = :viridis,
                         colorbar_title = "Energy (A²)",
                         size      = (800, 400))
end

"""
    plot_hilbert_spectrum(result; n_freq, f_max, title)

Classic 2-D Hilbert spectrum: time × carrier frequency, coloured by
instantaneous amplitude. One heatmap per Stage-1 IMF.
"""
function plot_hilbert_spectrum(result::HoloResult;
                               n_freq ::Int     = 128,
                               f_max  ::Float64 = result.fs / 2,
                               title  ::String  = "Hilbert Spectrum")
    n      = length(result.imfs[1])
    fs     = result.fs
    t      = (0:n-1) ./ fs
    f_axis = range(0.0, f_max, length = n_freq)
    Δf     = step(f_axis)

    H2 = zeros(n_freq, n)  # freq × time
    for j in 1:length(result.imfs)-1    # skip residue
        amp_j  = result.carrier_amp[j]
        freq_j = result.carrier_freq[j]
        for ti in 1:n
            fi = clamp(round(Int, freq_j[ti] / Δf) + 1, 1, n_freq)
            H2[fi, ti] += amp_j[ti]
        end
    end

    return Plots.heatmap(t, collect(f_axis), H2;
                         xlabel = "Time (s)",
                         ylabel = "Frequency (Hz)",
                         title,
                         color  = :hot,
                         colorbar_title = "Amplitude",
                         size   = (900, 400))
end
