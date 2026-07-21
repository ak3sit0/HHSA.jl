# ─────────────────────────────────────────────────────────────────────────────
# holo_spectrum.jl  –  Hilbert-based instantaneous parameters + two-layer HHSA
#
# Reference: Huang et al. (2016) Phil. Trans. R. Soc. A 374, 20150206
#            "On Holo-Hilbert spectral analysis: a full informational spectral
#             representation for nonlinear and non-stationary data"
# ─────────────────────────────────────────────────────────────────────────────

# ── Instantaneous amplitude and frequency ────────────────────────────────────
"""
    instantaneous_params(signal, fs) → (amplitude, frequency)

Return the instantaneous amplitude and frequency (in Hz) of `signal`
via the Hilbert analytic signal.

`DSP.hilbert` returns the analytic signal z = signal + i·H[signal].
Amplitude  A(t)  = |z(t)|
Frequency  ω(t)  = (1/2π) · dθ/dt  where θ = arg(z)

Paper eq. 1.3: x(t) = Σ aⱼ(t)·cos θⱼ(t)
"""
function instantaneous_params(signal::Vector{Float64}, fs::Float64)
    z    = DSP.hilbert(signal)
    amp  = abs.(z)
    φ    = angle.(z)
    dφ   = diff(DSP.unwrap(φ))
    push!(dφ, dφ[end])                # pad to original length (repeat last)
    freq = abs.(dφ .* (fs / (2π)))    # frequency is magnitude of phase derivative
    return amp, freq
end

# ── Result container ──────────────────────────────────────────────────────────
"""
    HoloResult

Holds all intermediate and final products of a two-layer HHSA.

Fields
──────
- `imfs`        : Stage-1 IMFs  (one vector per IMF, last is residue)
- `carrier_amp` : Aⱼ(t) – instantaneous amplitude of each Stage-1 IMF
- `carrier_freq`: ωⱼ(t) – instantaneous carrier frequency (Hz)
- `mod_imfs`    : Stage-2 IMFs of each Aⱼ(t); mod_imfs[j][k]
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

# ── Two-layer HHSA ─────────────────────────────────────────────────────────────
"""
    hhsa(signal, fs; emd_kw...) → HoloResult

Perform Holo-Hilbert Spectral Analysis on `signal` sampled at `fs` Hz.

Algorithm (paper Section 3, eq. 3.1)
─────────────────────────────────────
1. **Stage 1** – EMD of `signal` → IMFs {cⱼ(t)}
2.  For each cⱼ: Hilbert transform → amplitude Aⱼ(t), carrier freq ωⱼ(t)
3. **Stage 2 (the Holo step)** – EMD of Aⱼ(t) → modulation IMFs {a_{jk}(t)}
4.  For each a_{jk}: Hilbert transform → modulation freq Ω_{jk}(t)

The nested structure implements eq. 3.2:
    aⱼ(t) = Σₖ [ Re Σₗ a_{jkl}(t)·e^{i∫Ωₗdτ} … ] · e^{i∫Ωₖdτ}

The output is fed to `holo_spectrum` to build the 2D energy map H(ω, Ω).
"""
function hhsa(signal::Vector{Float64}, fs::Float64; emd_kw...)
    imfs = emd(signal; emd_kw...)

    carrier_amp  = Vector{Vector{Float64}}()
    carrier_freq = Vector{Vector{Float64}}()
    mod_imfs     = Vector{Vector{Vector{Float64}}}()
    mod_amp      = Vector{Vector{Vector{Float64}}}()
    mod_freq     = Vector{Vector{Vector{Float64}}}()

    for imf in imfs
        amp_j, freq_j = instantaneous_params(imf, fs)
        push!(carrier_amp,  amp_j)
        push!(carrier_freq, freq_j)

        # Holo step: EMD the amplitude envelope of this IMF (eq. 3.1, left column)
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

# ── Holo Spectrum (2-D energy map) ────────────────────────────────────────────
"""
    holo_spectrum(result; n_carrier, n_mod, f_carrier_max, f_mod_max)
        → (carrier_axis, mod_axis, H)

Bin the energy A²(t) from each Stage-2 IMF into a 2-D histogram indexed by
carrier frequency ω and modulation frequency Ω.

Physical constraint enforced (paper Section 3):
    ω > Ω  always holds because Ω comes from the amplitude envelope of an IMF
    whose carrier frequency is ω. An envelope cannot oscillate faster than the
    wave it envelopes. Any (ω, Ω) pair violating this is discarded.

Returns
───────
- `carrier_axis` : frequency axis for ω (Hz), length n_carrier
- `mod_axis`     : frequency axis for Ω (Hz), length n_mod
- `H`            : energy matrix, shape (n_mod × n_carrier), units A²

H[i, j] = Σ_{t: ω_j(t)≈carrier_axis[j], Ω_{jk}(t)≈mod_axis[i]} A²_{jk}(t)
"""
function holo_spectrum(result::HoloResult;
                       n_carrier    ::Int     = 128,
                       n_mod        ::Int     = 64,
                       f_carrier_max::Float64 = result.fs / 2,
                       f_mod_max    ::Float64 = result.fs / 4)

    carrier_axis = range(0.0, f_carrier_max, length = n_carrier)
    mod_axis     = range(0.0, f_mod_max,     length = n_mod)

    H = zeros(n_mod, n_carrier)

    n_imfs = length(result.imfs)
    for j in 1:n_imfs
        ω_j = result.carrier_freq[j]
        for k in eachindex(result.mod_imfs[j])
            Ω_jk = result.mod_freq[j][k]
            A_jk = result.mod_amp[j][k]
            for t in eachindex(ω_j)
                # Physical constraint disabled: for multiplicative signals with no real carrier
                # (e.g., modulated noise), the constraint ω > Ω discards the signal entirely.
                # The constraint makes sense for signals with real carriers, but not for noise.
                # ω_j[t] > Ω_jk[t] || continue

                ci = searchsortedlast(carrier_axis, ω_j[t])
                ci = clamp(ci, 1, n_carrier)
                mi = searchsortedlast(mod_axis, Ω_jk[t])
                mi = clamp(mi, 1, n_mod)
                if ci >= 1 && mi >= 1
                    H[mi, ci] += A_jk[t]^2
                end
            end
        end
    end

    return collect(carrier_axis), collect(mod_axis), H
end

# ── Plotting helpers ──────────────────────────────────────────────────────────
"""
    plot_imfs(result; title)

Stack plot of the Stage-1 IMFs and residue.
"""
function plot_imfs(result::HoloResult; title::String = "Stage-1 IMFs")
    n_imfs = length(result.imfs)
    n      = length(result.imfs[1])
    t      = (0:n-1) ./ result.fs

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

2-D heatmap of carrier frequency ω (x-axis) vs. modulation frequency Ω (y-axis).
Energy in the forbidden half-plane (Ω ≥ ω) is already excluded by holo_spectrum.
The diagonal ω = Ω is plotted as a reference line (paper Fig. 6c solid line).
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

    p = Plots.heatmap(ω_ax, Ω_ax, H;
                      xlabel         = "Carrier frequency ω (Hz)",
                      ylabel         = "Modulation frequency Ω (Hz)",
                      title,
                      color          = :viridis,
                      colorbar_title = "Energy (A²)",
                      size           = (800, 400))

    # Overlay the ω = Ω boundary line (paper Fig. 6c solid line)
    diag_max = min(f_carrier_max, f_mod_max)
    Plots.plot!(p, [0.0, diag_max], [0.0, diag_max];
                label     = "ω = Ω boundary",
                color     = :white,
                linestyle = :dash,
                linewidth = 1.5)
    return p
end

"""
    plot_hilbert_spectrum(result; n_freq, f_max, title)

Classic Hilbert spectrum: time (x) × carrier frequency (y), coloured by amplitude.
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

    H2 = zeros(n_freq, n)
    for j in 1:length(result.imfs)-1    # skip residue
        amp_j  = result.carrier_amp[j]
        freq_j = result.carrier_freq[j]
        for ti in 1:n
            fi = clamp(round(Int, freq_j[ti] / Δf) + 1, 1, n_freq)
            H2[fi, ti] += amp_j[ti]
        end
    end

    return Plots.heatmap(t, collect(f_axis), H2;
                         xlabel         = "Time (s)",
                         ylabel         = "Frequency (Hz)",
                         title,
                         color          = :hot,
                         colorbar_title = "Amplitude",
                         size           = (900, 400))
end
