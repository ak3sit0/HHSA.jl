# ─────────────────────────────────────────────────────────────────────────────
# test/runtests.jl  –  HHSA.jl test suite
#
# Every @testset is annotated with the paper section / equation / figure
# that motivates it, so failures point directly to a theoretical claim.
#
# Reference: Huang et al. (2016) Phil. Trans. R. Soc. A 374, 20150206
# ─────────────────────────────────────────────────────────────────────────────

using Test
using Statistics
using Random
using DSP
using HHSA   # rename to HoloHilbert once the package is restructured

# ── Test utilities ─────────────────────────────────────────────────────────────

"""
    peak_coords(H, ω_ax, Ω_ax) → (peak_ω, peak_Ω)

Return the (carrier, modulation) frequency pair at the global energy maximum
of the Holo-Hilbert spectrum H.
"""
function peak_coords(H::Matrix, ω_ax::Vector, Ω_ax::Vector)
    idx = argmax(H)
    return ω_ax[idx[2]], Ω_ax[idx[1]]
end

"""
    forbidden_fraction(H, ω_ax, Ω_ax) → Float64

Fraction of total HHS energy in the physically forbidden half-plane Ω ≥ ω.
Should be ≈ 0 after the constraint fix.  Paper Section 3: 'ω > Ω always valid'.
"""
function forbidden_fraction(H::Matrix, ω_ax::Vector, Ω_ax::Vector)
    total = sum(H)
    iszero(total) && return 0.0
    bad = 0.0
    for (mi, Ω) in enumerate(Ω_ax), (ci, ω) in enumerate(ω_ax)
        Ω >= ω && (bad += H[mi, ci])
    end
    return bad / total
end

"""
    window_energy(H, ω_ax, Ω_ax, ω_lo, ω_hi, Ω_lo, Ω_hi) → Float64

Sum of HHS energy inside a rectangular frequency window.
Used to verify that a known AM component deposits energy in the right region.
"""
function window_energy(H, ω_ax, Ω_ax, ω_lo, ω_hi, Ω_lo, Ω_hi)
    ω_mask = (ω_ax .>= ω_lo) .& (ω_ax .<= ω_hi)
    Ω_mask = (Ω_ax .>= Ω_lo) .& (Ω_ax .<= Ω_hi)
    return sum(H[Ω_mask, ω_mask])
end

# ══════════════════════════════════════════════════════════════════════════════
@testset "HHSA.jl" begin
# ══════════════════════════════════════════════════════════════════════════════

# ── Group 1: EMD correctness ──────────────────────────────────────────────────
@testset "EMD" begin

    @testset "perfect reconstruction  [paper eq. 1.3: x(t) = Σcⱼ(t) + residue]" begin
        # The defining property of EMD: the sum of all returned vectors
        # (IMFs + residue) must equal the original signal to machine precision.
        # This is not approximate — it must be exact.
        rng = MersenneTwister(42)
        fs  = 512.0
        N   = round(Int, 2 * fs)
        t   = (0:N-1) ./ fs
        x   = sin.(2π * 30 .* t) .+ 0.5 .* sin.(2π * 8 .* t) .+
              0.1 .* randn(rng, N)

        imfs = emd(Float64.(x))
        reconstructed = sum(imfs)          # Vector + Vector = element-wise sum

        @test maximum(abs.(reconstructed .- x)) < 1e-10   # machine precision
    end

    @testset "monotonic residue  [stopping rule: < 3 total extrema in residue]" begin
        # EMD stops extracting IMFs when the residue has fewer than 3 extrema,
        # meaning it is monotonic (a trend with no oscillation to extract).
        rng  = MersenneTwister(17)
        x    = randn(rng, 600)
        imfs = emd(x)
        res  = imfs[end]
        # Count extrema manually (avoid calling private _find_extrema)
        n_ext = count(i -> (res[i] > res[i-1] && res[i] > res[i+1]) ||
                           (res[i] < res[i-1] && res[i] < res[i+1]),
                      2:length(res)-1)
        @test n_ext < 3
    end

    @testset "constant signal  [degenerate: no oscillation, returns as residue]" begin
        x    = fill(3.14, 200)
        imfs = emd(x)
        @test sum(imfs) ≈ x  atol=1e-10
    end

    @testset "monotonic signal  [degenerate: EMD returns immediately]" begin
        x    = Float64.(1:150)          # linearly increasing — nothing to sift
        imfs = emd(x)
        @test sum(imfs) ≈ x  atol=1e-10
    end

    @testset "single sinusoid yields ≥ 2 vectors (1 IMF + residue)" begin
        t    = (0:511) ./ 512.0
        x    = sin.(2π * 40.0 .* t)
        imfs = emd(Float64.(x))
        @test length(imfs) >= 2
    end

    @testset "multi-component: more IMFs than single sinusoid" begin
        # Two well-separated sinusoids should produce ≥ 2 non-trivial IMFs.
        t    = (0:1023) ./ 512.0
        x    = sin.(2π * 60.0 .* t) .+ sin.(2π * 10.0 .* t)
        imfs_multi  = emd(Float64.(x))
        imfs_single = emd(sin.(2π * 60.0 .* t))
        @test length(imfs_multi) > length(imfs_single)
    end

end  # EMD

# ── Group 2: Hilbert transform / instantaneous parameters ─────────────────────
@testset "Instantaneous parameters" begin

    @testset "amplitude of pure sinusoid  [analytic signal: |z(t)| = A]" begin
        # For x(t) = A·sin(2πft), the analytic signal is z(t) = A·e^{iωt}.
        # Therefore |z(t)| = A at every point (Bedrosian, 1963; paper Section 5).
        # We exclude 5% at each edge because the FFT-based Hilbert transform
        # has boundary artefacts for non-periodic signals.
        fs   = 1024.0
        f    = 50.0
        A    = 2.5
        t    = (0:round(Int, 2fs)-1) ./ fs
        x    = A .* sin.(2π * f .* t)
        amp, _ = instantaneous_params(Float64.(x), fs)

        trim = round(Int, 0.05 * length(x))
        @test mean(amp[trim:end-trim]) ≈ A  rtol=0.02    # within 2%
        @test  std(amp[trim:end-trim])     < 0.05 * A    # low variance too
    end

    @testset "frequency of pure sinusoid  [ω(t) = dθ/dt = 2πf]" begin
        # For x(t) = sin(2πft), the unwrapped phase grows linearly,
        # so its derivative is constant: ω(t) = f (Hz) everywhere.
        fs = 1024.0
        f  = 30.0
        t  = (0:round(Int, 2fs)-1) ./ fs
        x  = sin.(2π * f .* t)
        _, freq = instantaneous_params(Float64.(x), fs)

        trim = round(Int, 0.05 * length(x))
        @test mean(freq[trim:end-trim]) ≈ f  rtol=0.02   # within 2%
    end

    @testset "frequency is non-negative  [clamp of dθ/dt at edges]" begin
        # Edge artefacts in the Hilbert transform can produce small negative
        # instantaneous frequencies.  The implementation clamps to zero.
        rng  = MersenneTwister(99)
        x    = randn(rng, 512)
        _, freq = instantaneous_params(x, 512.0)
        @test all(freq .>= 0.0)
    end

end  # Instantaneous parameters

# ── Group 3: Physical constraint ω > Ω ───────────────────────────────────────
@testset "Physical constraint: ω > Ω in HHS" begin

    @testset "no energy in forbidden half-plane  [paper Section 3]" begin
        # Paper Section 3: 'The high-dimensional spectral representation would
        # only fill half of the ω–Ω space, for Ω is derived from the slowly
        # varying amplitude; therefore, the condition ω > Ω is always valid.'
        #
        # Before the fix, numerical noise from EMD boundary effects deposited
        # spurious energy at Ω ≥ ω.  After the fix, this region must be empty.
        t = (0:2047) ./ 512.0
        x = @. (1 + 0.8sin(2π * 6t)) * sin(2π * 60t)

        result        = hhsa(Float64.(x), 512.0; max_imfs=6)
        ω_ax, Ω_ax, H = holo_spectrum(result;
                                       n_carrier     = 128,
                                       n_mod         = 64,
                                       f_carrier_max = 200.0,
                                       f_mod_max     = 60.0)

        @test forbidden_fraction(H, ω_ax, Ω_ax) < 0.01   # < 1% in forbidden zone
    end

    @testset "diagonal boundary in HHS is respected for noisy signal" begin
        rng = MersenneTwister(11)
        t   = (0:2047) ./ 512.0
        x   = @. sin(2π * 40t) + 0.3randn()
        x   = Float64.(x) .+ 0.3 .* randn(rng, length(t))

        result        = hhsa(x, 512.0; max_imfs=6)
        ω_ax, Ω_ax, H = holo_spectrum(result; n_carrier=128, n_mod=64,
                                               f_carrier_max=150.0, f_mod_max=50.0)
        @test forbidden_fraction(H, ω_ax, Ω_ax) < 0.01
    end

end  # Physical constraint

# ── Group 4: HHS peak localization ───────────────────────────────────────────
@testset "HHS peak localization" begin

    @testset "single pure AM — peak at (f_c, f_m)  [paper Fig. 6c concept]" begin
        # x(t) = (1 + 0.8·sin(2π·f_m·t))·sin(2π·f_c·t)
        #
        # This is the cleanest possible test: no noise, known ground truth,
        # modulation index < 1 so the envelope is always positive (no rectification).
        # The HHS global peak must fall near the input frequencies.
        fs  = 512.0
        f_c = 60.0
        f_m = 6.0
        t   = (0:round(Int, 4fs)-1) ./ fs
        x   = @. (1 + 0.8sin(2π * f_m * t)) * sin(2π * f_c * t)

        result        = hhsa(Float64.(x), fs; max_imfs=6, sd_threshold=0.05)
        ω_ax, Ω_ax, H = holo_spectrum(result;
                                       n_carrier     = 256,
                                       n_mod         = 128,
                                       f_carrier_max = 150.0,
                                       f_mod_max     = 30.0)
        peak_ω, peak_Ω = peak_coords(H, ω_ax, Ω_ax)

        @test abs(peak_ω - f_c) < 10.0   # carrier within 10 Hz of f_c = 60 Hz
        @test abs(peak_Ω - f_m) <  3.0   # modulation within 3 Hz of f_m = 6 Hz
    end

    @testset "two AM components — energy in both expected windows" begin
        # x = AM(60 Hz, 6 Hz) + AM(20 Hz, 2 Hz)  (the demo signal, no noise)
        #
        # We don't require exact peak coordinates (two components interact in EMD),
        # but a meaningful fraction of the total energy must appear in windows
        # surrounding each expected (ω, Ω) coordinate.
        rng = MersenneTwister(42)
        fs  = 512.0
        t   = (0:round(Int, 4fs)-1) ./ fs
        N   = length(t)
        x   = @. (1 + 0.8sin(2π * 6t)) * sin(2π * 60t) +
                 (1 + 0.5sin(2π * 2t)) * sin(2π * 20t)
        x   = Float64.(x) .+ 0.05 .* randn(rng, N)

        result        = hhsa(x, fs; max_imfs=8, sd_threshold=0.05)
        ω_ax, Ω_ax, H = holo_spectrum(result;
                                       n_carrier     = 256,
                                       n_mod         = 128,
                                       f_carrier_max = 150.0,
                                       f_mod_max     = 20.0)
        total = sum(H)

        # Window centred on (ω=60, Ω=6) — ±20 Hz carrier, ±4 Hz modulation
        e1 = window_energy(H, ω_ax, Ω_ax, 40.0, 80.0, 2.0, 10.0)
        # Window centred on (ω=20, Ω=2) — ±10 Hz carrier, ±2 Hz modulation
        e2 = window_energy(H, ω_ax, Ω_ax, 10.0, 30.0, 0.5,  4.0)

        @test e1 / total > 0.10   # ≥10% of energy near the 60 Hz component
        @test e2 / total > 0.05   # ≥5%  of energy near the 20 Hz component
    end

end  # HHS peak localization

# ── Group 5: Paper Figures 2, 3, 6c core demonstration ───────────────────────
@testset "Paper core demonstration  [Figs 2, 3, 6c]" begin

    # Shared signal construction — matches the paper's Section 2 setup.
    # fs = 1 Hz, N = 1000 samples (= 1000 seconds), f_sine ≈ 0.01 Hz.
    rng    = MersenneTwister(314)
    fs     = 1.0
    N      = 1000
    t      = (0:N-1) ./ fs
    f_sine = 0.01     # period = 100 s, ~10 cycles visible in Fig. 1a

    noise  = randn(rng, N)
    s_wave = sin.(2π * f_sine .* t)

    x_add  = s_wave .+ noise     # paper eq. 2.6, upper:  x₁(t) = cos A + Σ cos θⱼ
    x_mult = s_wave .* noise     # paper eq. 2.6, lower:  x₂(t) = cos A · Σ cos θⱼ

    @testset "Fig 3a: Fourier DOES detect sine in additive signal" begin
        # The Fourier spectrum of the additive signal (x_add) must show a clear
        # peak at f_sine, because the sine wave simply adds to the noise floor.
        # This is the 'control' case — if this fails, something is wrong with
        # the signal construction, not with HHSA.
        P_add  = abs2.(fft(x_add))[1:N÷2+1] ./ N
        freqs  = (0:N÷2) ./ (N * (1/fs))     # Hz

        bin       = argmin(abs.(freqs .- f_sine))
        neighbors = vcat(P_add[max(1, bin-10):bin-1],
                         P_add[bin+1:min(end, bin+10)])

        @test P_add[bin] > 3.0 * mean(neighbors)   # clear peak at f_sine
    end

    @testset "Fig 3b: Fourier CANNOT detect sine in multiplicative signal" begin
        # For x_mult = sin(Ωt)·noise, the trigonometric identity
        # cos(A)·cos(B) = ½[cos(A+B) + cos(A-B)] spreads the sine's energy
        # symmetrically around every noise component.  The original frequency
        # Ω disappears from the spectrum entirely.  Paper eq. 2.3–2.6.
        P_mult = abs2.(fft(x_mult))[1:N÷2+1] ./ N
        freqs  = (0:N÷2) ./ (N * (1/fs))

        bin       = argmin(abs.(freqs .- f_sine))
        neighbors = vcat(P_mult[max(1, bin-10):bin-1],
                         P_mult[bin+1:min(end, bin+10)])

        # The sine peak should NOT stand out — it should be buried in noise.
        # We allow up to 3× the local mean before calling it a visible peak.
        @test P_mult[bin] < 3.0 * mean(neighbors)
    end

    @testset "Fig 6c: HHSA finds modulation frequency in multiplicative signal" begin
        # After Fourier fails (previous test), HHSA recovers the modulation.
        #
        # Mechanism (paper Fig. 6c caption):
        # For full modulation x = cos(Ωt)·noise, the amplitude envelope of each
        # noise IMF oscillates as |cos(Ωt)|, which has frequency 2Ω (rectified
        # cosine).  'Because of the full modulation, this frequency is twice the
        # value of the sine wave.'
        #
        # Target band: Ω_target = 2·f_sine, tolerance ±factor of 2.
        result = hhsa(x_mult, fs;
                      max_imfs     = 8,
                      max_sift     = 80,      # faster for CI
                      sd_threshold = 0.15)

        ω_ax, Ω_ax, H = holo_spectrum(result;
                                       n_carrier     = 128,
                                       n_mod         = 64,
                                       f_carrier_max = 0.5,
                                       f_mod_max     = 0.08)

        Ω_target = 2 * f_sine              # = 0.02 Hz
        Ω_lo     = Ω_target * 0.5          # 0.01 Hz
        Ω_hi     = Ω_target * 3.0          # 0.06 Hz  (generous: EMD is adaptive)
        Ω_mask   = (Ω_ax .>= Ω_lo) .& (Ω_ax .<= Ω_hi)

        frac = sum(H[Ω_mask, :]) / sum(H)
        @test frac > 0.20   # ≥20% of energy in the 2·f_sine band
    end

end  # Paper core demonstration

# ══════════════════════════════════════════════════════════════════════════════
end  # @testset "HHSA.jl"
# ══════════════════════════════════════════════════════════════════════════════
