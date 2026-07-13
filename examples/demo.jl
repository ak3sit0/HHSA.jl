# ─────────────────────────────────────────────────────────────────────────────
# demo.jl  –  HHSA on a synthetic AM signal
#
# Signal construction (two AM components + white noise):
#
#   x(t) = [1 + 0.8·sin(2π·f_m1·t)] · sin(2π·f_c1·t)   (60 Hz carrier, 6 Hz AM)
#         + [1 + 0.5·sin(2π·f_m2·t)] · sin(2π·f_c2·t)   (20 Hz carrier, 2 Hz AM)
#         + 0.1·ε(t)                                      (white Gaussian noise)
#
# With fs=512 Hz, T=4 s → N=2048 samples.
# A correct HHSA should place energy near (ω=60, Ω=6) and (ω=20, Ω=2) in the HHS.
# ─────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using HHSA
using Plots
using Random
gr()

# ── 1. Signal parameters ──────────────────────────────────────────────────────
const fs   = 512.0
const T    = 4.0
const t    = range(0.0, T - 1/fs; step = 1/fs)
const N    = length(t)

const f_c1, f_m1 = 60.0,  6.0
const f_c2, f_m2 = 20.0,  2.0

# ── 2. Generate signal ────────────────────────────────────────────────────────
# FIX: noise must be a length-N vector, not a scalar broadcast.
# The original code used `0.1 * randn()` inside @., which called randn() once
# and broadcast the same scalar to every sample — a constant DC offset, not noise.
rng   = MersenneTwister(42)          # fixed seed → reproducible output
noise = 0.1 .* randn(rng, N)        # ← correct: one independent draw per sample

carriers = @. (1 + 0.8*sin(2π*f_m1*t)) * sin(2π*f_c1*t) +
              (1 + 0.5*sin(2π*f_m2*t)) * sin(2π*f_c2*t)
x = Float64.(carriers) .+ noise

println("Signal: N=$(N) samples  ($(T)s @ $(fs) Hz)")
println("Components: ($f_c1 Hz, $f_m1 Hz AM) + ($f_c2 Hz, $f_m2 Hz AM) + noise")

# ── 3. Run HHSA ───────────────────────────────────────────────────────────────
println("\nRunning HHSA …")
@time result = hhsa(x, fs;
                    max_imfs     = 8,
                    max_sift     = 200,
                    sd_threshold = 0.1)

println("Stage-1 IMFs: $(length(result.imfs))  (last is residue)")

# ── 4. Save figures ───────────────────────────────────────────────────────────
mkpath(joinpath(@__DIR__, "..", "output"))
out(name) = joinpath(@__DIR__, "..", "output", name)

p0 = Plots.plot(collect(t), x;
                label     = "x(t)",
                xlabel    = "Time (s)",
                ylabel    = "Amplitude",
                title     = "Synthetic AM signal",
                linewidth = 1.0,
                size      = (900, 250))
savefig(p0, out("01_signal.png"))
println("Saved 01_signal.png")

p1 = plot_imfs(result; title = "Stage-1 IMFs — EMD of x(t)")
savefig(p1, out("02_stage1_imfs.png"))
println("Saved 02_stage1_imfs.png")

p2 = plot_hilbert_spectrum(result;
                           n_freq = 200,
                           f_max  = 80.0,
                           title  = "Hilbert spectrum (time × carrier frequency)")
savefig(p2, out("03_hilbert_spectrum.png"))
println("Saved 03_hilbert_spectrum.png")

# HHS: expect two blobs near (ω=60, Ω=6) and (ω=20, Ω=2)
p3 = plot_holo_spectrum(result;
                        n_carrier     = 200,
                        n_mod         = 100,
                        f_carrier_max = 80.0,
                        f_mod_max     = 20.0,
                        title         = "Holo-Hilbert Spectrum — carrier ω vs. modulation Ω")
savefig(p3, out("04_holo_spectrum.png"))
println("Saved 04_holo_spectrum.png")

println("\nDone. All figures in output/")
