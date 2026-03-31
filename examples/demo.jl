# ─────────────────────────────────────────────────────────────────────────────
# demo.jl  –  HHSA on a synthetic AM signal
#
# Signal construction (two AM components):
#   x(t) = [1 + 0.8·sin(2π·f_m1·t)] · sin(2π·f_c1·t)     (fast carrier, slow AM)
#         + [1 + 0.5·sin(2π·f_m2·t)] · sin(2π·f_c2·t)     (mid carrier, mid AM)
#         + 0.1·randn()                                      (noise)
#
# With fs=512 Hz, fc1=60 Hz, fm1=6 Hz, fc2=20 Hz, fm2=2 Hz
# A perfect HHSA should reveal peaks at (ω=60,Ω=6) and (ω=20,Ω=2).
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Load the package (activate the project first) ────────────────────────
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))   # activate HHSA_Julia project
Pkg.instantiate()                        # install DSP.jl + Plots.jl if needed

using HHSA
using Plots
gr()   # fast, non-interactive backend; swap to plotlyjs() for interactive 3D

# ── 2. Generate synthetic signal ─────────────────────────────────────────────
fs   = 512.0        # Hz
T    = 4.0          # seconds
t    = 0:1/fs:T-1/fs
N    = length(t)

f_c1, f_m1 = 60.0, 6.0    # carrier 1, its AM frequency
f_c2, f_m2 = 20.0, 2.0    # carrier 2, its AM frequency

x = @. (1 + 0.8*sin(2π*f_m1*t)) * sin(2π*f_c1*t) +
       (1 + 0.5*sin(2π*f_m2*t)) * sin(2π*f_c2*t) +
       0.1 * randn()

println("Signal length: $N samples  ($(T)s @ $(fs)Hz)")

# ── 3. Run HHSA ───────────────────────────────────────────────────────────────
println("Running HHSA … (may take a few seconds)")
@time result = hhsa(Float64.(x), fs;
                    max_imfs      = 8,
                    max_sift      = 200,
                    sd_threshold  = 0.1)

println("Stage-1 IMFs found: ", length(result.imfs), " (last is residue)")

# ── 4. Plots ──────────────────────────────────────────────────────────────────
mkpath(joinpath(@__DIR__, "..", "output"))
out = path -> joinpath(@__DIR__, "..", "output", path)

# 4a. Raw signal
p0 = plot(collect(t), x;
          label = "x(t)", xlabel = "Time (s)", ylabel = "Amplitude",
          title = "Synthetic AM signal", linewidth = 1.5, size = (900, 250))
savefig(p0, out("01_signal.png"))
println("Saved 01_signal.png")

# 4b. Stage-1 IMF decomposition
p1 = plot_imfs(result; title = "Stage-1 IMFs (EMD of x(t))")
savefig(p1, out("02_stage1_imfs.png"))
println("Saved 02_stage1_imfs.png")

# 4c. Classic Hilbert spectrum (time × carrier frequency)
p2 = plot_hilbert_spectrum(result;
                           n_freq = 200, f_max = 80.0,
                           title  = "Hilbert Spectrum  (Stage 1)")
savefig(p2, out("03_hilbert_spectrum.png"))
println("Saved 03_hilbert_spectrum.png")

# 4d. Holo-Hilbert spectrum (carrier ω × modulation Ω)
#     Expect peaks near (ω=60, Ω=6) and (ω=20, Ω=2)
p3 = plot_holo_spectrum(result;
                        n_carrier     = 200,
                        n_mod         = 100,
                        f_carrier_max = 80.0,
                        f_mod_max     = 20.0,
                        title         = "Holo-Hilbert Spectrum  (ω vs Ω)")
savefig(p3, out("04_holo_spectrum.png"))
println("Saved 04_holo_spectrum.png")

println("\nDone!  All figures written to output/")
