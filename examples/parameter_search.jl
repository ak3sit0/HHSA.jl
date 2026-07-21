# ─────────────────────────────────────────────────────────────────────────────
# Parameter search: find optimal settings for multiplicative signal detection
# ─────────────────────────────────────────────────────────────────────────────

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using HHSA
using DSP
using Random

Random.seed!(42)

# Parameter ranges to explore
durations = [200.0, 300.0, 500.0, 1000.0]          # seconds
amplitudes = [0.8, 1.0, 1.5, 2.0]                  # modulation strength
frequencies = [0.3, 0.5, 1.0]                      # Hz

fs = 100.0
emd_params = (max_imfs=15, max_sift=300, sd_threshold=0.05)

# Results table
println("="^80)
println("PARAMETER SEARCH: Multiplicative Modulation Detection")
println("="^80)
println()
println(lpad("T(s)", 8) * lpad("Amp", 8) * lpad("f_mod(Hz)", 10) * lpad("N_samples", 12) *
        lpad("Peak Energy", 14) * lpad("Peak Ratio", 12))
println("-"^80)

results = []

for T in durations
    for amp in amplitudes
        for f_mod in frequencies
            # Generate signal
            t = 0:1/fs:(T - 1/fs)
            N = length(t)

            envelope = 1.0 .+ amp .* sin.(2π * f_mod .* t)
            x_mult = envelope .* randn(N)

            # Run HHSA
            result = hhsa(x_mult, fs; emd_params...)

            # Compute holo-spectrum
            ω_ax, Ω_ax, H = holo_spectrum(result; n_carrier=150, n_mod=100,
                                          f_carrier_max=30.0, f_mod_max=10.0)

            # Measure: energy concentration in expected region
            # Expected region: ω ∈ [0, 5], Ω ∈ [f_mod - 0.2, f_mod + 0.2]
            ω_mask = (ω_ax .>= 0) .& (ω_ax .<= 5.0)
            Ω_center = f_mod
            Ω_mask = (Ω_ax .>= Ω_center - 0.2) .& (Ω_ax .<= Ω_center + 0.2)

            peak_energy = sum(H[Ω_mask, ω_mask])
            total_energy = sum(H)
            peak_ratio = peak_energy / total_energy

            # Print result
            println(lpad(Int(T), 8) * lpad(amp, 8) * lpad(f_mod, 10) *
                    lpad(N, 12) * lpad(round(peak_energy, digits=2), 14) *
                    lpad(round(peak_ratio, digits=4), 12))

            push!(results, (T=T, amp=amp, f_mod=f_mod, N=N,
                           peak_energy=peak_energy, peak_ratio=peak_ratio))
        end
    end
end

println("-"^80)
println()

# Sort by peak_ratio and show top 5
sort!(results, by = x -> x.peak_ratio, rev=true)
println("TOP 5 CONFIGURATIONS (by peak concentration):")
println()
for (i, r) in enumerate(results[1:5])
    println("$i. T=$(Int(r.T))s, Amp=$(r.amp), f_mod=$(r.f_mod)Hz")
    println("   → Peak ratio: $(round(r.peak_ratio, digits=4)) ($(round(r.peak_ratio*100, digits=2))%)")
    println()
end

# Show best configuration
best = results[1]
println("="^80)
println("RECOMMENDED CONFIGURATION:")
println("="^80)
println("Duration (T):              $(Int(best.T)) seconds")
println("Modulation amplitude:      $(best.amp)")
println("Modulation frequency:      $(best.f_mod) Hz")
println("Number of samples:         $(best.N)")
println("Peak energy concentration: $(round(best.peak_ratio * 100, digits=2))%")
println("="^80)
