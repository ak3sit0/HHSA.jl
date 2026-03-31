# ─────────────────────────────────────────────────────────────────────────────
# HHSA.jl  –  Holo-Hilbert Spectral Analysis in pure Julia
#
# Two external dependencies only:
#   • DSP.jl   – Hilbert transform + phase unwrapping (well-maintained, stable)
#   • Plots.jl – visualisation
#
# No EmpiricalModeDecomposition.jl: the EMD sifting is implemented here in
# pure Julia (no Dierckx, no BinaryProvider, works on Julia ≥ 1.6).
# ─────────────────────────────────────────────────────────────────────────────
module HHSA

using DSP
using Plots

include("emd.jl")
include("holo_spectrum.jl")

export emd,
       hhsa, HoloResult,
       holo_spectrum,
       plot_imfs,
       plot_holo_spectrum,
       plot_hilbert_spectrum

end # module
