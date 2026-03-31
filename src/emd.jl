# ─────────────────────────────────────────────────────────────────────────────
# emd.jl  –  Empirical Mode Decomposition (pure Julia, no external spline libs)
#
# Reference: Huang et al. (1998) Proc. R. Soc. Lond. A 454, 903-995
# ─────────────────────────────────────────────────────────────────────────────

# ── Natural cubic spline ────────────────────────────────────────────────────
# Solves the tridiagonal system that gives second-derivative coefficients M_i
# (natural BCs: M_0 = M_{n-1} = 0), then evaluates at arbitrary x_eval.
function _natural_cubic_spline(xs::Vector{Float64},
                                ys::Vector{Float64},
                                x_eval::Vector{Float64})
    n = length(xs)
    @assert n == length(ys)
    @assert issorted(xs) "Spline knots must be sorted"

    # Degenerate cases
    n == 1 && return fill(ys[1], length(x_eval))
    n == 2 && return ys[1] .+ (x_eval .- xs[1]) .* (ys[2] - ys[1]) / (xs[2] - xs[1])

    h = diff(xs)          # h[i] = xs[i+1] - xs[i],  length n-1
    m = n - 2             # number of interior knots

    # Right-hand side: 6 * divided difference of order 2
    rhs = [6.0 * ((ys[i+2] - ys[i+1]) / h[i+1] - (ys[i+1] - ys[i]) / h[i])
           for i in 1:m]

    # Thomas algorithm for tridiagonal system
    diag = [2.0 * (h[i] + h[i+1]) for i in 1:m]
    sup  = [h[i+1] for i in 1:m-1]   # super-diagonal
    sub  = [h[i+1] for i in 1:m-1]   # sub-diagonal (same values for this system)

    c = copy(diag)
    r = copy(rhs)
    for i in 2:m
        factor = sub[i-1] / c[i-1]
        c[i] -= factor * sup[i-1]
        r[i] -= factor * r[i-1]
    end

    Mi = zeros(m)
    Mi[end] = r[end] / c[end]
    for i in m-1:-1:1
        Mi[i] = (r[i] - sup[i] * Mi[i+1]) / c[i]
    end
    M = vcat(0.0, Mi, 0.0)   # pad with natural-BC zeros

    # Evaluate spline at each requested point
    result = Vector{Float64}(undef, length(x_eval))
    for (k, x) in enumerate(x_eval)
        # Clamp to the knot range to avoid out-of-bounds extrapolation
        i = clamp(searchsortedlast(xs, x), 1, n - 1)
        hi = h[i]
        u  = xs[i+1] - x
        v  = x - xs[i]
        result[k] = (M[i] * u^3 + M[i+1] * v^3) / (6hi) +
                    (ys[i] / hi - M[i] * hi / 6) * u +
                    (ys[i+1] / hi - M[i+1] * hi / 6) * v
    end
    return result
end

# ── Extrema detection ───────────────────────────────────────────────────────
function _find_extrema(signal::Vector{Float64})
    n = length(signal)
    max_idx = [i for i in 2:n-1 if signal[i] > signal[i-1] && signal[i] > signal[i+1]]
    min_idx = [i for i in 2:n-1 if signal[i] < signal[i-1] && signal[i] < signal[i+1]]
    return max_idx, min_idx
end

# ── Envelope via boundary-reflected spline ──────────────────────────────────
# Uses mirror reflection of the innermost extrema to reduce the end effect
# (Rilling et al., 2003 – IEEE-EURASIP NSIP).
function _build_envelope(signal::Vector{Float64},
                         extrema_idx::Vector{Int},
                         t::Vector{Float64})
    isempty(extrema_idx) && return fill(sum(signal) / length(signal), length(t))

    ext_t = t[extrema_idx]
    ext_v = signal[extrema_idx]

    # Reflect leftmost extremum past the left boundary
    if ext_t[1] > t[1]
        ext_t = vcat(2t[1] - ext_t[1], ext_t)
        ext_v = vcat(ext_v[1],          ext_v)
    end
    # Reflect rightmost extremum past the right boundary
    if ext_t[end] < t[end]
        ext_t = vcat(ext_t, 2t[end] - ext_t[end])
        ext_v = vcat(ext_v, ext_v[end])
    end

    return _natural_cubic_spline(ext_t, ext_v, t)
end

# ── Single sifting iteration ─────────────────────────────────────────────────
function _sift(h::Vector{Float64}, t::Vector{Float64})
    max_idx, min_idx = _find_extrema(h)
    (length(max_idx) < 2 || length(min_idx) < 2) && return h, false  # can't sift

    upper    = _build_envelope(h, max_idx, t)
    lower    = _build_envelope(h, min_idx, t)
    mean_env = (upper .+ lower) ./ 2
    return h .- mean_env, true
end

# ── EMD ──────────────────────────────────────────────────────────────────────
"""
    emd(signal; max_imfs, max_sift, sd_threshold) → Vector{Vector{Float64}}

Decompose `signal` into Intrinsic Mode Functions (IMFs) plus a trend residue.

# Keyword arguments
- `max_imfs`      : maximum number of IMFs to extract  (default 10)
- `max_sift`      : maximum sifting iterations per IMF (default 200)
- `sd_threshold`  : Cauchy-style stopping criterion    (default 0.1)

The last element of the returned vector is always the residue (trend).

# Reference
Huang et al. (1998), Proc. R. Soc. Lond. A 454, 903-995.
"""
function emd(signal::Vector{Float64};
             max_imfs::Int      = 10,
             max_sift::Int      = 200,
             sd_threshold::Float64 = 0.1)

    n  = length(signal)
    t  = Float64.(1:n)
    x  = copy(signal)
    imfs = Vector{Vector{Float64}}()

    for _ in 1:max_imfs
        h = copy(x)

        for _ in 1:max_sift
            h_new, ok = _sift(h, t)
            ok || break

            # Normalised SD criterion (Huang 1998, eq. 5.5)
            sd = sum((h .- h_new) .^ 2) / (sum(h .^ 2) + eps())
            h = h_new
            sd < sd_threshold && break
        end

        push!(imfs, h)
        x = x .- h

        # Stop when the residue is monotonic (< 3 total extrema)
        max_r, min_r = _find_extrema(x)
        (length(max_r) + length(min_r)) < 3 && break
    end

    push!(imfs, x)   # residue / trend
    return imfs
end
