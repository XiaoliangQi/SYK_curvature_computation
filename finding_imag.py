import numpy as np

def imag_f_from_even_F(omega, F, use_second_deriv_at_zero=True):
    """
    Given samples of an even F(ω) on (possibly) non-uniform ω, return Im f(ω)
    where f(ω) = F(0) + ω^2 ( g(ω) + i H[g](ω) ), g(ω)=(F(ω)-F(0))/ω^2.
    Sign convention: H[g](ω) = (1/π) PV ∫ g(ω')/(ω-ω') dω'.

    Parameters
    ----------
    omega : (N,) array_like
    F     : (N,) array_like
    use_second_deriv_at_zero : bool
        If True and ω=0 is present, set g(0)=0.5*F''(0) via a symmetric quadratic fit.

    Returns
    -------
    Im_f : (N,) ndarray
        Imaginary part on the provided ω points.
    """
    omega = np.asarray(omega, float)
    F = np.asarray(F, float)
    if omega.shape != F.shape:
        raise ValueError("omega and F must have the same shape.")

    # Sort for stability, remember permutation
    idx = np.argsort(omega)
    om = omega[idx]
    Fm = F[idx]

    # Build g(ω) = (F(ω)-F(0))/ω^2 with careful ω=0 handling
    F0 = Fm[np.argmin(np.abs(om))]  # approximate F(0)
    g = np.empty_like(Fm)
    nonzero = om != 0.0
    g[nonzero] = (Fm[nonzero] - F0) / (om[nonzero]**2)

    # If ω=0 present, estimate g(0)=0.5 F''(0) by a tiny quadratic fit around 0
    zero_mask = (om == 0.0)
    if zero_mask.any():
        if use_second_deriv_at_zero and om.size >= 5:
            # pick a small symmetric window around zero
            k = min(20, (om.size - 1) // 2)
            # ensure we have points on both sides
            left = np.where(om < 0)[0][-min(k, np.sum(om < 0)):]
            right = np.where(om > 0)[0][:min(k, np.sum(om > 0))]
            sel = np.concatenate([left, right])
            x = om[sel]
            y = Fm[sel]
            # quadratic fit y ≈ a x^2 + b
            A = np.vstack([x**2, np.ones_like(x)]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
            g0 = 0.5 * (2*a)  # since F''(0) = 2a for y=a x^2 + b
            g[zero_mask] = g0
        else:
            # fallback: use nearest nonzero value of g
            g[zero_mask] = g[nonzero][np.argmin(np.abs(om[nonzero]))]

    # Trapezoidal weights for non-uniform PV integration
    N = om.size
    w = np.zeros_like(om)
    if N > 1:
        d = np.diff(om)
        w[0] = 0.5 * d[0]
        w[-1] = 0.5 * d[-1-1] if N > 2 else 0.5 * d[0]
        if N > 2:
            w[1:-1] = 0.5 * (d[:-1] + d[1:])

    # Kernel 1/(ω - ω') with PV implemented by zeroing diagonal
    # We need H[g](ω_i) = (1/π) PV ∑_j g_j w_j / (ω_i - ω_j).
    denom = om[:, None] - om[None, :]           # (i,j): ω_i - ω_j
    np.fill_diagonal(denom, np.inf)             # PV: exclude j=i
    Kg = (g * w)[None, :] / denom               # broadcast over i

    Hg_sorted = (1.0 / np.pi) * np.sum(Kg, axis=1)  # Hilbert transform of g
    Imf_sorted = (om**2) * Hg_sorted

    # Unsort to original order
    Imf = np.empty_like(Imf_sorted)
    Imf[idx] = Imf_sorted
    return Imf
