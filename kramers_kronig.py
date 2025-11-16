import numpy as np
from scipy.signal import hilbert

def calculate_R_w(A_func, w_array, **kwargs):
    """
    Calculates the causal kernel R(w) from A(w) using the Kramers-Kronig relations.

    Args:
        A_func (callable): A function that takes a frequency array (w) and optional
                           keyword arguments, and returns A(w).
        w_array (np.ndarray): A 1D numpy array of real frequency points.
                              Should be symmetric around w=0 for the Hilbert
                              transform to be accurate.
        **kwargs: Additional keyword arguments to be passed to A_func.

    Returns:
        np.ndarray: The complex-valued causal kernel R(w) at the given frequency points.
    """
    # 1. Calculate A(w)
    A_w = A_func(w_array, **kwargs)

    # 2. Calculate alpha(w) = 0.5 * ln(A(w))
    # Add a small epsilon for numerical stability to avoid log(0)
    alpha_w = 0.5 * np.log(A_w + 1e-20)

    # 3. Calculate beta(w) using the Hilbert transform of alpha(w)
    # The imaginary part of the analytic signal is the Hilbert transform.
    beta_w = np.imag(hilbert(alpha_w))

    # 4. Construct the causal kernel R(w)
    R_w = np.exp(alpha_w + 1j * beta_w)

    return R_w

def calculate_R_t(R_w, w_array):
    """
    Calculates the time-domain response R(t) by taking the inverse Fourier
    transform of R(w).

    Args:
        R_w (np.ndarray): The complex-valued causal kernel in the frequency domain.
        w_array (np.ndarray): The corresponding 1D numpy array of real frequency points.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the time array t and
                                       the real-valued time-domain response R(t).
    """
    # Determine frequency step and number of points
    N = len(w_array)
    dw = w_array[1] - w_array[0]

    # The inverse Fourier transform for causality uses exp(-iwt), which
    # corresponds to a forward FFT in numpy's convention.
    # We use ifftshift to handle the frequency ordering.
    R_t_complex = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(R_w)))

    # Calculate the corresponding time array
    # The time step dt is determined by the total frequency range.
    t_max = np.pi / dw
    dt = 2 * t_max / N
    t = np.arange(-t_max, t_max, dt)
    
    # Apply scaling for the transform
    R_t_scaled = R_t_complex * dw / (2 * np.pi)

    return t, np.real(R_t_scaled)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- Example Usage ---

    # Define the specific A(w) function from our analysis
    def A_example(w, nu):
        # A numerically stable way to compute cosh(w)/cosh(nu*w)
        with np.errstate(over='ignore', invalid='ignore'):
            res = np.exp(np.abs(w) - np.abs(nu * w))
        # For w where the above is not stable, cosh is fine.
        mask = np.abs(w) < 50
        res[mask] = np.cosh(w[mask]) / np.cosh(nu * w[mask])
        return res

    # Set up parameters
    nu_param = 2.0
    N_points = 2**14
    dw_step = 0.01
    w_max_val = (N_points / 2) * dw_step
    w = np.linspace(-w_max_val, w_max_val - dw_step, N_points)

    # 1. Calculate R(w)
    print("Calculating R(w) using Kramers-Kronig...")
    R_w_calculated = calculate_R_w(A_example, w, nu=nu_param)

    # 2. Calculate R(t)
    print("Calculating R(t) via inverse Fourier transform...")
    t_calculated, R_t_calculated = calculate_R_t(R_w_calculated, w)

    # 3. Plot the results to verify
    print("Plotting results...")
    plt.figure(figsize=(12, 6))
    plt.plot(t_calculated, R_t_calculated, label='R(t)')
    plt.title('Causal Time Response R(t) from Kramers-Kronig Module')
    plt.xlabel('t')
    plt.ylabel('R(t)')
    plt.xlim(-10, 50)
    plt.grid(True)
    plt.legend()
    plt.axvline(0, color='r', linestyle='--', linewidth=0.8, label='t=0')
    plt.show()

    # Verify that R(t) is close to zero for t < 0
    t_negative_mask = t_calculated < -1e-6 # Use a small tolerance
    max_val_for_negative_t = np.max(np.abs(R_t_calculated[t_negative_mask]))
    print(f"\\nMaximum absolute value of R(t) for t < 0: {max_val_for_negative_t:.2e}")
    if max_val_for_negative_t < 1e-3:
        print("Verification successful: R(t) is effectively zero for t < 0.")
    else:
        print("Verification failed: R(t) is not zero for t < 0.")

