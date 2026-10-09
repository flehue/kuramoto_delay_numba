import numpy as np
from numba import njit


def normalize_connectivity_ref(C_raw):
    """
    off-diagonal mean forced to 1, diagonal forced to 0.
    """
    N = C_raw.shape[0]
    C = C_raw.copy().astype(float)
    mask = ~np.eye(N, dtype=bool)
    C[mask] /= C[mask].mean()
    np.fill_diagonal(C, 0.0)
    return C


def build_K_matrix_ref(C_normalized, K_scalar):
    """
    Returns the matrix ready to pass as `K` into run_kuramoto.
    C_normalized must already be normalized
    """
    N = C_normalized.shape[0]
    global_coupling = K_scalar / N
    return global_coupling * C_normalized


def build_delay_matrix_ref(D_raw, C_normalized):
    """
    Normalizes D_raw so its mean over
    CONNECTED edges (C_normalized != 0) equals exactly 1. Pass this as
    D_base into run_kuramoto together with MD = your desired mean delay
    in seconds.
    """
    connected = C_normalized != 0
    D = D_raw.copy().astype(float)
    D = D / D[connected].mean()
    np.fill_diagonal(D, 0.0)
    return D


def build_natural_frequencies_ref(N, freq_mean_hz, freq_std_hz):
    freqs_hz = np.random.normal(freq_mean_hz, freq_std_hz, size=N)
    return 2 * np.pi * freqs_hz #note the multiplication by 2pi


def generate_initial_phases_ref(N):
    return np.random.uniform(0, 2 * np.pi, size=N)




#Kuramoto integration loop (Numba-jitted), delay-only.
@njit(cache=True)
def _run_kuramoto_core(frequencies, K, T,
                        D_base, MD,
                        initial_phases,dt):
    """
    Euler integration of the delay-Kuramoto phase equations.

    initial_phases : (N,) float64 array, REQUIRED. See note in the
        original module about Numba's separate RNG state -- generate
        phases with generate_initial_phases_ref(N) (or your own array)
        BEFORE calling this function.

    Delay handling
    --------------
    D_base : (N, N) float64 array (base delay structure, mean 1 over
        connected edges -- see build_delay_matrix_ref).
    MD : float64. Delay actually used = MD * D_base[i, j].
        Set MD = 0.0 to force instantaneous coupling everywhere even
        with a nonzero D_base.

    Coupling term
    -------------
    Uses sin(theta_j - theta_i) = sin(theta_j)cos(theta_i) - cos(theta_j)sin(theta_i),
    so sin/cos of each node's phase are computed once per time step (stored in
    sin_t / cos_t) instead of calling sin once per pair (i, j):

        sum_j K_ij sin(theta_j(t - d_ij) - theta_i)
            = cos(theta_i) * sum_j K_ij sin(theta_j(t - d_ij))
            - sin(theta_i) * sum_j K_ij cos(theta_j(t - d_ij))
    """
    N = frequencies.shape[0]
    n_steps = int(T / dt)

    # ---- delay, in integer steps of dt ----
    delay_steps = np.zeros((N, N), dtype=np.int64)
    for i in range(N):
        for j in range(N):
            d = int(round(MD * D_base[i, j] / dt))
            if d < 0:
                d = 0
            delay_steps[i, j] = d

    # initial phases
    phases = initial_phases.copy()

    phases_t = np.zeros((N, n_steps))
    sin_t = np.zeros((N, n_steps))   # sin of stored phase history
    cos_t = np.zeros((N, n_steps))   # cos of stored phase history
    for i in range(N):
        phases_t[i, 0] = phases[i]
        sin_t[i, 0] = np.sin(phases[i])
        cos_t[i, 0] = np.cos(phases[i])

    for t in range(1, n_steps):
        t_prev = t - 1   # `phases` currently holds phases_t[:, t_prev]

        sin_sum = np.zeros(N)

        for i in range(N):
            acc_s = 0.0   # sum_j K_ij sin(theta_j delayed)
            acc_c = 0.0   # sum_j K_ij cos(theta_j delayed)
            for j in range(N):
                if i == j:
                    continue

                # possibly-delayed lookup into stored history
                tt = t_prev - delay_steps[i, j]
                if tt < 0:
                    tt = 0
                acc_s += K[i, j] * sin_t[j, tt]
                acc_c += K[i, j] * cos_t[j, tt]

            # theta_i is the current phase, i.e. phases_t[i, t_prev]
            sin_sum[i] = cos_t[i, t_prev] * acc_s - sin_t[i, t_prev] * acc_c

        # integrate phases
        for i in range(N):
            phases[i] = (phases[i] + dt * (frequencies[i] + sin_sum[i])) % (2 * np.pi)
            phases_t[i, t] = phases[i]
            sin_t[i, t] = np.sin(phases[i])
            cos_t[i, t] = np.cos(phases[i])

    return phases_t


# ---------------------------------------------------------------------
# 5. Public entry point. Plain Python (not jitted) so it can have real defaults
# ---------------------------------------------------------------------
def run_kuramoto(frequencies, K, T,
                 D_base, MD,
                 initial_phases, dt=1e-4):
    """
    Run the delay-Kuramoto simulation.
    Returns
    -------
    phases_t : (N, n_steps) array of phases over time
    """
    N = frequencies.shape[0]

    return _run_kuramoto_core(
        frequencies, K, T,
        D_base, MD,
        initial_phases, dt,
    )