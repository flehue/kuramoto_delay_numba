import numpy as np
from scipy.io import loadmat
from scipy import signal
from time import time
import matplotlib.pyplot as plt
import kuramoto_delay as km


np.random.seed(335)

#load your favorite matrix
# sc_data = loadmat(r"C:\Users\flehu\OneDrive\Escritorio\postdocs\wael_post\fernando_stroke\data\raw\ANTONIO_20regiones_DET_PROB.mat")
# C_raw, D_raw = sc_data["Cprob"], sc_data["Dprob"]

sc_data = loadmat("input/matrices_syn_seed_1002_sin ajuste.mat")
C_raw, D_raw = sc_data["W_syn"], sc_data["D_syn"]


N = len(C_raw)
K_scalar, MD = 400, 0.024

#let the module normalize matrices internally
C_norm = km.normalize_connectivity_ref(C_raw)
K_used = km.build_K_matrix_ref(C_norm, K_scalar=K_scalar)
K_initial = K_used.copy()
D_base = km.build_delay_matrix_ref(D_raw, C_norm)

halt
#%%sample frequencies and initial conditions
omega = km.build_natural_frequencies_ref(N, freq_mean_hz=40, freq_std_hz=2)
initial_phases = km.generate_initial_phases_ref(N)

T, dt = 10, 0.0001
cutoff = int(2/dt) ##seconds
#%% run
print("Running!")
t1 = time()
phases_t = km.run_kuramoto(
    omega, K_used, T=T, dt=dt,
    D_base=D_base, MD=MD,
    initial_phases=initial_phases,
)

runtime = time() - t1
print(f"Run!, time = {runtime:.3f} s")

# np.save("phase_trig_identity.npy",phases_t)
#%% calculate spectra to observe

phases_t = phases_t.T
ts = np.sin(phases_t[cutoff:])
fs = 1/dt
nperseg = fs * 5
noverlap = nperseg // 2
f, PSD = signal.welch(ts, fs, nperseg=nperseg, noverlap=noverlap, axis=0)

#%% plot
plt.figure(1)
plt.clf()
ax = plt.subplot(221)
ax.set_title(f"T={T:.2f} s, dt = {dt:.4f}\nruntime = {runtime:.3f} s")
ax.plot(f, PSD.mean(axis=1))
ax.set_xlim((-1, 60))
ax = plt.subplot(224)
ax.set_title("structural matrix")
im = ax.imshow(K_initial)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()


#%%

phases_trig = np.load("")


