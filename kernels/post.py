import numpy as np
from sklearn.preprocessing import KernelCenterer
def nearest_psd(K: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    K = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(K)
    w = np.clip(w, a_min=eps, a_max=None)
    return (V * w) @ V.T
def center_train_test(Ktr: np.ndarray, Kte: np.ndarray):
    kc = KernelCenterer()
    return kc.fit_transform(Ktr), kc.transform(Kte)
