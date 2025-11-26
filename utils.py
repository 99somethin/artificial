import cv2
import numpy as np
from scipy import interpolate

def read_rgb(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def to_grayscale(img):
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def resample_contour(contour, k):
    # contour: Nx2 array of (x,y) points (integers). Возвращает k точек (float)
    pts = contour.reshape(-1,2).astype(np.float64)
    # compute cumulative arc length
    d = np.sqrt(((np.diff(pts, axis=0))**2).sum(axis=1))
    d = np.concatenate([[0.0], d])
    cum = np.cumsum(d)
    if cum[-1] == 0:
        return np.repeat(pts[:1], k, axis=0)
    fx = interpolate.interp1d(cum, pts[:,0], kind='linear')
    fy = interpolate.interp1d(cum, pts[:,1], kind='linear')
    newlen = np.linspace(0, cum[-1], k)
    newx = fx(newlen)
    newy = fy(newlen)
    return np.vstack([newx, newy]).T

def contour_to_complex_vector(contour_resampled):
    # contour_resampled: k x 2
    diffs = np.diff(contour_resampled, axis=0, append=contour_resampled[:1])
    complex_vec = diffs[:,0] + 1j * diffs[:,1]
    return complex_vec

def normalized_scalar_product(G, K):
    # NSP = Re( (G, K) / (|G||K|) )  - but paper says NSP is (G,K)/|G||K| (complex possible)
    num = np.vdot(G, np.conj(K))  # vdot conjugates first arg, so use np.vdot(K, np.conj(G))? Simpler:
    # We'll compute (G,K) = sum( G_i * conj(K_i) )
    dot = np.sum(G * np.conj(K))
    denom = np.sqrt(np.sum(np.abs(G)**2) * np.sum(np.abs(K)**2))
    if denom == 0:
        return 0.0
    return dot / denom

def nsp_similarity(G, K):
    # return absolute value or real part? We'll return abs of NSP as measure [0..1]
    val = normalized_scalar_product(G, K)
    return np.abs(val)
