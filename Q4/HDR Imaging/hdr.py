import numpy as np
import imageio.v2 as imageio
from PIL import Image

def resize_all(imgs):
    h, w = imgs[0].shape[:2]
    out = []
    for im in imgs:
        if im.shape[:2] != (h, w):
            im = np.array(Image.fromarray(im).resize((w, h)))
        out.append(im)
    return out

def gsolve(Z, B, lam=100):
    n = 256
    P, N = Z.shape
    w = np.array([min(z, 255 - z) for z in range(256)])
    A = np.zeros((P*N + n + 1, n + P))
    b = np.zeros(P*N + n + 1)
    k = 0
    for i in range(P):
        for j in range(N):
            z = Z[i, j]
            wj = w[z]
            A[k, z] = wj
            A[k, n+i] = -wj
            b[k] = wj * B[j]
            k += 1
    A[k, 128] = 1
    k += 1
    for z in range(1, n-1):
        A[k, z-1:z+2] = lam * np.array([1, -2, 1])
        k += 1
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    return x[:n]

def build_hdr(imgs, log_t, lam=100, samples=100):
    H, W = imgs[0].shape[:2]
    N = len(imgs)
    ys = np.random.randint(0, H, samples)
    xs = np.random.randint(0, W, samples)
    Z = lambda c: np.stack([im[ys, xs, c] for im in imgs], axis=1)
    gR = gsolve(Z(0), log_t, lam)
    gG = gsolve(Z(1), log_t, lam)
    gB = gsolve(Z(2), log_t, lam)
    w = np.array([min(z, 255 - z) for z in range(256)])
    hdr = np.zeros((H, W, 3), float)
    for c, g in enumerate([gR, gG, gB]):
        num = np.zeros((H, W))
        den = np.zeros((H, W))
        for j in range(N):
            Zc = imgs[j][:, :, c]
            ww = w[Zc]
            num += ww * (g[Zc] - log_t[j])
            den += ww
        hdr[:, :, c] = np.exp(num / np.maximum(den, 1e-8))
    return hdr

def reinhard(hdr, key=0.18, gamma=1/2.2):
    eps = 1e-6
    L = 0.2126*hdr[:, :, 0] + 0.7152*hdr[:, :, 1] + 0.0722*hdr[:, :, 2]
    L = np.maximum(L, eps)
    Lavg = np.exp(np.mean(np.log(L)))
    Lm = (key / Lavg) * L
    Ld = Lm / (1 + Lm)
    out = hdr * (Ld / L)[:, :, None]
    out /= out.max() + eps
    out **= gamma
    return (out * 255).clip(0, 255).astype(np.uint8)

if __name__ == "__main__":
    files = ["1_bright.jpg", "2_medium.jpg", "3_dark.jpg"]
    imgs = resize_all([imageio.imread(f) for f in files])
    t = np.array([1/30, 1/60, 1/125])
    log_t = np.log(t)
    hdr = build_hdr(imgs, log_t)
    ldr = reinhard(hdr)
    imageio.imwrite("hdr_output.jpg", ldr)
    print("HDR image saved as hdr_output.jpg")