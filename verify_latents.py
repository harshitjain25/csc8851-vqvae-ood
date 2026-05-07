import os
import numpy as np

FILES = [
    "latent_codes/cifar10.npy",
    "latent_codes/cifar100.npy",
    "latent_codes/svhn.npy",
]

def main():
    arrays = {}
    for f in FILES:
        if not os.path.exists(f):
            print(f"[MISSING] {f}")
            return
        x = np.load(f)
        arrays[f] = x
        print(f"[OK] {f} shape={x.shape} dtype={x.dtype}")

    # checks
    shapes = [arrays[f].shape[1] for f in FILES]
    if len(set(shapes)) != 1:
        print("[ERROR] D mismatch across files:", shapes)
        return

    dtypes = [arrays[f].dtype for f in FILES]
    if any(dt != np.float32 for dt in dtypes):
        print("[WARN] dtype not float32, got:", dtypes)

    print("\nAll good ✅ Latents are consistent (N, D) with same D.")

if __name__ == "__main__":
    main()