import numpy as np

def linear_betas(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    return np.linspace(beta_start, beta_end, timesteps)

betas = linear_betas(1000, 1e-4, 0.02)
alphas = 1.0 - betas
alphas_cumprod = np.cumprod(alphas)
stds = np.sqrt(1.0 - alphas_cumprod)

print(f"timesteps={len(stds)}")
print(f"min std: {stds.min():.6f}, max std: {stds.max():.6f}, mean std: {stds.mean():.6f}")
print("first 10:")
for i in range(10):
    print(i, f"{stds[i]:.6f}")
print("last 5:")
for i in range(len(stds)-5, len(stds)):
    print(i, f"{stds[i]:.6f}")
print(f"expected std uniform t: {stds.mean():.6f}")
