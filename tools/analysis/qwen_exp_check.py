import numpy as np, math

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def time_shift(mu, sigma, t):
    em = math.exp(mu)
    return em/(em + np.power((1.0/t - 1.0), sigma))

rng=np.random.default_rng(1234)
r=rng.standard_normal(200000)
tvals=sigmoid(r)

for rsize in (256,512,1024):
    pixels = rsize * rsize
    ref = 512**2
    min_exp = 1.0/3.0
    max_exp = 3.0
    exp_raw = pixels / ref
    exp_clamped = min(max(exp_raw, min_exp), max_exp)
    mu = math.log(exp_clamped)
    shifted = np.array([time_shift(mu,1.0,t) for t in tvals])
    print(f"{rsize}x{rsize}: pixels={pixels}, raw_exp={exp_raw:.6f}, clamped_exp={exp_clamped:.6f}, mu={mu:.6f}, median_sigma={np.median(shifted):.6f}")
