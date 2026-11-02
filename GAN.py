"""
Optimal Transport and Game Dynamics in GANs -- image version (MNIST).

Scalar toy  ->  this script
---------------------------------------------------------------
payoff V(g,d)                 WGAN-GP minimax game  min_G max_D E[D(x)] - E[D(G(z))]
W1(Pg, Pr) = |g - mean|       Sliced Wasserstein-1 between generated and real images
path length of g(t)           sum of SW1 between consecutive generator snapshots
efficiency                    (SW0 - SW_T) / path length
eps convergence               eps = 1.5 x noise floor (real-vs-real SW1)
lr_g, lr_d regimes            same idea: different (lr_g, lr_d, n_critic) settings

Run:  pip install torch torchvision matplotlib numpy
      python gan_images_wasserstein_dynamics.py
Use a GPU (Colab T4 is fine). Set QUICK = True for a fast smoke test.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torchvision.utils import make_grid

QUICK = False                      # True -> tiny run to check everything works
DATASET = "MNIST"                  # "MNIST" or "FashionMNIST"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

Z_DIM = 64
BATCH = 64
GP_LAMBDA = 10.0
G_STEPS = 300 if QUICK else 4000   # generator updates per regime
EVAL_EVERY = 50 if QUICK else 100
N_EVAL = 1000 if QUICK else 2000   # images used per snapshot
N_PROJ = 128                       # random projections for sliced Wasserstein

# (lr_g, lr_d, n_critic)
EXPERIMENTS = {
    "Balanced":            (1e-4, 1e-4, 5),
    "Fast critic (TTUR)":  (1e-4, 4e-4, 5),
    "Fast generator":      (4e-4, 1e-4, 1),
    "Aggressive (unstable)": (2e-3, 1e-4, 1),
}

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- data
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize((0.5,), (0.5,))])   # -> [-1, 1]
DS = getattr(datasets, DATASET)
train_ds = DS("./data", train=True, download=True, transform=tf)
test_ds = DS("./data", train=False, download=True, transform=tf)

loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                                     drop_last=True, num_workers=2)


def take(ds, n, seed):
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    return torch.stack([ds[i][0] for i in idx]).view(n, -1)


real_eval = take(test_ds, N_EVAL, 1).to(DEVICE)          # P_r reference
real_floor = take(train_ds, N_EVAL, 2).to(DEVICE)        # for noise floor

# ---------------------------------------------------------------- metric
proj_gen = torch.Generator(device="cpu").manual_seed(0)
PROJ = torch.randn(784, N_PROJ, generator=proj_gen)
PROJ = (PROJ / PROJ.norm(dim=0, keepdim=True)).to(DEVICE)


@torch.no_grad()
def sliced_w1(a, b):
    """Sliced Wasserstein-1 between two equal-size sets of flattened images."""
    pa = (a @ PROJ).sort(dim=0).values
    pb = (b @ PROJ).sort(dim=0).values
    return (pa - pb).abs().mean().item()


NOISE_FLOOR = sliced_w1(real_eval, real_floor)
EPS = 1.5 * NOISE_FLOOR
print(f"Device: {DEVICE} | noise floor SW1 = {NOISE_FLOOR:.4f} | eps = {EPS:.4f}")

# ---------------------------------------------------------------- models
class G(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(Z_DIM, 128 * 7 * 7), nn.ReLU(True))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh())

    def forward(self, z):
        return self.net(self.fc(z).view(-1, 128, 7, 7))


class D(nn.Module):                    # critic: no BatchNorm (needed for GP)
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Flatten(), nn.Linear(128 * 7 * 7, 1))

    def forward(self, x):
        return self.net(x).view(-1)


def grad_penalty(d, real, fake):
    eps = torch.rand(real.size(0), 1, 1, 1, device=DEVICE)
    xh = (eps * real + (1 - eps) * fake).requires_grad_(True)
    g = torch.autograd.grad(d(xh).sum(), xh, create_graph=True)[0]
    return ((g.view(g.size(0), -1).norm(dim=1) - 1) ** 2).mean()


def infinite(loader):
    while True:
        for x, _ in loader:
            yield x


# ---------------------------------------------------------------- training
fixed_z = torch.randn(N_EVAL, Z_DIM, device=DEVICE)
sample_z = torch.randn(64, Z_DIM, device=DEVICE)


def train(lr_g, lr_d, n_critic):
    torch.manual_seed(SEED)
    g, d = G().to(DEVICE), D().to(DEVICE)
    og = torch.optim.Adam(g.parameters(), lr=lr_g, betas=(0.0, 0.9))
    od = torch.optim.Adam(d.parameters(), lr=lr_d, betas=(0.0, 0.9))
    data = infinite(loader)

    dists, snaps, w_est = [], [], []

    def snapshot():
        g.eval()
        with torch.no_grad():
            fake = g(fixed_z).view(N_EVAL, -1)
        g.train()
        dists.append(sliced_w1(fake, real_eval))
        snaps.append(fake)

    snapshot()
    for it in range(1, G_STEPS + 1):
        for _ in range(n_critic):                       # critic ascent
            real = next(data).to(DEVICE)
            z = torch.randn(BATCH, Z_DIM, device=DEVICE)
            fake = g(z).detach()
            w = d(real).mean() - d(fake).mean()          # critic's W1 estimate
            loss_d = -w + GP_LAMBDA * grad_penalty(d, real, fake)
            od.zero_grad(); loss_d.backward(); od.step()
        z = torch.randn(BATCH, Z_DIM, device=DEVICE)     # generator descent
        loss_g = -d(g(z)).mean()
        og.zero_grad(); loss_g.backward(); og.step()

        if it % EVAL_EVERY == 0:
            snapshot()
            w_est.append(w.item())
    g.eval()
    with torch.no_grad():
        grid = g(sample_z).cpu()
    return dists, snaps, w_est, grid


results = {}
for name, (lg, ld, nc) in EXPERIMENTS.items():
    print(f"Training: {name} (lr_g={lg}, lr_d={ld}, n_critic={nc})")
    dists, snaps, w_est, grid = train(lg, ld, nc)
    dists = np.array(dists)
    # path length in (sliced) Wasserstein space: consecutive generator snapshots
    steps_w = np.array([sliced_w1(snaps[i], snaps[i + 1]) for i in range(len(snaps) - 1)])
    path = steps_w.sum()
    eff = (dists[0] - dists[-1]) / path if path > 1e-12 else 0.0
    below = np.where(dists < EPS)[0]
    conv = int(below[0]) * EVAL_EVERY if len(below) else None
    worsen = int(np.sum(np.diff(dists) > 0))             # oscillation analogue
    results[name] = dict(dists=dists, path=path, eff=eff, conv=conv,
                         worsen=worsen, w_est=w_est, grid=grid,
                         lr_g=lg, lr_d=ld, nc=nc)
    del snaps

# ---------------------------------------------------------------- report
print("\nGAN GAME DYNAMICS ON IMAGES: RESULTS")
print(f"Dataset: {DATASET} | metric: sliced W1 | noise floor {NOISE_FLOOR:.4f} | eps {EPS:.4f}")
print("-" * 110)
print(f"{'Regime':<24}{'lr_g':>8}{'lr_d':>8}{'n_crit':>8}{'Init SW1':>10}"
      f"{'Final SW1':>11}{'Path len':>10}{'Efficiency':>11}{'Conv iter':>10}{'Worsen':>8}")
print("-" * 110)
for n, r in results.items():
    ct = str(r["conv"]) if r["conv"] is not None else "never"
    print(f"{n:<24}{r['lr_g']:>8.0e}{r['lr_d']:>8.0e}{r['nc']:>8d}"
          f"{r['dists'][0]:>10.4f}{r['dists'][-1]:>11.4f}{r['path']:>10.4f}"
          f"{r['eff']:>11.4f}{ct:>10}{r['worsen']:>8d}")

# ---------------------------------------------------------------- plots
xs = np.arange(len(next(iter(results.values()))["dists"])) * EVAL_EVERY

plt.figure(figsize=(10, 5))
for n, r in results.items():
    plt.semilogy(xs, r["dists"], label=n)
plt.axhline(EPS, color="gray", ls=":", label="eps")
plt.xlabel("Generator iteration"); plt.ylabel("Sliced W1(Pg, Pr)")
plt.title(f"Convergence of Generator Distribution ({DATASET})")
plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10, 5))
for n, r in results.items():
    plt.plot(xs[1:], r["w_est"], label=n)
plt.xlabel("Generator iteration"); plt.ylabel("Critic Wasserstein estimate")
plt.title("Critic's W1 estimate (value of the game)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

plt.figure(figsize=(7, 5))
for n, r in results.items():
    plt.scatter(r["path"], r["eff"], s=90, label=n)
plt.xlabel("Path length in Wasserstein space"); plt.ylabel("Efficiency")
plt.title("Trajectory length vs efficiency")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

fig, axes = plt.subplots(1, len(results) + 1, figsize=(4 * (len(results) + 1), 4.4))
real_grid = make_grid(real_eval[:64].view(-1, 1, 28, 28).cpu(), nrow=8, normalize=True, value_range=(-1, 1))
axes[0].imshow(real_grid.permute(1, 2, 0)); axes[0].set_title("Real"); axes[0].axis("off")
for ax, (n, r) in zip(axes[1:], results.items()):
    gr = make_grid(r["grid"], nrow=8, normalize=True, value_range=(-1, 1))
    ax.imshow(gr.permute(1, 2, 0)); ax.set_title(n, fontsize=9); ax.axis("off")
plt.tight_layout(); plt.show()

print("\nLIMITATION: SW1 is a projection-based approximation of W1 in pixel space; "
      "a feature-space metric (e.g. Inception/FID features) would be a stronger check.")
