# Radiotherapy Treatment Planning — Linear Programming
# MTH3107 Optimization project.
# Run: python3 radiotherapy_optimization.py
# Outputs: figures in figures/, all numbers printed to results.txt

# ---------- 1. Parameters ----------

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from pathlib import Path
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

FIG = Path(__file__).parent / "figures" 

N = 40                                # grid size - the slice is a 40 x 40 grid of points
TUMOR_C = np.array([0.15, 0.10]);  TUMOR_R = 0.30   # tumor center (x, y) and radius
OAR_C = TUMOR_C + 0.44 * np.array([0.635, -0.772])
OAR_RADII = (0.30, 0.16);  OAR_ROT = 39.0
ANGLES_DEG = [0, 51, 103, 154, 206, 257, 309]       # the 7 beam directions, in degrees
BEAMLET_CENTERS = np.linspace(-0.9, 0.9, 15)        # positions of the 15 beamlets in each direction
STRIP_W = BEAMLET_CENTERS[1] - BEAMLET_CENTERS[0]   # beamlet width, measured from the centers - do not type a rounded value, it changes the results
MU = 0.60                             # how fast dose fades with depth - bigger means a faster fade
SCATTER = 0.002                       # tiny dose every voxel gets, even outside any beam's path
D_RX = 60.0                           # prescription dose in Gy - every tumor voxel must get at least this
D_OAR = 30.0                          # organ limit in Gy - no organ voxel may get more than this
W_OAR = 3.0                           # organ weight in the objective - organ dose counts 3 times as much

# ---------- 2. Phantom ----------

xs = np.linspace(-1, 1, N)
ys = np.linspace(-1, 1, N)
XX, YY = np.meshgrid(xs, ys)
pts = np.column_stack([XX.ravel(), YY.ravel()])

# Body: an ellipse — wider (0.99) than tall (0.80),
# like a torso cross-section.
body = (pts[:, 0] / 0.99) ** 2 + (pts[:, 1] / 0.80) ** 2 <= 1

# Tumor: a blob. Its radius wobbles with direction —
# two sine waves over the angle around the center —
# so the edge is irregular, like a real mass.
dx = pts[:, 0] - TUMOR_C[0]
dy = pts[:, 1] - TUMOR_C[1]
r = np.hypot(dx, dy)
theta = np.arctan2(dy, dx)
r_eff = TUMOR_R * (1 + 0.16 * np.sin(3 * theta + 0.7)
                     + 0.10 * np.sin(5 * theta + 2.1))
tumor = (r <= r_eff) & body

# Organ: an elongated ellipse, tilted 39 degrees.
# u, v are the point's coordinates in the organ's
# own rotated frame.
dxo = pts[:, 0] - OAR_C[0]
dyo = pts[:, 1] - OAR_C[1]
rot = np.deg2rad(OAR_ROT)
u = dxo * np.cos(rot) + dyo * np.sin(rot)
v = -dxo * np.sin(rot) + dyo * np.cos(rot)
oar = ((u / OAR_RADII[0]) ** 2 + (v / OAR_RADII[1]) ** 2 <= 1) & body & ~tumor

healthy = body & ~tumor & ~oar
print(body.sum(), tumor.sum(), oar.sum(), healthy.sum())

# ---------- 3. Dose matrix ----------
idx_body = np.where(body)[0]
voxel_pts = pts[idx_body]
NB = len(ANGLES_DEG) * len(BEAMLET_CENTERS)   # 105
A = np.zeros((len(idx_body), NB))

beamlets = [(np.deg2rad(a), c)
            for a in ANGLES_DEG
            for c in BEAMLET_CENTERS]

for j, (ang, c) in enumerate(beamlets):
    direction = np.array([np.cos(ang), np.sin(ang)])
    perp = np.array([-np.sin(ang), np.cos(ang)]) 

    # 1. s: sideways position of each voxel for this beam.
    s = voxel_pts @ perp

     # 2. depth: distance of each voxel along the beam.
    depth = voxel_pts @ direction + 1.0

    # 3. atten: the fade — e to the power of 
    atten = np.exp(-MU * np.clip(depth, 0, None))

    # 4. strip: True where a voxel is in this strip. In = size of (s - c) at most half the width of the strip.
    strip = np.abs(s - c) <= STRIP_W / 2.0

    # 5. column j of A: atten in the strip, SCATTER * atten outside it. Tool:
    A[:, j] = np.where(strip,atten, SCATTER * atten)

row_of = {v: k for k, v in enumerate(idx_body)}
t_rows = [row_of[i] for i in np.where(tumor)[0]]
o_rows = [row_of[i] for i in np.where(oar)[0]]
h_rows = [row_of[i] for i in np.where(healthy)[0]]
A_T, A_O, A_H = A[t_rows], A[o_rows], A[h_rows]


# Building the anatomy grid (grayscale, like a scan)
img = np.ones((N, N))
img[healthy.reshape(N, N)] = 0.66
img[oar.reshape(N, N)] = 0.43
img[tumor.reshape(N, N)] = 0.27

# Texture: seeded grain, a different amount per tissue
rng = np.random.default_rng(7)
noise = rng.normal(0, 1, img.shape)
tex = img.copy()
tex[healthy.reshape(N, N)] += 0.045 * noise[healthy.reshape(N, N)]
tex[oar.reshape(N, N)] += 0.030 * noise[oar.reshape(N, N)]
tex[tumor.reshape(N, N)] += 0.055 * noise[tumor.reshape(N, N)]
img = np.clip(tex, 0, 1)

# Plotting the anatomy grid
fig, ax = plt.subplots()
ax.imshow(img, cmap="gray", vmin=0, vmax=1, extent=(-1, 1, -1, 1), origin="lower")

for a_deg in ANGLES_DEG:
    d = np.array([np.cos(np.deg2rad(a_deg)), np.sin(np.deg2rad(a_deg))])
    ax.annotate("", xy=-1.05 * d, xytext=-1.35 * d, arrowprops=dict(arrowstyle="->"))

ax.set_xlim(-1.4, 1.4)
ax.set_ylim(-1.4, 1.4)

legend_handles = [
    Patch(color=(0.66, 0.66, 0.66), label="Healthy tissue"),
    Patch(color=(0.43, 0.43, 0.43), label="Organ at risk"),
    Patch(color=(0.27, 0.27, 0.27), label="Tumor"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0))

ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_title("Phantom - Tumor and Organ at Risk")

fig.savefig(FIG / "fig1_phantom.png", dpi=200, bbox_inches="tight")

# ---------- 4. Solver ----------

def solve_lp(d_rx=D_RX, d_oar=D_OAR):
    n =len(h_rows) + W_OAR * len(o_rows)
    c_obj = (A_H.sum(axis=0) + W_OAR * A_O.sum(axis=0)) / n
    A_ub = np.vstack([-A_T, A_O])
    b_ub = np.concatenate([np.full(len(t_rows), -d_rx), np.full(len(o_rows), d_oar)])

    return linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method="highs")



# ---------- 5. Metrics ----------

def metrics(x): 

    d = A @ x
    d_t = d[t_rows]
    d_o = d[o_rows]
    d_h = d[h_rows]
    return {
        "tumor_min": float(d_t.min()),
        "tumor_mean": float(d_t.mean()),
        "tumor_cov95": float((d_t >= D_RX * 0.95).sum()/ len(d_t) * 100),
        "healthy_mean": float(d_h.mean()),
        "healthy_max": float(d_h.max()),
        "oar_mean": float(d_o.mean()),
        "oar_max": float(d_o.max()),
        "oar_violations": float((d_o > D_OAR + 1e-6).sum()),
        "beamlets_used": float((x > 1e-6).sum())
    }

# ---------- 6. Baseline and optimized runs ----------

x_base = np.ones(NB) 
m0 = metrics(x_base)
scale = D_RX / m0["tumor_min"]

x_base = scale * x_base
m_base = metrics(x_base)

res = solve_lp()
x_opt = res.x
m_opt = metrics(x_opt)

# ---------- 7. Sensitivity sweep ----------

limits = [42, 36, 30, 24, 18, 12]
sweep_rows = []
for lim in limits:
    r = solve_lp(D_RX, lim)
    m = metrics(r.x)
    sweep_rows.append((lim, m["healthy_mean"]))
    print(lim, round(m["healthy_mean"], 3))

# Binding proof: slack under the optimized plan
d_opt = A @ x_opt
oar_slack = D_OAR - d_opt[o_rows]
n_oar_at_limit = int((oar_slack < 1e-6).sum())
tum_slack = d_opt[t_rows] - D_RX
n_tum_at_rx = int((tum_slack < 1e-6).sum())
print("organ min slack:", oar_slack.min(), "| organ voxels at limit:", n_oar_at_limit)
print("tumor voxels at prescription:", n_tum_at_rx, "of", len(t_rows))


# Shadow price of the organ limit, two ways
duals = res.ineqlin.marginals[len(t_rows):]
price_dual = float(duals.sum())
r_lo = solve_lp(D_RX, 29.4)
r_hi = solve_lp(D_RX, 30.6)
price_fd = (r_hi.fun - r_lo.fun) / 1.2
print("shadow price (dual):", price_dual)
print("shadow price (re-solve):", price_fd)


# ---------- 8. Outputs ----------

def dose_grid(x): 
    full = np.full(len(pts), np.nan)
    full[idx_body] = A @ x
    return full.reshape(N, N)

# Dose heatmaps

fig, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].imshow(dose_grid(x_base), vmin=0, vmax=90, cmap="inferno", extent=(-1, 1, -1, 1), origin="lower")
im = axes[1].imshow(dose_grid(x_opt), vmin=0, vmax=90, cmap="inferno", extent=(-1, 1, -1, 1), origin="lower")


# Outline the tumor (white) and the organ (cyan) on both maps
for ax in axes:
    t = tumor.reshape(N, N)
    o = oar.reshape(N, N)
    ax.contour(t, levels=[0.5], colors="black", linewidths=4.5, extent=(-1, 1, -1, 1))
    ax.contour(t, levels=[0.5], colors="white", linewidths=2.5, extent=(-1, 1, -1, 1))
    ax.contour(o, levels=[0.5], colors="black", linewidths=4.5, linestyles="dashed", extent=(-1, 1, -1, 1))
    ax.contour(o, levels=[0.5], colors="cyan", linewidths=2.5, linestyles="dashed", extent=(-1, 1, -1, 1))



# Region Labels
for ax in axes:
    ax.text(TUMOR_C[0], TUMOR_C[1], "Tumor", color="white", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5))
    ax.text(OAR_C[0], OAR_C[1], "Organ", color="white", ha="center", va="center",
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=1.5))

fig.colorbar(im, ax=axes, label="Dose (Gy)")


# Labels and title
axes[0].set_xlabel("X Position")
axes[0].set_ylabel("Y Position")
axes[0].set_title("Baseline: Equal Beamlet Intensities")

axes[1].set_xlabel("X Position")
axes[1].set_ylabel("Y Position")
axes[1].set_title("Optimized: Linear Programming Solution")

# Save figure
fig.savefig(FIG / "fig2_dose_maps.png", dpi=200, bbox_inches="tight")

# Figure 3: Dose Volume Histograms (DVH)
fig, ax = plt.subplots()

def dvh(doses):
    d = np.sort(doses)[::-1]
    pct = np.arange(1, len(d) + 1) / len(d) * 100
    return d, pct

dt_base = dose_grid(x_base)[tumor.reshape(N, N)]
do_base = dose_grid(x_base)[oar.reshape(N, N)]
dt_opt = dose_grid(x_opt)[tumor.reshape(N, N)]
do_opt = dose_grid(x_opt)[oar.reshape(N, N)]

d, pct = dvh(dt_base)
ax.plot(d, pct, label="Tumor (Baseline)", color="red", linestyle="--")
d, pct = dvh(do_base)
ax.plot(d, pct, label="Organ (Baseline)", color="royalblue", linestyle="--")
d, pct = dvh(dt_opt)
ax.plot(d, pct, label="Tumor (Optimized)", color="red", linewidth=2.2)
d, pct = dvh(do_opt)
ax.plot(d, pct, label="Organ (Optimized)", color="royalblue", linewidth=2.2)

ax.set_xlabel("Dose (Gy)")
ax.set_ylabel("Volume (%)")
ax.set_title("Dose Volume Histograms (DVH)")

ax.axvline(D_OAR, color="gray", linestyle=":", label="Organ Limit")
ax.axvline(D_RX, color="gray", linestyle=":", label="Prescription Dose")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

fig.savefig(FIG / "fig3_dvh.png", dpi=200, bbox_inches="tight")


# Figure 4: Sensitivity — healthy mean dose vs organ limit
fig, ax = plt.subplots()
xs = [row[0] for row in sweep_rows]
ys = [row[1] for row in sweep_rows]
ax.plot(xs, ys, marker="o")
ax.set_xlabel("Organ dose limit (Gy)")
ax.set_ylabel("Healthy mean dose (Gy)")
ax.set_title("Sensitivity: healthy dose vs organ limit")
fig.savefig(FIG / "fig4_sensitivity.png", dpi=200, bbox_inches="tight")


# Results table — written to report/results_table.md
REPORT = Path(__file__).parent / "report"
rows = [
    ("Tumor min dose (Gy)", "tumor_min", ".1f"),
    ("Tumor mean dose (Gy)", "tumor_mean", ".1f"),
    ("Tumor coverage >= 57 Gy (%)", "tumor_cov95", ".1f"),
    ("Healthy mean dose (Gy)", "healthy_mean", ".1f"),
    ("Healthy max dose (Gy)", "healthy_max", ".1f"),
    ("Organ mean dose (Gy)", "oar_mean", ".1f"),
    ("Organ max dose (Gy)", "oar_max", ".1f"),
    ("Organ limit violations", "oar_violations", ".0f"),
    ("Beamlets used (of 105)", "beamlets_used", ".0f"),
]
lines = ["| Metric | Baseline | Optimized |", "|---|---|---|"]
for label, key, fmt in rows:
    lines.append(f"| {label} | {m_base[key]:{fmt}} | {m_opt[key]:{fmt}} |")
(REPORT / "results_table.md").write_text("\n".join(lines) + "\n")


# results.txt — every number the report quotes
L = []
L.append("RADIOTHERAPY TREATMENT PLANNING - RESULTS")
L.append(f"Phantom: body {body.sum()} | tumor {tumor.sum()} | organ {oar.sum()} | healthy {healthy.sum()} | beamlets {NB}")
L.append("")
L.append("BASELINE (equal beamlets, scaled to prescription)")
for k, v in m_base.items():
    L.append(f"  {k}: {v}")
L.append("OPTIMIZED (linear programming)")
for k, v in m_opt.items():
    L.append(f"  {k}: {v}")
L.append("")
L.append(f"Binding: organ voxels at limit: {n_oar_at_limit} (min slack {oar_slack.min():.2e}) | tumor voxels at prescription: {n_tum_at_rx} of {len(t_rows)}")
L.append(f"Shadow price: dual {price_dual:.4f} | re-solve {price_fd:.4f} (Gy of objective per Gy of organ limit)")
L.append("")
L.append("Sweep (organ limit -> healthy mean dose):")
for lim, hm in sweep_rows:
    L.append(f"  {lim} Gy -> {hm:.3f} Gy")
(Path(__file__).parent / "results.txt").write_text("\n".join(L) + "\n")
