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

N = 30                                # grid size - the slice is a 30 x 30 grid of points
TUMOR_C = np.array([0.15, 0.10]);  TUMOR_R = 0.32   # tumor center (x, y) and radius
OAR_C   = np.array([0.52, -0.35]); OAR_R   = 0.22   # organ center and radius - placed close to the tumor on purpose
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

# Worked example: a point is in the body if its
# distance from (0, 0) is at most 0.98.
body = np.hypot(pts[:, 0], pts[:, 1]) <= 0.98

tumor = (np.hypot(pts[:, 0] - TUMOR_C[0], pts[:, 1] - TUMOR_C[1]) <= TUMOR_R) & body 

oar = (np.hypot(pts[:, 0] - OAR_C[0], pts[:, 1] - OAR_C[1]) <= OAR_R) & body & ~tumor

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

# ---------- 4. Solver ----------
# ---------- 5. Metrics ----------
# ---------- 6. Baseline and optimized runs ----------
# ---------- 7. Sensitivity sweep ----------
# ---------- 8. Outputs ----------
