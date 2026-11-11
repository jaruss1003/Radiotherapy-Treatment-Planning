# Radiotherapy Treatment Planning by Linear Programming

A course project in mathematical optimization (MTH 3107). Radiotherapy treatment
planning is posed as a **linear program**: choose the intensities of 105 radiation
beamlets so that a simulated tumor receives its full prescription dose while a
neighboring organ at risk is protected — with both rules enforced at **every voxel**,
not on average.

The phantom is a simulated two-dimensional slice (40 × 40 grid, 944 voxels): an
irregular tumor (112 voxels), one nearby organ at risk (59 voxels), and healthy
tissue. Doses are in real Gray — a 60 Gy prescription and a 30 Gy organ limit.

**Portfolio Page:** [jacobruss.pages.dev/#/projects/radiotherapy-treatment-planning](https://jacobruss.pages.dev/#/projects/radiotherapy-treatment-planning)

## Key Findings

- **Organ mean dose: 61.4 Gy → 10.8 Gy (−82%)** versus an equal-beamlet baseline
  scaled to the same tumor prescription.
- **Organ limit violations: 59 of 59 → 0.** The optimized organ maximum sits
  exactly at the 30 Gy limit — the constraint binds (2 voxels), it is not luck.
- **Healthy hot spots are capped:** no healthy voxel exceeds 65 Gy — the
  optimized maximum is exactly 65.0 Gy, binding at 7 voxels. The cap is set just
  under the baseline's own healthy maximum (65.1 Gy), so the optimized plan
  cannot create a worse hot spot than the naive plan. Without it, this same
  model drives one border voxel to 88.4 Gy.
- **The tumor guarantee is kept exactly:** minimum tumor dose 60.0 Gy and 100%
  coverage in both plans; healthy-tissue mean dose falls 62.5 → 24.6 Gy.
- **The optimum is sparse:** only 24 of the 105 beamlets are used, as expected
  for a vertex solution of an LP.
- **Protection is priced:** the shadow price of the organ limit is ≈ −0.18 Gy
  of objective per Gy of limit (solver duals −0.184, confirmed by re-solving:
  −0.176) — about three times the price the same geometry reports without the
  healthy cap. Banning hot spots reprices protection.
- **Sensitivity has a wall:** the sweep starts at the 30 Gy rule and reports
  every other limit against it — loosening to 42 Gy buys just −1.330 Gy of
  healthy mean dose, and tightening runs into a wall. Bisecting the gap puts
  the smallest feasible organ limit at 26.66 Gy: the rule has 3.34 Gy of
  wiggle room, and below the wall the three rules admit no plan at all.
- **One honest cost remains:** the tumor mean rises to 73.1 Gy — the objective
  never charges for dose inside the tumor, it only enforces the 60 Gy floor.

## The model

- **Variables:** x_j ≥ 0 — the intensity of beamlet j (7 beam angles × 15
  beamlets = 105).
- **Dose rule:** d = A·x, where A (944 × 105) is precomputed: a beamlet's dose
  fades exponentially with depth inside its strip (exp(−0.60 · depth)) with a
  0.2% scatter contribution outside it.
- **Objective:** minimize the weighted mean dose outside the tumor; organ voxels
  count 3×.
- **Constraints:** every tumor voxel ≥ 60 Gy; every organ voxel ≤ 30 Gy;
  every healthy voxel ≤ 65 Gy (hot-spot cap).
- **Solver:** SciPy `linprog` (HiGHS).

The full formulation, in the author's words, is in
[`report/formulation.md`](report/formulation.md).

## Run it

```bash
pip install numpy scipy matplotlib
python radiotherapy_optimization.py
```

One script produces every output: the four figures in `figures/`, the metric
table in `report/results_table.md`, and `results.txt` — the single source of
truth for every number above.

## Repository contents

| Path | What it is |
|---|---|
| `radiotherapy_optimization.py` | The whole project: phantom, dose matrix, baseline, LP, sensitivity sweep, figures |
| `results.txt` | Verified run output — every reported number |
| `figures/` | Figure 1 phantom · Figure 2 dose maps · Figure 3 dose–volume histograms · Figure 4 sensitivity sweep |
| `report/formulation.md` | The LP formulation (variables, dose rule, objective, constraints, assumptions) |
| `report/results_table.md` | Baseline vs. optimized metrics table |

## Limitations

A two-dimensional slice with one tumor and one organ; a simplified dose model
(exponential fall-off plus constant scatter), not clinical physics. Inside the
phantom, tissues differ in the rules applied to them, not in physical properties.
This is a learning build — not clinical software.
