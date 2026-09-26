# Low-Rank Model Merging Under Worst-Task Functional Error (P2)

Working title. Status: **TECHNICAL_TEST_PASS + SCIENCE_NOT_EVALUATED**; the real-adapter
pilot is **BLOCKED** (see `STATUS.md`, in Korean). Prior-art comparison: `PRIOR_ART.md`.

## What is implemented (CPU, Python standard library only)

NumPy/PyTorch are not installed in the development environment and were not installed,
so everything is pure Python and meant for small matrices.

| Module | Content |
|---|---|
| `src/lowrank_merge/adapters.py` | LoRA adapter, effective delta `dW = scaling * B @ A`, gauge `(B R, R^-1 A)`, parameter accounting |
| `src/lowrank_merge/objectives.py` | per-task empirical functional error `e_t(M) = ||(M - D_t) X_t||^2 / n_t`, relative version, cal/dev/test evaluation |
| `src/lowrank_merge/wrrr.py` | fixed-task-weight weighted reduced-rank regression; `pd` (S positive definite), `pinv` (singular S, same objective), `ridge` (explicitly a different objective) |
| `src/lowrank_merge/minimax.py` | `min_{rank(M)<=k} max_t e_t(M)`: Lagrangian dual lower bound (exact inner WRRR), primal heuristics (dual candidates, block-convex alternation, smoothed local descent); reports UB / LB / gap, no global-optimality claim |
| `src/lowrank_merge/references.py` | independent references: raw-data Izenman RRR, grid + ellipsoid (d_out=2, k=1), multi-start ALS |
| `src/lowrank_merge/baselines.py` | TA+SVD, RegMean (alpha trick; full / Euclidean truncation / whitened truncation), unofficial KnOTS-TA/TIES and CtM-like, factor averaging (non-invariant diagnostic) |
| `src/lowrank_merge/pipeline.py`, `run_cpu.py` | split discipline (cal = fit, dev = baseline hyper-parameters, test = report), runner with raw log + manifest |
| `src/lowrank_merge/pilot.py` | preflight for the real 4-adapter pilot config (never downloads/installs) |

## Commands

```bash
# unit tests (46 tests, ~10 s)
python3 -m unittest discover -s tests -v

# synthetic layer-level fixture (3 seeds, ~1 min, writes runs/<id>/)
PYTHONPATH=src python3 -m lowrank_merge.run_cpu --config configs/cpu_synthetic.json

# real-adapter pilot preflight (exit code 2 while blocked)
PYTHONPATH=src python3 -m lowrank_merge.pilot --config configs/pilot_4task_flan_t5_base_glue.json
```

## Scope notes

- The functional error is layer-local; it is not the end-to-end task loss.
- The synthetic fixture checks the implementation only; it is not evidence about real adapters.
- `READY_FOR_PILOT` (not reached) would certify the checklist only, not novelty or results.
