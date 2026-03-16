# Research Log — ACDC Cardiac MRI 5-Fold CV

---
## Experiment 1 — 2026-03-16T04:13Z
**Experiment ID (commit hash):** 60f2bc54d74a

**Hypothesis:** Establish baseline with current config: wider CNN (1→32→64→128→256, 10.7M params), cross-attention fusion, deeper ClinicalEncoder (5→64→128→128), TTA=8, label_smoothing=0.1, WD=0.05, MAX_EPOCHS=200, 5-fold CV on all 100 patients.

**Change made:**
```diff
First experiment — no changes from starting config.
```

**Results:**
| Metric | Value |
|--------|-------|
| val_acc (mean) | 0.5900 |
| val_acc (std)  | 0.0583 |
| per_fold_acc   | [0.65, 0.65, 0.55, 0.60, 0.50] |
| per_class_acc  | NOR=0.60  DCM=0.65  HCM=0.50  MINF=0.50  RV=0.70 |
| prev best      | N/A (first experiment) |

**Interpretation:** The wider CNN (10.7M params) is severely overfitting on 80 training patients — train_acc reaches 0.95+ but val_acc is only 0.59. HCM and MINF are at chance (0.50). The cross-attention fusion with this large model is not working well. Need to dramatically reduce model size.

**Next hypothesis:** Revert to smaller CNN (1→16→32→64→128, ~1.16M params) with gated fusion (simpler than cross-attention), reduce MAX_EPOCHS to 80, increase DROPOUT to 0.6, and increase WEIGHT_DECAY to 0.1 to combat overfitting.
