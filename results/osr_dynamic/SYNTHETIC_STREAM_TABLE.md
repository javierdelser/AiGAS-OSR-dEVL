# Synthetic stream baselines (MINAS, KNNENS, SENCForest)

Prequential stream protocol (Table I, arXiv:2504.05761): file order,
`T_s` supervised prefix, batch size `n // batches`, sliding train window
(`GNG_UTILS.definir_sets_training_test`). Labels train only for `t < T_s`;
later batches update MINAS from unlabeled features (extra micro-clusters).

Metrics: OSR micro-F1 (%) over all scored test instances; prequential error (%)
= `errorVERDADERA` × 100 accumulated over stream batches (reject → 100).

| Dataset | T_s | MINAS F1 | MINAS Preq | KNNENS F1 | KNNENS Preq | SENC F1 | SENC Preq |
|---------|-----|----------|------------|-----------|-------------|---------|-----------|
| 1CDT_MOD_OSR | 760 | 40.21 | 58.43 | 71.25 | 35.26 | 66.84 | 38.88 |
| 1CHT_MOD_OSR | 720 | 38.78 | 67.16 | 82.24 | 26.68 | 73.27 | 37.30 |
| 1CSurr_MOD_OSR | 808 | 26.16 | 75.01 | 76.94 | 33.09 | 76.25 | 33.89 |
| 2CDT_MOD_OSR | 660 | 11.44 | 85.13 | 36.61 | 70.47 | 30.51 | 74.32 |
| 2CHT_MOD_OSR | 660 | 14.05 | 83.78 | 44.28 | 64.86 | 35.61 | 71.00 |
| 4CE1CF_NO_OSR | 3664 | 48.68 | 64.05 | 82.19 | 28.55 | 71.14 | 42.30 |
| 4CRE-V1_NO_OSR | 1246 | 7.19 | 88.32 | 35.69 | 71.82 | 30.58 | 75.60 |
| 4CRE-V2_NO_OSR | 1120 | 6.31 | 88.79 | 31.86 | 75.74 | 33.28 | 73.46 |
| 4CR_NO_OSR | 6176 | 9.87 | 88.41 | 32.66 | 75.05 | 24.40 | 80.29 |
| 5CVT_MOD_OSR | 1000 | 15.90 | 67.94 | 47.36 | 51.29 | 41.54 | 54.87 |
| FG_2C_2D | 5000 | 54.65 | 62.40 | 83.00 | 29.06 | 86.36 | 24.00 |
| GEARS_2C_2D_SI_OSR | 855 | 40.00 | 71.40 | 96.52 | 6.41 | 93.94 | 10.89 |
| MG_2C_2D | 5000 | 22.62 | 87.25 | 63.48 | 53.51 | 51.01 | 65.77 |
| UG_2C_2D_SI_OSR | 2100 | 17.10 | 86.25 | 58.25 | 56.05 | 51.15 | 62.45 |
| UG_2C_3D_SI_OSR | 1050 | 8.64 | 90.91 | 42.78 | 69.31 | 26.41 | 80.73 |
| UG_2C_5D_SI_OSR | 1890 | 4.58 | 88.18 | 81.51 | 28.19 | 52.27 | 58.35 |
