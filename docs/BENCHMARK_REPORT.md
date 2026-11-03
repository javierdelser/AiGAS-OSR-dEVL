# Benchmark report — stream tables (MINAS, KNNENS, SENCForest)

Final **table** metrics bundled in this release. Figures and static AiGAS-dEVL-OSR rescoring are out of scope here.

**EVL protocol:** [arXiv:2504.05761](https://arxiv.org/abs/2504.05761) (Table I: `T_s`, batches, datasets). Implementation: `src/osr_static/stream_benchmark.py`, `stream_eval.py`, `synthetic_metrics.py`.

---

## 1. Methods

| Method | Release module |
|--------|----------------|
| **MINAS** | `baselines/minas.py` |
| **KNNENS** | `baselines/knnens.py` |
| **SENCForest** | `baselines/sencforest.py` |

---

## 2. Letter / Pendigits — static OSR partitions

Micro-F1 (%). Partition logs: `data/osr_static/*.log`.

| Setting | MINAS | KNNENS | SENCForest |
|---------|-------|--------|------------|
| Letter 0 | 79.27 | 57.62 | 58.87 |
| Letter 15.48 | 23.78 | 46.17 | 44.12 |
| Letter 25.46 | 26.55 | 38.36 | 30.17 |
| Pendigits 0 | 96.95 | 89.71 | 89.63 |
| Pendigits 8.71 | 45.83 | 70.63 | 72.77 |
| Pendigits 18.35 | 47.14 | 61.64 | 65.72 |

**Run:** `bash launch.sh static --cpus 8` or `scripts/osr_static/run_benchmark.py` → `results/osr_static/STREAM_BASELINES.md`

---

## 3. Fourteen EVL streams — Table I prequential protocol

**Data:** `data/osr_dynamic/*.txt`

**Run:** `bash launch.sh dynamic` or `scripts/osr_dynamic/run_benchmark.py` → `results/osr_dynamic/SYNTHETIC_STREAM_TABLE.md`

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
| GEARS_2C_2D_SI_OSR | 855 | 40.00 | 71.40 | 96.52 | 6.41 | 93.94 | 10.89 |
| UG_2C_2D_SI_OSR | 2100 | 17.10 | 86.25 | 58.25 | 56.05 | 51.15 | 62.45 |
| UG_2C_3D_SI_OSR | 1050 | 8.64 | 90.91 | 42.78 | 69.31 | 26.41 | 80.73 |
| UG_2C_5D_SI_OSR | 1890 | 4.58 | 88.18 | 81.51 | 28.19 | 52.27 | 58.35 |

FG2C2D / MG2C2D appear in Table I but are not in the shipped CSV set.

---

## 4. Key paths

| Item | Path |
|------|------|
| Static stream config | `configs/baselines_stream.yaml` |
| Static benchmark | `scripts/osr_static/run_benchmark.py` |
| Dynamic benchmark | `scripts/osr_dynamic/run_benchmark.py` |
| Table I layout | `src/osr_static/stream_benchmark.py` |
