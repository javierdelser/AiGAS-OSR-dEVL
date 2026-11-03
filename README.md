# AiGAS-OSR-dEVL: Incremental Open Set Recognition in Non-stationary Data Streams under Extreme Verification Latency

**Code release** — reproduce stream baseline tables (MINAS, KNNENS, SENCForest) on static OSR partitions (LETTER, PENDIGITS) and on fourteen EVL prequential streams from the paper experiments.

> Shipped **`results/`** contains **markdown tables only**. Each benchmark is a **single script**; intermediate JSON stays in a temp directory and is not written under `results/`.

---

## Authors

**Maria Arostegi** · [ORCID 0000-0001-6691-4276](https://orcid.org/0000-0001-6691-4276)  
TECNALIA, Basque Research and Technology Alliance (BRTA), 48160 Derio, Spain

**Miren Nekane Bilbao** · [ORCID 0000-0002-0519-8728](https://orcid.org/0000-0002-0519-8728)  
University of the Basque Country (EHU), 48940 Leioa, Spain

**Javier Del Ser** · [ORCID 0000-0002-1260-9775](https://orcid.org/0000-0002-1260-9775) · *Corresponding author* · [javier.delser@tecnalia.com](mailto:javier.delser@tecnalia.com)  
TECNALIA, BRTA, Derio, Spain · University of the Basque Country (EHU), Leioa, Spain

---

## Abstract

In many practical applications, batches of data arrive as a stream whose classes can drift over time. Labels are often given only during a short initial period, whereas new classes, such as unseen fault patterns in industrial machinery or new types of attacks in network security, can appear at any later time. A model that forces such samples into a known class gives a confident wrong answer instead of raising an alert. Open Set Recognition (OSR) methods assume stationary data, whereas stream learners for extreme verification latency assume that no new classes appear in the stream. Without label verification, a known class that has moved as a result of the drift and a new emerging class can look alike. In this context we present the Adaptive Incremental neural GAS model for Open Set Recognition in drifting data streams under Extreme Verification Latency (AiGAS-OSR-dEVL). The model extends our earlier AiGAS-dEVL-RC with an open set stage. On each unlabeled batch, this stage compares the clusters of the current batch with those of the previous one, decides whether a cluster has appeared, split, merged or vanished, and rejects the samples that fall outside every cluster. Against fifteen OSR, novelty detection and stream learning methods, AiGAS-OSR-dEVL achieves the highest micro F1 on fourteen of sixteen synthetic streams with emerging classes. On the fourteen streams where all methods run, it beats every competitor on every stream, with statistical significance as per several hypothesis tests and Bayesian analysis. On static OSR benchmarks, a simplified variant ranks unknown samples at least as well as the best baselines in three of four open settings, and it explains each rejection through exact per-feature contributions. Overall, these results show that the geometry of unlabeled data alone can separate drift from novelty, so a stream classifier can keep labeling known classes and flag new ones long after the last label has arrived. Code: [https://github.com/javierdelser/AiGAS-OSR-dEVL](https://github.com/javierdelser/AiGAS-OSR-dEVL).

---

| | |
|---|---|
| **Journal** | *Applied Soft Computing* (submitted) |
| **Benchmark tables** | [docs/BENCHMARK_REPORT.md](docs/BENCHMARK_REPORT.md) |
| **Repository** | [github.com/javierdelser/AiGAS-OSR-dEVL](https://github.com/javierdelser/AiGAS-OSR-dEVL) |

---

## What you can reproduce

| Benchmark | Data | Command | Output |
|-----------|------|---------|--------|
| Static (Letter / Pendigits) | `data/osr_static/` | `bash launch.sh static --cpus 8` | `results/osr_static/STREAM_BASELINES.md` |
| Dynamic (14 EVL streams) | `data/osr_dynamic/*.txt` | `bash launch.sh dynamic` | `results/osr_dynamic/SYNTHETIC_STREAM_TABLE.md` |

| Driver (direct) | |
|-----------------|---|
| Static | `python scripts/osr_static/run_benchmark.py --cpus 8` |
| Dynamic | `python scripts/osr_dynamic/run_benchmark.py` |

**Requirements:** Linux, Python 3.12. First time: `bash launch.sh setup`.

---

## Quick start

```bash
bash launch.sh setup
bash launch.sh static --cpus 8
bash launch.sh dynamic
```

---

## Repository layout

```
├── data/
│   ├── osr_static/              LETTER & PENDIGITS + partitions/*.json
│   └── osr_dynamic/             14 EVL stream CSVs (*.txt)
├── scripts/
│   ├── osr_static/run_benchmark.py
│   └── osr_dynamic/run_benchmark.py
├── src/osr_static/
├── configs/
├── results/
│   ├── osr_static/STREAM_BASELINES.md
│   └── osr_dynamic/SYNTHETIC_STREAM_TABLE.md
└── launch.sh
```

---

## Cite this work

BibTeX key: `aigas_osr_devl_2026`. Copy the entry below into your `.bib` file:

```bibtex
@article{aigas_osr_devl_2026,
  title   = {AiGAS-OSR-dEVL: Incremental Open Set Recognition in Non-stationary Data Streams under Extreme Verification Latency},
  author  = {Arostegi, Mar{\'i}a and Bilbao, Miren Nekane and {Del Ser}, Javier},
  journal = {Applied Soft Computing},
  year    = {2026},
  note    = {Submitted. Code: \url{https://github.com/javierdelser/AiGAS-OSR-dEVL}},
  url     = {https://github.com/javierdelser/AiGAS-OSR-dEVL}
}
```

---

## Maintainers

```bash
bash scripts/build_release.sh
```
