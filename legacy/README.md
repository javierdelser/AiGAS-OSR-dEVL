# Legacy AiGAS drivers

| Path | Role |
|------|------|
| `aigas_osr_devl/` | Prequential **AiGAS-OSR-dEVL** stream experiment (`Aigas_OSR_Open.py`, GNG, `ClassTracker`) |
| `osr_GNG_Train_VAl_Test_LOOP_Split26.py` | Phase-1 Growing Neural Gas training (NeuPy, Python 3.8) |
| `osr_utils.py` | Metrics and helpers used by the GNG trainer |

Static rescoring for the paper uses **`src/osr_static/`** + **`src/aigas_osr_devl/static/`** (no NeuPy).
Stream evaluation uses **`legacy/aigas_osr_devl/run_aigas_stream.py`**.