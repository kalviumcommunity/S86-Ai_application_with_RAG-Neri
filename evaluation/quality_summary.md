# RAG Answer Quality Evaluation

Offline deterministic evaluation using the production query pipeline and corpus evidence.

- Cases: 4
- Correctness: 100.0%
- Grounding: 100.0%
- Citation quality: 75.0%
- Perfect cases: 3/4

## Results

| Case | Correctness | Grounding | Citation quality | Retrieved sources |
| --- | ---: | ---: | ---: | --- |
| electrical-preinspection | 100.0% | 100.0% | 100.0% | electrical_safety.txt, temperature_procedure.txt, vibration_procedure.txt |
| vibration-response | 100.0% | 100.0% | 100.0% | vibration_procedure.txt |
| temperature-log | 100.0% | 100.0% | 100.0% | temperature_procedure.txt, vibration_procedure.txt |
| citation-failure | 100.0% | 100.0% | 0.0% | vibration_procedure.txt |

## Failures

- **citation-failure**: citation was missing, invalid, or unsupported
