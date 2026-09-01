# Chunk Re-Ranking Experiment

**Query:** What should a technician do if abnormal vibration is detected?

**Candidate set:** 10

**Final k:** 3

## Before Re-Ranking

| Rank | Vector Score | Source | Section | Chunk |
| ---: | ---: | --- | --- | ---: |
| 1 | 0.821783 | vibration_manual.txt | Document body | 0 |
| 2 | 0.800400 | vibration_procedure.txt | Document body | 0 |
| 3 | 0.798303 | machine_manual.txt | Document body | 2 |
| 4 | 0.745612 | vibration_incident.txt | Document body | 0 |
| 5 | 0.705493 | maintenance_log.txt | Document body | 1 |
| 6 | 0.700880 | temperature_procedure.txt | Document body | 0 |
| 7 | 0.693093 | bearing_inspection.txt | Document body | 0 |
| 8 | 0.676183 | motor_inspection.txt | Document body | 0 |
| 9 | 0.662201 | preventive_maintenance.txt | Document body | 0 |
| 10 | 0.662176 | ppe_safety.txt | Document body | 0 |

## After Re-Ranking

| Final Rank | Original Rank | Vector Score | Keyword Score | Phrase Score | Re-Rank Score | Source |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1 | 0.821783 | 0.750000 | 0.333333 | 0.730570 | vibration_manual.txt |
| 2 | 4 | 0.745612 | 0.750000 | 0.333333 | 0.684867 | vibration_incident.txt |
| 3 | 5 | 0.705493 | 0.750000 | 0.333333 | 0.660796 | maintenance_log.txt |
| 4 | 2 | 0.800400 | 0.500000 | 0.000000 | 0.605240 | vibration_procedure.txt |
| 5 | 3 | 0.798303 | 0.500000 | 0.000000 | 0.603982 | machine_manual.txt |
| 6 | 6 | 0.700880 | 0.250000 | 0.000000 | 0.483028 | temperature_procedure.txt |
| 7 | 7 | 0.693093 | 0.250000 | 0.000000 | 0.478356 | bearing_inspection.txt |
| 8 | 8 | 0.676183 | 0.000000 | 0.000000 | 0.405710 | motor_inspection.txt |
| 9 | 9 | 0.662201 | 0.000000 | 0.000000 | 0.397320 | preventive_maintenance.txt |
| 10 | 10 | 0.662176 | 0.000000 | 0.000000 | 0.397305 | ppe_safety.txt |

## Final Selected Chunks

### Rank 1

**Source:** vibration_manual.txt

**Section:** Document body

**Original rank:** 1

**Re-rank score:** 0.730570

Machine Vibration Manual  Abnormal vibration can indicate a problem with machine components. Inspect the machine for unusual movement, noise, loose components, and visible damage.  If abnormal vibration is detected, stop the machine and begin the approved inspection procedure.

### Rank 2

**Source:** vibration_incident.txt

**Section:** Document body

**Original rank:** 4

**Re-rank score:** 0.684867

Maintenance Incident Report  A technician reported abnormal vibration during machine operation.  The machine was stopped and isolated before inspection. The maintenance team inspected the drive system and identified wear in a mechanical component.  The incident was recorded in the maintenance log.

### Rank 3

**Source:** maintenance_log.txt

**Section:** Document body

**Original rank:** 5

**Re-rank score:** 0.660796

DEPARTMENT - DAILY LOG  The technician replaced a worn belt after detecting abnormal vibration.  Replacement part number: 4521.  The machine was tested after repair and returned to service.

## Trade-off

Initial vector retrieval is efficient because it searches the embedding space.

Re-ranking adds a second scoring stage over only the candidate set. This can improve precision, but it adds computation and latency.

A practical approach is to retrieve a larger candidate set and then keep only the highest-scoring final chunks.
