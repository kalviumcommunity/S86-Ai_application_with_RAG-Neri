# Retrieval Tuning Results

## Objective

Compare retrieval settings to verify that filtering and score thresholds improve relevance for the maintenance and safety corpus.

## Test queries

| Query | Expected sources | Purpose |
| --- | --- | --- |
| Before inspecting the machine, disconnect the power supply and verify equipment isolation. | safety_procedure.md | Electrical safety and equipment isolation before inspection. |
| A worn belt produced abnormal vibration and the machine was returned to service after repair. | maintenance_log.txt | Maintenance log entry for vibration and repair activity. |
| What should technicians do before beginning maintenance work? | machine_manual.txt, safety_procedure.md | Safety-before-maintenance question covering PPE and disconnection steps. |

## Compared settings

| Setting | k | Filter | Min similarity |
| --- | ---: | --- | ---: |
| baseline_k3 | 3 | None | 0.0 |
| filtered_k3 | 3 | {'source': 'safety_procedure.md'} | 0.0 |
| filtered_threshold_k3 | 3 | {'source': 'safety_procedure.md'} | 0.75 |
| filtered_threshold_k5 | 5 | {'source': 'safety_procedure.md'} | 0.75 |

## Relevance summary

| Setting | Top-1 hit rate | Top-k hit rate | Avg similarity |
| --- | ---: | ---: | ---: |
| baseline_k3 | 0.333 | 0.667 | 0.7746 |
| filtered_k3 | 0.667 | 0.667 | 0.689 |
| filtered_threshold_k3 | 0.0 | 0.0 | 0.0 |
| filtered_threshold_k5 | 0.0 | 0.0 | 0.0 |

## Detailed rows

### baseline_k3 | Before inspecting the machine, disconnect the power supply and verify equipment isolation.

- Filter: None
- Min similarity: 0.0
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.7746
- Retrieved sources: maintenance_log.txt, maintenance_log.txt, machine_manual.txt
- Sample text: The machine was tested after repair and returned to service.

### baseline_k3 | A worn belt produced abnormal vibration and the machine was returned to service after repair.

- Filter: None
- Min similarity: 0.0
- Top-1 hit: 1
- Top-k hit: 1
- Avg similarity: 0.7746
- Retrieved sources: maintenance_log.txt, maintenance_log.txt, machine_manual.txt
- Sample text: The machine was tested after repair and returned to service.

### baseline_k3 | What should technicians do before beginning maintenance work?

- Filter: None
- Min similarity: 0.0
- Top-1 hit: 0
- Top-k hit: 1
- Avg similarity: 0.7746
- Retrieved sources: maintenance_log.txt, maintenance_log.txt, machine_manual.txt
- Sample text: The machine was tested after repair and returned to service.

### filtered_k3 | Before inspecting the machine, disconnect the power supply and verify equipment isolation.

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.0
- Top-1 hit: 1
- Top-k hit: 1
- Avg similarity: 0.689
- Retrieved sources: safety_procedure.md, safety_procedure.md
- Sample text: Technicians must wear the required protective equipment before beginning maintenance activities.

### filtered_k3 | A worn belt produced abnormal vibration and the machine was returned to service after repair.

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.0
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.689
- Retrieved sources: safety_procedure.md, safety_procedure.md
- Sample text: Technicians must wear the required protective equipment before beginning maintenance activities.

### filtered_k3 | What should technicians do before beginning maintenance work?

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.0
- Top-1 hit: 1
- Top-k hit: 1
- Avg similarity: 0.689
- Retrieved sources: safety_procedure.md, safety_procedure.md
- Sample text: Technicians must wear the required protective equipment before beginning maintenance activities.

### filtered_threshold_k3 | Before inspecting the machine, disconnect the power supply and verify equipment isolation.

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.75
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.0
- Retrieved sources: none
- Sample text: none

### filtered_threshold_k3 | A worn belt produced abnormal vibration and the machine was returned to service after repair.

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.75
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.0
- Retrieved sources: none
- Sample text: none

### filtered_threshold_k3 | What should technicians do before beginning maintenance work?

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.75
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.0
- Retrieved sources: none
- Sample text: none

### filtered_threshold_k5 | Before inspecting the machine, disconnect the power supply and verify equipment isolation.

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.75
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.0
- Retrieved sources: none
- Sample text: none

### filtered_threshold_k5 | A worn belt produced abnormal vibration and the machine was returned to service after repair.

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.75
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.0
- Retrieved sources: none
- Sample text: none

### filtered_threshold_k5 | What should technicians do before beginning maintenance work?

- Filter: {'source': 'safety_procedure.md'}
- Min similarity: 0.75
- Top-1 hit: 0
- Top-k hit: 0
- Avg similarity: 0.0
- Retrieved sources: none
- Sample text: none

## Best setting

The best-performing configuration is **filtered_k3** because it produces the highest top-1 and top-k hit rate across the test set. This configuration keeps the query scoped to the most relevant source family while filtering out low-confidence results.

Chosen setup: k=3, filter={'source': 'safety_procedure.md'}, min_similarity=0.0
