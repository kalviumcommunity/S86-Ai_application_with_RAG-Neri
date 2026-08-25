# Document Chunking Comparison

Parameters: fixed-size chunks use 500 characters with 50 characters of overlap.

## Corpus Statistics

| Strategy | Chunk count | Average chunk size (characters) |
| --- | ---: | ---: |
| fixed | 4 | 358.5 |
| paragraph | 26 | 51.5 |

## Per-Document Statistics

| Source | Strategy | Chunks | Average characters |
| --- | --- | ---: | ---: |
| machine_manual.txt | fixed | 2 | 321.0 |
| machine_manual.txt | paragraph | 11 | 52.0 |
| maintenance_log.txt | fixed | 1 | 397.0 |
| maintenance_log.txt | paragraph | 9 | 42.3 |
| safety_procedure.md | fixed | 1 | 395.0 |
| safety_procedure.md | paragraph | 6 | 64.2 |

## Sample Chunks

### Fixed strategy

#### machine_manual.txt - chunk 0
Source path: `machine_manual.txt`

```text
ACME MANUFACTURING - EQUIPMENT MANUAL

Machine Maintenance Manual

The motor should be inspected every 30 days.

Technicians must check the motor housing for visible damage.

The recommended operating temperature is 80°C.

ACME MANUFACTURING - EQUIPMENT MANUAL

Before performing maintenance, disconnect the machine from
the main power supply.

Inspect the lubrication system and check for leaks.

ACME MANUFACTURING - EQUIPMENT MANUAL

Record all maintenance activities in the maintenance log.

If u
```

### Paragraph strategy

#### machine_manual.txt - chunk 0
Source path: `machine_manual.txt`

```text
ACME MANUFACTURING - EQUIPMENT MANUAL
```
