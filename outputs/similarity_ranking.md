# Embedding Similarity Ranking

## Metric justification

This demo uses cosine similarity. It compares the direction of embedding vectors, which is useful for semantic text matching when vector magnitude should not dominate. Higher scores indicate greater similarity; distance metrics reverse that interpretation, where lower scores are better.

## Query: What should I do before inspecting the machine?

| Rank | Cosine score | Source | Chunk | Section | Text |
| ---: | ---: | --- | ---: | --- | --- |
| 1 | 0.784660 | safety_procedure.md | 0 | Document body | SAFETY PROCEDURE - MACHINE MAINTENANCE  # Electrical Safety  Before inspecting or repairing equipment, disconnect the main power supply and verify that the equipment is isolated.  # Personal Protective Equipment  Technicians must wear the required protective equipment before beginning maintenance activities.  Keep the work area clear and report unsafe |
| 2 | 0.752880 | machine_manual.txt | 2 | Document body | .  If unusual vibration is detected, stop the machine and follow the approved inspection procedure. |
| 3 | 0.734160 | reranking_demo.txt | 1 | Document body | : Lubrication  Inspect the lubrication system regularly. Check for leaks and verify that lubricant levels are within the recommended range.  ---  DOCUMENT 3 Source: safety_procedure.md Section: Electrical Safety  Before inspecting or repairing equipment, disconnect the main power supply. Verify that the equipment is completely isolated |
| 4 | 0.728564 | vibration_procedure.txt | 0 | Document body | Vibration Inspection Procedure  If unusual vibration is detected during machine operation, stop the equipment immediately.  Inspect the machine for loose components, damaged bearings, worn belts, and other mechanical problems.  Do not restart the equipment until the inspection has been completed and the machine is considered safe. |
| 5 | 0.728348 | electrical_safety.txt | 0 | Document body | Electrical Safety Procedure  Before inspecting or repairing equipment, disconnect the main power supply.  Verify that the equipment is completely isolated before beginning maintenance.  Technicians must follow the approved electrical safety procedure. |
| 6 | 0.721312 | machine_manual.txt | 1 | Document body | URING - EQUIPMENT MANUAL  Before performing maintenance, disconnect the machine from the main power supply.  Inspect the lubrication system and check for leaks.  ACME MANUFACTURING - EQUIPMENT MANUAL  Record all maintenance activities in the maintenance log.  If unusual vibration is detected, stop the machine and follow the approved inspection |
| 7 | 0.719916 | reranking_demo.txt | 6 | Document body | . Keep the work area clear during inspection and repair.  ---  DOCUMENT 9 Source: machine_manual.txt Section: Temperature  Monitor the operating temperature of the machine. If the temperature exceeds the recommended range, stop the equipment and investigate the cause.  ---  DOCUMENT 10 Source: maintenance_log.txt Section: Bearing |
| 8 | 0.716538 | vibration_manual.txt | 0 | Document body | Machine Vibration Manual  Abnormal vibration can indicate a problem with machine components. Inspect the machine for unusual movement, noise, loose components, and visible damage.  If abnormal vibration is detected, stop the machine and begin the approved inspection procedure. |
| 9 | 0.715759 | reranking_demo.txt | 2 | Document body | repairing equipment, disconnect the main power supply. Verify that the equipment is completely isolated before beginning work.  ---  DOCUMENT 4 Source: maintenance_log.txt Section: Vibration Incident  The technician detected abnormal vibration during operation. The machine was stopped and an inspection procedure was initiated.  ---  DOCUMENT 5 Source: machine |
| 10 | 0.714372 | preventive_maintenance.txt | 0 | Document body | Preventive Maintenance  Perform scheduled preventive maintenance according to the equipment maintenance schedule.  Inspect important machine components regularly and record completed maintenance activities.  Preventive maintenance helps identify potential equipment problems before failure occurs. |
| 11 | 0.713637 | safety_procedure.md | 1 | Personal Protective Equipment | protective equipment before beginning maintenance activities.  Keep the work area clear and report unsafe conditions to the maintenance supervisor. |
| 12 | 0.704653 | reranking_demo.txt | 0 | Document body | DOCUMENT 1 Source: machine_manual.txt Section: Motor Inspection  Inspect the motor housing for visible damage before starting maintenance. Check the motor for unusual sounds or physical damage.  ---  DOCUMENT 2 Source: machine_manual.txt Section: Lubrication  Inspect the lubrication system regularly. Check for leaks and |
| 13 | 0.704479 | machine_manual.txt | 0 | Document body | ACME MANUFACTURING - EQUIPMENT MANUAL  Machine Maintenance Manual  The motor should be inspected every 30 days.  Technicians must check the motor housing for visible damage.  The recommended operating temperature is 80°C.  ACME MANUFACTURING - EQUIPMENT MANUAL  Before performing maintenance, disconnect the machine from the |
| 14 | 0.703781 | equipment_restart.txt | 0 | Document body | Equipment Restart Procedure  After maintenance is completed, verify that all tools have been removed from the machine.  Confirm that safety checks have been completed before restarting the equipment.  Only restart the machine when the equipment is safe to operate. |
| 15 | 0.684110 | reranking_demo.txt | 3 | Document body | stopped and an inspection procedure was initiated.  ---  DOCUMENT 5 Source: machine_manual.txt Section: Vibration Inspection  If unusual vibration is detected, stop the machine immediately. Follow the approved inspection procedure before restarting the equipment.  ---  DOCUMENT 6 Source: maintenance_log.txt Section: Belt Replacement  A worn |
| 16 | 0.682983 | lubrication.txt | 0 | Document body | Lubrication Procedure  Inspect the lubrication system regularly.  Check lubricant levels and inspect the equipment for leaks.  Use the approved lubricant specified for the equipment. |
| 17 | 0.681687 | temperature_procedure.txt | 0 | Document body | Temperature Monitoring  Monitor the operating temperature of the machine during operation.  If the temperature exceeds the recommended range, stop the equipment and investigate the cause.  Record abnormal temperature readings in the maintenance log. |
| 18 | 0.674603 | reranking_demo.txt | 5 | Document body | ive Maintenance  Perform scheduled preventive maintenance according to the equipment maintenance schedule. Record completed maintenance activities.  ---  DOCUMENT 8 Source: safety_procedure.md Section: Personal Protective Equipment  Technicians must wear the required protective equipment before beginning maintenance. Keep the work area clear during inspection and repair.  ---  DOCUMENT 9 |
| 19 | 0.673218 | bearing_inspection.txt | 0 | Document body | Bearing Inspection  Inspect machine bearings for wear, damage, unusual noise, and abnormal movement.  Damaged bearings should be replaced according to the approved maintenance procedure.  Record bearing inspection results after maintenance. |
| 20 | 0.670736 | reranking_demo.txt | 8 | Document body | .txt Section: Equipment Restart  After maintenance is completed, verify that all tools have been removed. Confirm that safety checks are complete before restarting the machine.  ---  DOCUMENT 12 Source: safety_procedure.md Section: Emergency Shutdown  Use the emergency shutdown procedure when an immediate equipment hazard is identified. Notify the maintenance |

**Most similar:** SAFETY PROCEDURE - MACHINE MAINTENANCE

# Electrical Safety

Before inspecting or repairing equipment, disconnect the
main power supply and verify that the equipment is isolated.

# Personal Protective Equipment

Technicians must wear the required protective equipment
before beginning maintenance activities.

Keep the work area clear and report unsafe

**Least similar:** .txt
Section: Equipment Restart

After maintenance is completed, verify that all tools have been removed.
Confirm that safety checks are complete before restarting the machine.

---

DOCUMENT 12
Source: safety_procedure.md
Section: Emergency Shutdown

Use the emergency shutdown procedure when an immediate equipment hazard is identified.
Notify the maintenance

A high similarity score identifies likely relevant context. It does not guarantee that the chunk is factually correct, current, complete, or safe to use without metadata, citations, freshness checks, and answer validation.
