# Sample Grounded Answers

## Task 1: Generate from Injected Context

### Question

What should a technician do if abnormal vibration is detected?

### Supporting Chunks

#### Source: vibration_manual.txt

Machine Vibration Manual

Abnormal vibration can indicate a problem with machine components.
Inspect the machine for unusual movement, noise, loose components,
and visible damage.

If abnormal vibration is detected, stop the machine and begin the
approved inspection procedure.

#### Source: vibration_procedure.txt

Vibration Inspection Procedure

If unusual vibration is detected during machine operation, stop the
equipment immediately.

Inspect the machine for loose components, damaged bearings, worn
belts, and other mechanical problems.

Do not restart the equipment until the inspection has been completed
and the machine is considered safe.

#### Source: machine_manual.txt

If unusual vibration is detected, stop the machine and follow the
approved inspection procedure.

### Generated Answer

If abnormal vibration is detected, the technician should stop the
machine immediately and begin the approved inspection procedure,
which includes checking for loose components, damaged bearings,
worn belts, and other mechanical problems. The equipment should not
be restarted until the inspection is complete and the machine is
considered safe.

### Sources

- vibration_manual.txt
- vibration_procedure.txt
- machine_manual.txt

### Grounding

Grounded: True

---

## Task 2: Source Accuracy Check

### Question

What should be inspected when abnormal vibration is detected?

### Generated Answer

When abnormal vibration is detected, you should inspect the machine
for unusual movement, noise, loose components, visible damage,
damaged bearings, and worn belts.

### Accuracy Check

The answer is supported by the retrieved chunks.

The vibration manual supports inspection for:

- unusual movement
- noise
- loose components
- visible damage

The vibration inspection procedure supports inspection for:

- loose components
- damaged bearings
- worn belts
- other mechanical problems

No unsupported claim was added to the answer.

### Sources

- vibration_manual.txt
- vibration_procedure.txt

### Grounding

Grounded: True

---

## Task 3: Missing-Context Fallback

### Question

Who is the president of India?

### Output

I don't have enough information in the provided context.

### Sources

None

### Grounding

Grounded: False

### Result

The system correctly avoided generating an answer because the
retrieved context does not contain information about the question.