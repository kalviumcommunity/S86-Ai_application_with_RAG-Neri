# Source Citation & Attribution Examples

## Task 1 - Generated Answer With Citations

### Question

What should a technician do if abnormal vibration is detected?

### Answer

If abnormal vibration is detected, the technician should stop the
machine immediately and begin the approved inspection procedure [1].
The equipment should not be restarted until the inspection is
completed and the machine is considered safe [2].

---

## Task 2 - Citation Mapping

### [1]

- Source: vibration_manual.txt
- Chunk ID: vibration_manual.txt:0
- Chunk Index: 0
- Section: Document body

Original retrieved text:

> If abnormal vibration is detected, stop the machine and begin the
> approved inspection procedure.

### [2]

- Source: vibration_procedure.txt
- Chunk ID: vibration_procedure.txt:0
- Chunk Index: 0
- Section: Document body

Original retrieved text:

> Do not restart the equipment until the inspection has been
> completed and the machine is considered safe.

---

## Task 3 - Source Verification

Citation [1] was checked against the original retrieved chunk
from `vibration_manual.txt`.

The original chunk explicitly states that when abnormal vibration
is detected, the machine should be stopped and the approved
inspection procedure should be started.

Therefore, citation [1] is supported by the retrieved source.

Citation [2] was checked against the original retrieved chunk
from `vibration_procedure.txt`.

The original chunk states that the equipment should not be restarted
until inspection is complete and the machine is considered safe.

Therefore, citation [2] is supported by the retrieved source.

---

## Task 4 - Missing Context / No-Source Fallback

### Question

What lubricant should be used for this machine?

### Result

I don't have enough information in the provided context to answer
this question.

### Citations

None.

No citation was generated because the provided retrieved context
does not contain sufficient information about the required lubricant.

---

## Task 5 - Fabricated Citation Protection

The citation validation checks every citation marker in the generated
answer against the citation map created from the retrieved chunks.

For example:

Valid:

[1] -> vibration_manual.txt:0

Invalid:

[99] -> No matching retrieved chunk

If a citation does not correspond to an actual retrieved chunk,
the answer is rejected and the safe fallback response is returned.

---

## Citation Flow

Question
    |
    v
Retrieve chunks
    |
    v
Build citation map
    |
    +---- [1] -> source chunk 1
    |
    +---- [2] -> source chunk 2
    |
    v
Generate answer
    |
    v
Validate citation markers
    |
    v
Verify citations against original chunks
    |
    v
Return answer + sources