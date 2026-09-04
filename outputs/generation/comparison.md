# With Retrieval vs Without Retrieval

## Question

What should a technician do if abnormal vibration is detected?

---

## With Retrieval

### Retrieved Sources

- vibration_manual.txt
- vibration_procedure.txt
- machine_manual.txt

### Answer

If abnormal vibration is detected, the technician should stop the
machine immediately and begin the approved inspection procedure,
which includes checking for loose components, damaged bearings,
worn belts, and other mechanical problems. The equipment should not
be restarted until the inspection is complete and the machine is
considered safe.

### Grounded

True

### Why?

The answer is generated using the retrieved chunks from the
maintenance corpus. The instructions and inspection steps are
supported by the retrieved source material.

---

## Without Retrieval

### Context

No retrieved documents are provided to the model.

### Answer

I don't have enough information in the provided context.

### Grounded

False

---

## Comparison

| Aspect | With Retrieval | Without Retrieval |
|---|---|---|
| Retrieved context | Yes | No |
| Supporting sources | vibration_manual.txt, vibration_procedure.txt, machine_manual.txt | None |
| Grounded answer | Yes | No |
| Source attribution | Yes | No |
| Uses corpus information | Yes | No |
| Hallucination risk | Reduced by grounding | Higher if unrestricted generation is allowed |

## Conclusion

Retrieval supplies the model with relevant source material before
generation. This allows the generated answer to be grounded in the
maintenance documents and provides source attribution.

Without retrieved context, the grounded generation pipeline cannot
support the answer from the corpus and therefore falls back instead
of inventing information.