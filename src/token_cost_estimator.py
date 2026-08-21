from dataclasses import dataclass
from pathlib import Path

import tiktoken


ROOT_DIR = Path(__file__).resolve().parents[1]
ENCODING = tiktoken.get_encoding("cl100k_base")
INPUT_RATE_PER_1K = 0.0005
OUTPUT_RATE_PER_1K = 0.0015


@dataclass(frozen=True)
class Sample:
    name: str
    kind: str
    text: str


def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def extract_paragraph(text: str, paragraph_index: int) -> str:
    paragraphs = [block.strip() for block in text.split("\n\n") if block.strip()]
    if paragraph_index >= len(paragraphs):
        raise IndexError("Paragraph index is out of range.")
    return paragraphs[paragraph_index]


def extract_block(text: str, start_marker: str, end_marker: str) -> str:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    return text[start:end].strip()


def format_ratio(characters: int, tokens: int) -> str:
    if tokens == 0:
        return "n/a"
    return f"{characters / tokens:.2f} chars/token"


def load_samples() -> list[Sample]:
    readme_text = (ROOT_DIR / "README.md").read_text(encoding="utf-8")
    prompt_text = (ROOT_DIR / "prompts" / "prompt_comparison.py").read_text(encoding="utf-8")
    llm_text = (ROOT_DIR / "src" / "llm_api.py").read_text(encoding="utf-8")
    result_text = (ROOT_DIR / "outputs" / "prompt_comparison.log").read_text(encoding="utf-8")

    return [
        Sample(
            name="Short prompt",
            kind="input",
            text="Explain our refund policy.",
        ),
        Sample(
            name="README paragraph",
            kind="input",
            text=extract_paragraph(readme_text, 3),
        ),
        Sample(
            name="LLM request block",
            kind="input",
            text=extract_block(
                llm_text,
                "messages = [",
                "try:",
            ),
        ),
        Sample(
            name="Saved response",
            kind="output",
            text=extract_block(
                result_text,
                "RESPONSE 2",
                "COMPARISON",
            ),
        ),
        Sample(
            name="System prompt block",
            kind="input",
            text=extract_block(
                prompt_text,
                'system_prompt = """',
                '# --------------------------------------------------\n# Task 3:',
            ),
        ),
    ]


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000 * INPUT_RATE_PER_1K) + (output_tokens / 1000 * OUTPUT_RATE_PER_1K)


def main() -> None:
    samples = load_samples()

    print("NERI Token Counting & Cost Estimation")
    print("=" * 44)
    print("Tokenizer: cl100k_base")
    print(f"Input rate:  ${INPUT_RATE_PER_1K:.4f} per 1K tokens")
    print(f"Output rate: ${OUTPUT_RATE_PER_1K:.4f} per 1K tokens")
    print()

    input_samples = [sample for sample in samples if sample.kind == "input"]
    output_samples = [sample for sample in samples if sample.kind == "output"]

    print("SAMPLE TOKEN COUNTS")
    print("-" * 44)
    for sample in samples:
        characters = len(sample.text)
        tokens = count_tokens(sample.text)
        print(f"{sample.name}")
        print(f"  kind: {sample.kind}")
        print(f"  chars: {characters}")
        print(f"  tokens: {tokens}")
        print(f"  length/token ratio: {format_ratio(characters, tokens)}")
        print(f"  preview: {sample.text.replace(chr(10), ' ')[:110]}")
        print()

    total_input_tokens = sum(count_tokens(sample.text) for sample in input_samples)
    total_output_tokens = sum(count_tokens(sample.text) for sample in output_samples)
    estimated_total_cost = estimate_cost(total_input_tokens, total_output_tokens)

    print("COST ESTIMATE")
    print("-" * 44)
    print(f"input tokens:  {total_input_tokens}")
    print(f"output tokens: {total_output_tokens}")
    print(f"estimated cost: ${estimated_total_cost:.6f}")
    print()

    print("LENGTH VS TOKEN RELATIONSHIP")
    print("-" * 44)
    ranked_samples = sorted(samples, key=lambda sample: len(sample.text))
    for sample in ranked_samples:
        characters = len(sample.text)
        tokens = count_tokens(sample.text)
        print(f"{sample.name:<20} chars={characters:<4} tokens={tokens:<4} chars/token={format_ratio(characters, tokens)}")

    print()
    print("Interpretation: longer text generally uses more tokens, but the ratio changes across prose, code,")
    print("and quoted responses because tokenization splits text into word pieces rather than characters.")


if __name__ == "__main__":
    main()