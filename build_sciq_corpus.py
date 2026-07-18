"""
build_sciq_corpus.py
---------------------
Downloads the SciQ dataset (real science exam questions - physics, chemistry,
biology, earth science - each with a correct answer + explanation) and
flattens each example into a single line of the form:

    Question: <question> Answer: <answer>. Explanation: <support>

ready for:
    python data_cleaning.py --mode decoder --unit lines --source sciq_corpus.txt

Run:
    pip install datasets
    python build_sciq_corpus.py
"""

from datasets import load_dataset


def main():
    print("Loading allenai/sciq from Hugging Face ...")
    ds = load_dataset("allenai/sciq")["train"]

    written = 0
    with open("sciq_corpus.txt", "w", encoding="utf-8") as f:
        for ex in ds:
            question = (ex["question"] or "").strip()
            answer = (ex["correct_answer"] or "").strip()
            support = (ex["support"] or "").strip()

            if not question or not answer:
                continue

            line = f"Question: {question} Answer: {answer}."
            if support:
                line += f" Explanation: {support}"

            # collapse any embedded newlines/extra whitespace - this pipeline
            # treats one line as one training example
            line = " ".join(line.split())

            f.write(line + "\n")
            written += 1

    print(f"Wrote {written} examples to sciq_corpus.txt")


if __name__ == "__main__":
    main()