from pathlib import Path
import re

DATASET = Path("data/raw/mlops_knowledge.md")

REQUIRED_SECTIONS = [
    "What is MLOps?",
    "What is DVC?",
    "What is ChromaDB?",
    "What is RAG?",
    "What is FastAPI?",
]


def validate_content(content):
    headings = re.findall(
        r"(?m)^#{1,6}\s+.+$",
        content
    )

    sections = list(
        re.finditer(r"(?m)^##\s+.+$", content)
    )

    empty_sections = []

    for index, section in enumerate(sections):
        start = section.end()
        end = (
            sections[index + 1].start()
            if index + 1 < len(sections)
            else len(content)
        )

        if not content[start:end].strip():
            empty_sections.append(section.group().strip())

    normalized_headings = [
        heading.strip().casefold()
        for heading in headings
    ]

    return {
        "File not empty": bool(content.strip()),
        "Has main heading": bool(
            re.search(r"(?m)^#\s+.+$", content)
        ),
        "Has knowledge sections": len(sections) > 0,
        "No duplicate headings": (
            len(normalized_headings)
            == len(set(normalized_headings))
        ),
        "No empty sections": len(empty_sections) == 0,
        "Required sections present": all(
            f"## {name}" in content
            for name in REQUIRED_SECTIONS
        ),
    }


def validate_dataset(path=DATASET):
    if not path.exists():
        raise FileNotFoundError(path)

    content = path.read_text(encoding="utf-8")
    checks = validate_content(content)

    print("===== DATASET QUALITY REPORT =====")

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    print(f"\nCharacters: {len(content)}")
    print(f"Words: {len(content.split())}")

    if not all(checks.values()):
        raise SystemExit("Dataset validation FAILED")

    print("\nDATASET VALIDATION SUCCESSFUL")


if __name__ == "__main__":
    validate_dataset()
