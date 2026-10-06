"""Create the portable code bundle without secrets, environments or model weights."""
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / "dist" / "assessment_code.zip"
    output.parent.mkdir(exist_ok=True)
    patterns = ["shared/*.py","task1_financial/*.py","task2_genai/*.py","task3_agentic/*.py",
                "tests/*.py","specs/*.md","scripts/*.py","task*/README.md","task*/*.ipynb",
                "task2_genai/MODEL_CARD_TEMPLATE.md","task1_financial/outputs/*_indicators.csv",
                "task1_financial/outputs/*_data_evidence.json", "task2_genai/outputs/tokenizer_check.json"]
    files = {ROOT/name for name in ["pyproject.toml","README.md","SETUP.md","PLAN.md","CITATIONS.md","REFLECTION.md","EVIDENCE.md","requirements-local.lock.txt"]}
    for pattern in patterns:
        files.update(ROOT.glob(pattern))
    with ZipFile(output,"w",ZIP_DEFLATED) as archive:
        for file in sorted(files):
            if file.is_file():
                archive.write(file,file.relative_to(ROOT).as_posix())
    print(output, output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
