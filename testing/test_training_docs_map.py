import glob
from pathlib import Path


def test_docs_have_files_section():
    docs_dir = Path("docs/training_lifecycle")
    md_files = list(docs_dir.glob("*.md"))
    assert md_files, "No training lifecycle docs found"
    keywords = ["Files", "Files & Symbols", "Exact files", "files & symbols"]
    missing = []
    for p in md_files:
        text = p.read_text(encoding="utf-8")
        if not any(k in text for k in keywords):
            missing.append(p.name)
    assert not missing, f"The following docs are missing a 'Files' mapping section: {missing}"
