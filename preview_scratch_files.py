"""Preview all ambiguously-named files (Untitled*, untitled*, etc.) in one pass.
Prints a short excerpt from each so they can all be renamed in a single batch
instead of one round-trip per file.
Run from the repo root: python preview_scratch_files.py
"""
import os
import json
import glob

PATTERNS = ["Untitled*", "untitled*", "New*.py", "New*.ipynb", "test*.py", "scratch*"]
SEARCH_DIRS = ["src", "."]

def preview_notebook(path, max_lines=15):
    try:
        with open(path, "r", encoding="utf-8") as f:
            nb = json.load(f)
        lines_out = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code" and cell.get("source"):
                src = "".join(cell["source"])
                if src.strip():
                    lines_out.extend(src.split("\n")[:max_lines])
                    break
        if not lines_out:
            return "(empty or no code cells)"
        return "\n".join(lines_out[:max_lines])
    except Exception as e:
        return "(could not parse: {})".format(e)

def preview_py(path, max_lines=15):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        content = "".join(lines[:max_lines])
        return content if content.strip() else "(empty file)"
    except Exception as e:
        return "(could not read: {})".format(e)

if __name__ == "__main__":
    found = set()
    for d in SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for pat in PATTERNS:
            for f in glob.glob(os.path.join(d, pat)):
                if os.path.isfile(f):
                    found.add(f)

    if not found:
        print("No ambiguously-named files found. Nothing to preview.")
    else:
        print("=" * 70)
        print("SCRATCH FILE PREVIEW — {} files found".format(len(found)))
        print("Copy this ENTIRE output and paste it back for a rename plan.")
        print("=" * 70)
        for f in sorted(found):
            size = os.path.getsize(f)
            print("\n{}".format("-" * 70))
            print("FILE: {}  ({} bytes)".format(f, size))
            print("-" * 70)
            if f.endswith(".ipynb"):
                print(preview_notebook(f))
            else:
                print(preview_py(f))
        print("\n" + "=" * 70)
        print("END OF PREVIEW — paste everything above this line")
        print("=" * 70)
