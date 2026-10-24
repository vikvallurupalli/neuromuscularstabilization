#!/usr/bin/env python3
"""
PreToolUse hook: backs up all .py files from analysis/src
into analysis/backup/back_N before any edit to an analysis .py file.
"""
import sys
import json
import os
import shutil
import re

ANALYSIS_DIR = os.path.join(
    "c:\\", "Users", "User", "000000", "vik", "ESP32",
    "New-NeckResearch", "NeuroMuscularStabilization", "analysis"
)
SRC_DIR     = os.path.join(ANALYSIS_DIR, "src")
BACKUP_BASE = os.path.join(ANALYSIS_DIR, "backup")


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return

    fp = data.get("tool_input", {}).get("file_path", "")
    fp_norm = fp.replace("\\", "/")

    # Only act on .py files inside the analysis folder
    if "analysis" not in fp_norm or not fp_norm.endswith(".py"):
        return

    os.makedirs(BACKUP_BASE, exist_ok=True)

    # Find next incremental backup number
    existing = []
    for name in os.listdir(BACKUP_BASE):
        m = re.match(r"^back_(\d+)$", name)
        if m:
            existing.append(int(m.group(1)))
    next_n = max(existing, default=0) + 1

    dest = os.path.join(BACKUP_BASE, f"back_{next_n}")
    os.makedirs(dest, exist_ok=True)

    copied = []
    if os.path.isdir(SRC_DIR):
        for fname in os.listdir(SRC_DIR):
            if fname.endswith(".py"):
                shutil.copy2(os.path.join(SRC_DIR, fname), dest)
                copied.append(fname)

    print(
        f"[backup] {len(copied)} files → analysis/backup/back_{next_n}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
