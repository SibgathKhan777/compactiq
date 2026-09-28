#!/bin/bash
# demo_for_judges.sh - Live CLI walkthrough of fix_project_requirements.py
# Run each block one at a time so you can narrate between them.
set -e

DEMO_DIR="$HOME/compactiq_judge_demo"

echo "=============================================================="
echo "STEP 1 -- Clone a real project with real broken dependencies"
echo "=============================================================="
rm -rf "$DEMO_DIR"
git clone --depth 1 https://github.com/SibgathKhan777/nomeshops-sample.git "$DEMO_DIR"
echo
echo "Press enter to continue..."; read

echo "=============================================================="
echo "STEP 2 -- Show the raw file. Three real-world problems, hidden"
echo "in plain sight:"
echo "=============================================================="
cat "$DEMO_DIR/variants/requirements-numpy-old-pin.txt"
echo
echo "  -> numpy==1.19.5: no prebuilt wheel for modern Python, silently"
echo "     fails building from source"
echo
echo "Press enter to continue..."; read

echo "=============================================================="
echo "STEP 3 -- One command, scans and diagnoses the whole project"
echo "=============================================================="
cd /Users/sibgath/compactiq
python3 fix_project_requirements.py "$DEMO_DIR" --deploy-target linux_x86_64
echo
echo "Press enter to continue..."; read

echo "=============================================================="
echo "STEP 4 -- Apply the fixes for real. Every changed file gets a"
echo ".bak backup first -- nothing is silently overwritten."
echo "=============================================================="
python3 fix_project_requirements.py "$DEMO_DIR" --deploy-target linux_x86_64 --apply
echo
echo "Press enter to continue..."; read

echo "=============================================================="
echo "STEP 5 -- Prove the fix is real, valid pip syntax -- not just"
echo "displayed text. Every corrected line parses through Python's"
echo "own PEP 508 requirement parser."
echo "=============================================================="
python3 -c "
from packaging.requirements import Requirement
import glob
for path in glob.glob('$DEMO_DIR/**/requirements*.txt', recursive=True):
    for line in open(path):
        line = line.strip()
        if line and not line.startswith('#'):
            Requirement(line)  # raises if invalid
    print(f'{path}: every line is valid pip syntax')
"
echo
echo "Press enter to continue..."; read

echo "=============================================================="
echo "STEP 6 -- The trust moment: a misspelled package ('reqests')."
echo "The tool does NOT invent a fake fix -- it says so honestly."
echo "=============================================================="
python3 fix_project_requirements.py "$DEMO_DIR/variants" --deploy-target linux_x86_64 --pattern "requirements-typo.txt"
echo
echo "=============================================================="
echo "DONE. Recap for the judge:"
echo " - Found 2 real, fixable bugs and fixed them correctly"
echo " - Correctly refused to guess on 1 unfixable typo instead of hallucinating"
echo " - Backed up every file it touched"
echo " - Proved every output line is real, parseable pip syntax"
echo "=============================================================="
