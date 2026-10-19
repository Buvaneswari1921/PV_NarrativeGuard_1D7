from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1]
engine = root / "meddra_rag" / "terminology_engine.py"
classifier = root / "src" / "chatbot" / "query_classifier.py"

for path in (engine, classifier):
    if not path.exists():
        raise FileNotFoundError(path)
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")

text = engine.read_text(encoding="utf-8")

old = '''    "itchy skin": "pruritus",
    "itching": "pruritus",
    "itch": "pruritus",'''

new = '''    "itchy skin": "pruritus",
    "skin itching": "pruritus",
    "skin itch": "pruritus",
    "itching": "pruritus",
    "itch": "pruritus",'''

if old not in text:
    raise RuntimeError("Pruritus mapping block not found; no engine change made.")
text = text.replace(old, new, 1)

old_final = '''    return {
        "status":
            "OUT_OF_CONTEXT",

        "message":
            "This is out of my context.",

        "result":
            None,

        "candidates":
            evaluated[:5]
    }
'''

new_final = '''    return {
        "status":
            "NOT_FOUND",

        "message":
            "No sufficiently supported terminology was found in the current MedDRA knowledge dataset.",

        "result":
            None,

        "candidates":
            evaluated[:5]
    }
'''

if old_final not in text:
    raise RuntimeError("Final terminology block not found; no engine change made.")
text = text.replace(old_final, new_final, 1)
engine.write_text(text, encoding="utf-8")
print("Updated terminology_engine.py")

ctext = classifier.read_text(encoding="utf-8")

old_patterns = '''CLINICAL_EVENT_PATTERNS = (
    r"\\bpatient\\s+(?:has|had|developed|experienced|reports?|reported|"
    r"complains?\\s+of|complained\\s+of|presented\\s+with)\\b",

    r"\\bsubject\\s+(?:has|had|developed|experienced|reports?|reported|"
    r"complains?\\s+of|complained\\s+of|presented\\s+with)\\b",

    r"\\bpatient's\\s+\\w+",
    r"\\bsubject's\\s+\\w+",
)
'''

new_patterns = '''CLINICAL_EVENT_PATTERNS = (
    r"\\bpatient\\s+(?:has|had|developed|experienced|reports?|reported|"
    r"complains?\\s+of|complained\\s+of|presented\\s+with|"
    r"feels?|felt|is\\s+experiencing|was\\s+experiencing)\\b",

    r"\\bsubject\\s+(?:has|had|developed|experienced|reports?|reported|"
    r"complains?\\s+of|complained\\s+of|presented\\s+with|"
    r"feels?|felt|is\\s+experiencing|was\\s+experiencing)\\b",

    r"\\bpatient's\\s+\\w+",
    r"\\bsubject's\\s+\\w+",
)
'''

if old_patterns not in ctext:
    raise RuntimeError("Clinical-event pattern block not found; no classifier change made.")
ctext = ctext.replace(old_patterns, new_patterns, 1)
classifier.write_text(ctext, encoding="utf-8")
print("Updated query_classifier.py")
print("Patch completed successfully.")
