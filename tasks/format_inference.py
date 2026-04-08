"""Task 1 (Easy): Format Inference.

The agent receives raw, unstructured data and must infer the correct output
format — with zero instructions. It must figure out what transformation to
apply purely from the shape of the input.
"""

import json
import random
from typing import Any, Dict, List, Tuple


TASK_BANK: List[Dict[str, Any]] = [
    # --- CSV to JSON (first,last,age) ---
    {"input": "john,doe,25", "expected": {"first": "john", "last": "doe", "age": 25}, "category": "csv_to_json"},
    {"input": "alice,smith,30", "expected": {"first": "alice", "last": "smith", "age": 30}, "category": "csv_to_json"},
    {"input": "bob,jones,42", "expected": {"first": "bob", "last": "jones", "age": 42}, "category": "csv_to_json"},
    {"input": "maria,garcia,28", "expected": {"first": "maria", "last": "garcia", "age": 28}, "category": "csv_to_json"},
    {"input": "chen,wei,35", "expected": {"first": "chen", "last": "wei", "age": 35}, "category": "csv_to_json"},

    # --- Fix swapped date (YYYY-DD-MM → YYYY-MM-DD) ---
    {"input": "2024-13-01", "expected": "2024-01-13", "category": "fix_date"},
    {"input": "2023-25-12", "expected": "2023-12-25", "category": "fix_date"},
    {"input": "2025-31-07", "expected": "2025-07-31", "category": "fix_date"},
    {"input": "2024-15-06", "expected": "2024-06-15", "category": "fix_date"},
    {"input": "2026-28-02", "expected": "2026-02-28", "category": "fix_date"},

    # --- Title case ---
    {"input": "hello world from zero prompt", "expected": "Hello World From Zero Prompt", "category": "title_case"},
    {"input": "the quick brown fox", "expected": "The Quick Brown Fox", "category": "title_case"},
    {"input": "machine learning is amazing", "expected": "Machine Learning Is Amazing", "category": "title_case"},
    {"input": "deep reinforcement learning agent", "expected": "Deep Reinforcement Learning Agent", "category": "title_case"},
    {"input": "natural language processing system", "expected": "Natural Language Processing System", "category": "title_case"},

    # --- Key-value pairs to JSON ---
    {"input": "name:John age:25 city:NYC", "expected": {"name": "John", "age": "25", "city": "NYC"}, "category": "kv_to_json"},
    {"input": "color:red size:large qty:5", "expected": {"color": "red", "size": "large", "qty": "5"}, "category": "kv_to_json"},
    {"input": "host:localhost port:8080 proto:https", "expected": {"host": "localhost", "port": "8080", "proto": "https"}, "category": "kv_to_json"},
    {"input": "lang:python version:3.11 os:linux", "expected": {"lang": "python", "version": "3.11", "os": "linux"}, "category": "kv_to_json"},

    # --- Extract emails ---
    {"input": "Contact us at support@example.com or sales@example.com for help.", "expected": ["support@example.com", "sales@example.com"], "category": "extract_emails"},
    {"input": "Send to admin@test.org and info@test.org please.", "expected": ["admin@test.org", "info@test.org"], "category": "extract_emails"},
    {"input": "Reach out to hr@company.io, legal@company.io, or ceo@company.io.", "expected": ["hr@company.io", "legal@company.io", "ceo@company.io"], "category": "extract_emails"},

    # --- Normalize whitespace + trim ---
    {"input": "  hello   world   foo  ", "expected": "hello world foo", "category": "normalize_whitespace"},
    {"input": "   too    many    spaces   here  ", "expected": "too many spaces here", "category": "normalize_whitespace"},
    {"input": "  clean   this   text   up  ", "expected": "clean this text up", "category": "normalize_whitespace"},

    # --- TSV to JSON (name\tvalue\tunit) ---
    {"input": "temperature\t98.6\tfahrenheit", "expected": {"name": "temperature", "value": "98.6", "unit": "fahrenheit"}, "category": "tsv_to_json"},
    {"input": "distance\t42.0\tkilometers", "expected": {"name": "distance", "value": "42.0", "unit": "kilometers"}, "category": "tsv_to_json"},
    {"input": "weight\t150\tpounds", "expected": {"name": "weight", "value": "150", "unit": "pounds"}, "category": "tsv_to_json"},

    # --- Remove duplicates from comma-separated list ---
    {"input": "apple,banana,apple,cherry,banana,date", "expected": "apple,banana,cherry,date", "category": "dedup_list"},
    {"input": "red,blue,red,green,blue,yellow", "expected": "red,blue,green,yellow", "category": "dedup_list"},
    {"input": "x,y,z,x,y,w,z", "expected": "x,y,z,w", "category": "dedup_list"},
]


def _generate_procedural_task(rng: random.Random) -> Dict[str, Any]:
    """Generate a random task procedurally for more variety."""
    category = rng.choice([
        "csv_to_json", "fix_date", "kv_to_json", "normalize_whitespace", "dedup_list",
    ])

    if category == "csv_to_json":
        firsts = ["emma", "liam", "sofia", "noah", "mia", "james", "olivia", "ethan"]
        lasts = ["lee", "kim", "patel", "brown", "silva", "wang", "taylor", "martin"]
        first = rng.choice(firsts)
        last = rng.choice(lasts)
        age = rng.randint(18, 65)
        return {
            "input": f"{first},{last},{age}",
            "expected": {"first": first, "last": last, "age": age},
            "category": "csv_to_json",
        }

    if category == "fix_date":
        year = rng.randint(2020, 2026)
        month = rng.randint(1, 12)
        day = rng.randint(13, 28)  # day > 12 so swapped is clearly wrong
        return {
            "input": f"{year}-{day:02d}-{month:02d}",
            "expected": f"{year}-{month:02d}-{day:02d}",
            "category": "fix_date",
        }

    if category == "kv_to_json":
        keys = rng.sample(["name", "type", "mode", "env", "level", "role", "status", "region"], 3)
        vals = rng.sample(["alpha", "beta", "prod", "dev", "admin", "user", "active", "east"], 3)
        pairs = [f"{k}:{v}" for k, v in zip(keys, vals)]
        expected = {k: v for k, v in zip(keys, vals)}
        return {"input": " ".join(pairs), "expected": expected, "category": "kv_to_json"}

    if category == "normalize_whitespace":
        words = rng.sample(["fix", "this", "messy", "text", "now", "please", "data", "clean"], rng.randint(3, 5))
        messy = "  " + "   ".join(words) + "  "
        return {"input": messy, "expected": " ".join(words), "category": "normalize_whitespace"}

    # dedup_list
    items = rng.sample(["a", "b", "c", "d", "e", "f", "g", "h"], 4)
    duped = items + rng.sample(items, 2)
    rng.shuffle(duped)
    seen = []
    for x in duped:
        if x not in seen:
            seen.append(x)
    return {"input": ",".join(duped), "expected": ",".join(seen), "category": "dedup_list"}


def pick_task(seed: int = None) -> Tuple[Any, Any, str]:
    """Pick a task — 60% from bank, 40% procedurally generated."""
    rng = random.Random(seed)
    if rng.random() < 0.6:
        task = rng.choice(TASK_BANK)
    else:
        task = _generate_procedural_task(rng)
    return task["input"], task["expected"], task["category"]


def grade(response: str, expected: Any) -> Tuple[float, str]:
    """Grade the agent's response against the expected output.

    Returns (score, feedback) where score is 0.0–1.0.
    """
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        parsed = response.strip()

    # Exact match (case-sensitive for strings)
    if _exact_match(parsed, expected):
        return 1.0, "Correct! Exact match."

    # Case-insensitive match (slightly lower score)
    if _normalize(parsed) == _normalize(expected):
        return 0.9, "Almost correct — minor formatting difference."

    # Partial credit
    score = _partial_score(parsed, expected)
    if score > 0:
        return score, f"Partially correct (score: {score:.2f}). Output structure or values differ from expected."
    return 0.0, "Incorrect. The output does not match the expected format or values."


def _exact_match(parsed: Any, expected: Any) -> bool:
    """Check for exact match — case-sensitive for strings, type-flexible for values."""
    if isinstance(expected, dict) and isinstance(parsed, dict):
        if set(k.lower() for k in parsed) != set(k.lower() for k in expected):
            return False
        for ek, ev in expected.items():
            pk = next((k for k in parsed if k.lower() == ek.lower()), None)
            if pk is None:
                return False
            if not _exact_match(parsed[pk], ev):
                return False
        return True
    if isinstance(expected, list) and isinstance(parsed, list):
        if len(parsed) != len(expected):
            return False
        return all(_exact_match(p, e) for p, e in zip(parsed, expected))
    if isinstance(expected, str) and isinstance(parsed, str):
        return parsed.strip() == expected.strip()
    # Flexible number comparison: "5" == 5
    return str(parsed).strip() == str(expected).strip()


def _normalize(val: Any) -> Any:
    """Normalize a value for comparison — coerce all scalars to lowercase strings."""
    if isinstance(val, str):
        return val.strip().lower()
    if isinstance(val, dict):
        return {str(k).strip().lower(): _normalize(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_normalize(v) for v in val]
    return str(val).strip().lower()


def _partial_score(parsed: Any, expected: Any) -> float:
    """Compute partial credit between parsed and expected."""
    if isinstance(expected, dict) and isinstance(parsed, dict):
        if not expected:
            return 0.0
        matching_keys = set(k.lower() for k in parsed) & set(k.lower() for k in expected)
        key_score = len(matching_keys) / len(expected)

        value_matches = 0
        for k in matching_keys:
            pv = _normalize(parsed.get(k, parsed.get(k.lower(), "")))
            ev = _normalize(expected.get(k, expected.get(k.lower(), "")))
            if pv == ev:
                value_matches += 1
        value_score = value_matches / len(expected) if expected else 0

        return round(0.4 * key_score + 0.6 * value_score, 2)

    if isinstance(expected, list) and isinstance(parsed, list):
        if not expected:
            return 0.0
        norm_expected = set(str(e).strip().lower() for e in expected)
        norm_parsed = set(str(p).strip().lower() for p in parsed)
        overlap = norm_expected & norm_parsed
        return round(len(overlap) / len(norm_expected), 2)

    if isinstance(expected, str) and isinstance(parsed, str):
        exp = expected.strip()
        par = parsed.strip()
        if exp == par:
            return 1.0
        # Character-level similarity
        common = sum(1 for a, b in zip(exp, par) if a == b)
        return round(common / max(len(exp), len(par), 1), 2)

    return 0.0
