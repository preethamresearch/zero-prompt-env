"""Task 3 (Hard): Multi-Step Task Composition — Zero Instructions.

The agent receives a set of input/output examples that demonstrate a hidden
transformation pipeline. It must infer the transformation rule(s) and apply
them to new, unseen inputs.

This is the hardest task: the agent must do inductive reasoning over examples,
decompose a multi-step transformation, and generalize to new data.
"""

import json
import random
from typing import Any, Dict, List, Tuple


TASK_BANK: List[Dict[str, Any]] = [
    # --- Reverse words + uppercase ---
    {
        "examples": [
            {"input": "hello world", "output": "WORLD HELLO"},
            {"input": "foo bar baz", "output": "BAZ BAR FOO"},
            {"input": "one two", "output": "TWO ONE"},
        ],
        "test_input": "alpha beta gamma",
        "expected": "GAMMA BETA ALPHA",
        "category": "reverse_upper",
    },
    {
        "examples": [
            {"input": "cat dog", "output": "DOG CAT"},
            {"input": "a b c", "output": "C B A"},
            {"input": "x y", "output": "Y X"},
        ],
        "test_input": "red green blue",
        "expected": "BLUE GREEN RED",
        "category": "reverse_upper",
    },
    # --- Sort words + join with dash ---
    {
        "examples": [
            {"input": "banana apple cherry", "output": "apple-banana-cherry"},
            {"input": "dog cat ant", "output": "ant-cat-dog"},
            {"input": "zoo park mall", "output": "mall-park-zoo"},
        ],
        "test_input": "grape fig elderberry",
        "expected": "elderberry-fig-grape",
        "category": "sort_dash",
    },
    {
        "examples": [
            {"input": "c b a", "output": "a-b-c"},
            {"input": "zebra yak xray", "output": "xray-yak-zebra"},
            {"input": "mars earth venus", "output": "earth-mars-venus"},
        ],
        "test_input": "python java go",
        "expected": "go-java-python",
        "category": "sort_dash",
    },
    # --- Extract first char of each word + concatenate ---
    {
        "examples": [
            {"input": "Hello World Program", "output": "HWP"},
            {"input": "Artificial Intelligence Lab", "output": "AIL"},
            {"input": "New York City", "output": "NYC"},
        ],
        "test_input": "Machine Learning Operations",
        "expected": "MLO",
        "category": "acronym",
    },
    {
        "examples": [
            {"input": "Central Processing Unit", "output": "CPU"},
            {"input": "Graphics Processing Unit", "output": "GPU"},
            {"input": "Random Access Memory", "output": "RAM"},
        ],
        "test_input": "Read Only Memory",
        "expected": "ROM",
        "category": "acronym",
    },
    # --- Word count + repeat last word that many times ---
    {
        "examples": [
            {"input": "a b c", "output": "c c c"},
            {"input": "hello world", "output": "world world"},
            {"input": "one two three four", "output": "four four four four"},
        ],
        "test_input": "x y z w v",
        "expected": "v v v v v",
        "category": "count_repeat",
    },
    # --- Caesar cipher shift +1 ---
    {
        "examples": [
            {"input": "abc", "output": "bcd"},
            {"input": "hello", "output": "ifmmp"},
            {"input": "xyz", "output": "yza"},
        ],
        "test_input": "zero",
        "expected": "afsp",
        "category": "caesar",
    },
    {
        "examples": [
            {"input": "dog", "output": "eph"},
            {"input": "cat", "output": "dbu"},
            {"input": "zzz", "output": "aaa"},
        ],
        "test_input": "test",
        "expected": "uftu",
        "category": "caesar",
    },
    # --- Replace vowels with '*' + reverse ---
    {
        "examples": [
            {"input": "hello", "output": "*ll*h"},
            {"input": "world", "output": "dlr*w"},
            {"input": "apple", "output": "*lpp*"},
        ],
        "test_input": "orange",
        "expected": "*gn*r*",
        "category": "vowel_mask_reverse",
    },
    # --- Double each word + join with colon ---
    {
        "examples": [
            {"input": "hi", "output": "hihi"},
            {"input": "go fast", "output": "gogo:fastfast"},
            {"input": "a b c", "output": "aa:bb:cc"},
        ],
        "test_input": "red blue",
        "expected": "redred:blueblue",
        "category": "double_colon",
    },
    # --- Length of each word, joined by + ---
    {
        "examples": [
            {"input": "hi there", "output": "2+5"},
            {"input": "a bb ccc", "output": "1+2+3"},
            {"input": "hello world", "output": "5+5"},
        ],
        "test_input": "foo bars toolbox",
        "expected": "3+4+7",
        "category": "word_lengths",
    },
    # --- Interleave first and last char of each word ---
    {
        "examples": [
            {"input": "hello", "output": "ho"},
            {"input": "hello world", "output": "ho wd"},
            {"input": "cat dog", "output": "ct dg"},
        ],
        "test_input": "python java rust",
        "expected": "pn ja rt",
        "category": "first_last_char",
    },
    # --- Reverse each word individually (keep order) ---
    {
        "examples": [
            {"input": "hello world", "output": "olleh dlrow"},
            {"input": "foo bar", "output": "oof rab"},
            {"input": "abc def ghi", "output": "cba fed ihg"},
        ],
        "test_input": "python code test",
        "expected": "nohtyp edoc tset",
        "category": "reverse_words_individual",
    },
    # --- ROT13 cipher ---
    {
        "examples": [
            {"input": "abc", "output": "nop"},
            {"input": "hello", "output": "uryyb"},
            {"input": "test", "output": "grfg"},
        ],
        "test_input": "python",
        "expected": "clguba",
        "category": "rot13",
    },
    # --- Remove vowels ---
    {
        "examples": [
            {"input": "hello world", "output": "hll wrld"},
            {"input": "python code", "output": "pythn cd"},
            {"input": "artificial intelligence", "output": "rtfcl ntllgnc"},
        ],
        "test_input": "machine learning",
        "expected": "mchn lrnng",
        "category": "remove_vowels",
    },
]


def pick_task(seed: int = None) -> Tuple[Dict[str, Any], str, str]:
    """Pick a random task.

    Returns (input_data, expected_output, category) where input_data contains
    'examples' (list of input/output pairs) and 'test_input' (the unseen input).
    """
    rng = random.Random(seed)
    task = rng.choice(TASK_BANK)
    input_data = {
        "examples": task["examples"],
        "test_input": task["test_input"],
    }
    return input_data, task["expected"], task["category"]


def grade(response: str, expected: str) -> Tuple[float, str]:
    """Grade the agent's response. Returns (score, feedback)."""
    # Clean up response
    cleaned = response.strip().strip('"').strip("'").strip()

    # Try to parse JSON if wrapped
    try:
        parsed = json.loads(response)
        if isinstance(parsed, str):
            cleaned = parsed.strip()
        elif isinstance(parsed, dict) and "output" in parsed:
            cleaned = str(parsed["output"]).strip()
    except (json.JSONDecodeError, TypeError):
        pass

    expected_clean = expected.strip()

    # Exact match
    if cleaned == expected_clean:
        return 1.0, "Correct! Exact match."

    # Case-insensitive match
    if cleaned.lower() == expected_clean.lower():
        return 0.9, "Almost correct — case mismatch."

    # Whitespace-normalized match
    if " ".join(cleaned.split()).lower() == " ".join(expected_clean.split()).lower():
        return 0.85, "Almost correct — minor whitespace/case difference."

    # Partial: check if the transformation direction is right
    score = _partial_score(cleaned, expected_clean)
    if score > 0:
        return score, f"Partially correct (score: {score:.2f}). The transformation pattern was not fully applied."

    return 0.0, "Incorrect. Could not infer the correct transformation from the examples."


def _partial_score(response: str, expected: str) -> float:
    """Compute partial similarity."""
    resp = response.lower()
    exp = expected.lower()

    # Token overlap
    resp_tokens = set(resp.replace("-", " ").replace(":", " ").split())
    exp_tokens = set(exp.replace("-", " ").replace(":", " ").split())

    if not exp_tokens:
        return 0.0

    overlap = resp_tokens & exp_tokens
    token_score = len(overlap) / len(exp_tokens)

    # Character-level similarity
    common = sum(1 for a, b in zip(resp, exp) if a == b)
    char_score = common / max(len(resp), len(exp), 1)

    # Length similarity
    len_score = 1.0 - abs(len(resp) - len(exp)) / max(len(resp), len(exp), 1)

    return round(0.4 * token_score + 0.3 * char_score + 0.3 * len_score, 2)
