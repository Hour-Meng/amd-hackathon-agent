"""Generate labels.csv with 200 labeled query pairs for cache calibration."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = ROOT / "calibration_data.json"
OUT = ROOT / "labels.csv"

PARAPHRASES = {
    "What is 2 + 2?": ["what is two plus two", "2+2 equals?", "calculate 2 plus 2"],
    "Capital of France": ["what is the capital of france", "france capital city", "name france's capital"],
    "Who wrote Hamlet": ["hamlet author", "who is the author of hamlet", "who penned hamlet"],
    "Hello": ["hi there", "hey", "good morning"],
    "What is machine learning?": ["define machine learning", "explain ml briefly", "what does machine learning mean"],
    "Write a binary search in Python": ["implement binary search python", "python binary search function", "code binary search in py"],
    "Explain quantum entanglement": ["what is quantum entanglement", "describe entanglement in quantum physics", "quantum entanglement explained"],
    "spell apple backward": ["reverse the word apple", "apple spelled backwards", "backward spelling of apple"],
    "What is the population of Japan": ["japan population count", "how many people live in japan", "population figure for japan"],
    "Compare REST and GraphQL": ["rest vs graphql differences", "graphql compared to rest api", "contrast rest and graphql"],
}

DIFFERENT_PAIRS = [
    ("What is 2 + 2?", "Capital of France"),
    ("Hello", "Write a binary search in Python"),
    ("Who wrote Hamlet", "What is machine learning?"),
    ("spell apple backward", "Explain quantum entanglement"),
    ("Capital of France", "Derive Euler's identity"),
    ("What is the population of Japan", "Hello"),
    ("Compare REST and GraphQL", "What is 2 + 2?"),
    ("Extract email addresses from this text: john@example.com", "Who wrote Hamlet"),
    ("What is machine learning?", "Capital of France"),
    ("Write a Python class for a binary tree", "Hello"),
]


def main() -> None:
    random.seed(42)
    rows: list[tuple[str, str, int]] = []

    base_queries: list[str] = []
    if CALIBRATION.exists():
        base_queries = [item["query"] for item in json.loads(CALIBRATION.read_text())]

    for base, variants in PARAPHRASES.items():
        for variant in variants:
            rows.append((base, variant, 1))

    for qa, qb in DIFFERENT_PAIRS:
        rows.append((qa, qb, 0))
        rows.append((qb, qa, 0))

    # Fill to 200 with synthetic paraphrases / mismatches
    fillers_same = [
        ("tell me the capital of france", "what is france's capital", 1),
        ("how are you today", "how are you doing", 1),
        ("thanks a lot", "thank you", 1),
        ("where is cambodia", "location of cambodia", 1),
        ("what is 9 plus 10", "9+10?", 1),
        ("reverse banana", "spell banana backwards", 1),
        ("summarize this article", "give me a short summary", 1),
        ("list prime numbers under 20", "primes below twenty", 1),
    ]
    fillers_diff = [
        ("what is the weather", "implement quicksort in rust", 0),
        ("who is lebron james", "count letters in strawberry", 0),
        ("translate hello to khmer", "what is photosynthesis", 0),
        ("build a rest api", "what is the capital of cambodia", 0),
        ("fix this bug in my code", "tell me a joke", 0),
        ("describe the eiffel tower", "write regex for emails", 0),
    ]

    for item in fillers_same + fillers_diff:
        rows.append(item)

    while len(rows) < 200 and base_queries:
        a = random.choice(base_queries)
        b = random.choice(base_queries)
        label = 1 if a == b else 0
        if label == 0 and random.random() < 0.3:
            label = 1
            b = a.lower()
        rows.append((a, b, label))

    rows = rows[:200]

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query_a", "query_b", "identical_intent"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} labeled pairs to {OUT}")


if __name__ == "__main__":
    main()
