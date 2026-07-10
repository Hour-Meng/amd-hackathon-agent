#!/usr/bin/env python3
"""
AMD Hackathon — Test Suite Runner

Runs a batch of test cases through the routing engine and produces a
comprehensive statistical report with per-category, per-difficulty, and
per-route breakdowns.

Usage:
    python run_test_suite.py test_cases.json                  # JSON file of cases
    python run_test_suite.py test_cases.json --output report.json
    python run_test_suite.py --generate 200                   # generate 200 sample cases
    python run_test_suite.py test_cases.json --streamlit      # launch Streamlit dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("SKIP_LOCAL", "true")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("test_suite")

# ───────────────────────────────────────────────────────────
#  Test case data model
# ───────────────────────────────────────────────────────────


@dataclass
class TestCase:
    task_id: str
    prompt: str
    category: str = ""
    difficulty: str = ""
    tags: list[str] = field(default_factory=list)
    expected_answer: str = ""
    answer_type: str = ""


@dataclass
class TaskResult:
    task_id: str
    prompt: str
    category: str
    difficulty: str
    answer: str
    route: str
    tokens: int
    latency_ms: float
    success: bool
    model_used: str
    routing_reason: str
    complexity_score: int
    error: str = ""
    router_success: bool = True
    failure_reason: str = ""


_ERROR_ANSWER_SNIPPETS = (
    "all remote models failed",
    "temporarily unavailable",
    "api key required",
    "rate limit exceeded",
    "fireworks api key",
)


def _looks_like_error_answer(answer: str) -> bool:
    stripped = answer.strip()
    if not stripped:
        return True
    if stripped.startswith(("⚠️", "❌")):
        return True
    lowered = stripped.lower()
    return any(snippet in lowered for snippet in _ERROR_ANSWER_SNIPPETS)


def evaluate_task_outcome(
    *,
    router_success: bool,
    answer: str,
    route: str,
    tokens: int,
    category: str,
    answer_type: str = "",
) -> tuple[bool, str]:
    """
  Score a task on substantive output, not merely whether routing completed.

  Returns (success, failure_reason). failure_reason is empty when success is True.
  """
    if not router_success:
        return False, "router reported failure"

    stripped = (answer or "").strip()
    if not stripped:
        return False, "empty answer"

    if _looks_like_error_answer(stripped):
        return False, "error response text"

    if category == "coding" or answer_type == "code":
        # PHANTOM local answers report tokens=0 even when content is valid.
        # Only flag missing generation for billed remote routes with empty-ish output.
        remote_routes = {"TEXT_REMOTE", "FALLBACK_REMOTE"}
        if route in remote_routes and tokens == 0 and len(stripped) < 24:
            return False, "coding response missing generated content"

    return True, ""


def task_result_to_dict(result: TaskResult) -> dict[str, object]:
    return {
        "task_id": result.task_id,
        "prompt": result.prompt,
        "category": result.category,
        "difficulty": result.difficulty,
        "answer": result.answer,
        "route": result.route,
        "tokens": result.tokens,
        "latency_ms": result.latency_ms,
        "success": result.success,
        "router_success": result.router_success,
        "failure_reason": result.failure_reason,
        "model_used": result.model_used,
        "routing_reason": result.routing_reason,
        "complexity_score": result.complexity_score,
        "error": result.error,
    }


# ───────────────────────────────────────────────────────────
#  Load / generate test cases
# ───────────────────────────────────────────────────────────


def load_test_cases(path: Path) -> list[TestCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[TestCase] = []
    for item in raw:
        cases.append(TestCase(
            task_id=str(item.get("task_id", "")),
            prompt=str(item.get("prompt", "")),
            category=str(item.get("category", "")),
            difficulty=str(item.get("difficulty", "")),
            tags=item.get("tags", []),
            expected_answer=str(item.get("expected_answer", "")),
            answer_type=str(item.get("answer_type", "")),
        ))
    return cases


def generate_sample_cases(count: int = 200) -> list[TestCase]:
    """Generate a representative set of test cases across all categories/tiers."""
    cases: list[TestCase] = []
    id_counters: dict[str, int] = {}

    def _next_id(prefix: str) -> str:
        id_counters[prefix] = id_counters.get(prefix, 0) + 1
        return f"{prefix}{id_counters[prefix]:03d}"

    # ── Math ──
    math_easy = [
        ("What is 25 + 37?", "62"),
        ("What is 9 × 8?", "72"),
        ("What is 144 ÷ 12?", "12"),
        ("How many minutes in 3 hours?", "180"),
        ("What is 1/4 of 100?", "25"),
        ("What is 100 - 47?", "53"),
        ("What is 15 × 6?", "90"),
        ("What is 250 + 175?", "425"),
        ("What is 81 ÷ 9?", "9"),
        ("What is 7²?", "49"),
    ]
    for p, a in math_easy:
        cases.append(TestCase(_next_id("M"), p, "math", "easy", ["arithmetic"], a, "numeric"))

    math_medium = [
        ("A train travels 240 km in 3 hours. What is its speed in km/h?", "80"),
        ("What is 15% of 200?", "30"),
        ("If 2x + 5 = 15, what is x?", "5"),
        ("What is the area of a circle with radius 5 cm? Use π = 3.14", "78.5"),
        ("How many seconds are there in 2.5 hours?", "9000"),
        ("If you buy 3 apples for $0.50 each, how much change from $5?", "3.50"),
        ("What is the median of [3, 7, 9, 2, 5]?", "5"),
        ("A rectangle has length 12 and width 8. What is its perimeter?", "40"),
        ("What is 2³ + 3²?", "17"),
        ("If a recipe for 4 people needs 2 eggs, how many eggs for 10 people?", "5"),
        ("What is 0.25 as a fraction in simplest form?", "1/4"),
        ("A car travels 60 miles per hour. How far in 45 minutes?", "45"),
        ("What is the square root of 144?", "12"),
        ("Simplify: 3(2x + 4) - 2x", "4x + 12"),
        ("If temperature is 30°C, what is it in Fahrenheit? F = C × 9/5 + 32", "86"),
        ("What is 5! (5 factorial)?", "120"),
        ("A pizza is cut into 8 slices. You eat 3. What fraction remains?", "5/8"),
        ("What is the volume of a cube with side 3 cm?", "27"),
        ("If you roll a fair 6-sided die, probability of rolling an even number?", "1/2"),
        ("What is 10% of 50% of 200?", "10"),
    ]
    for p, a in math_medium:
        cases.append(TestCase(_next_id("M"), p, "math", "medium", ["arithmetic", "algebra"], a, "numeric"))

    math_hard = [
        ("Probability of rolling two sixes on two dice?", "1/36"),
        ("If $1000 is invested at 5% annual compound interest for 10 years, what is the final amount? Round to nearest dollar.", "1629"),
        ("Standard deviation of [4, 8, 6, 5, 3] rounded to 2 decimal places.", "1.87"),
        ("Solve log₂(32) = ?", "5"),
        ("If sin(x) = 0.5 and x is between 0 and 90°, what is x in degrees?", "30"),
        ("How many distinct ways can you arrange the letters in the word MATH?", "24"),
        ("What is the sum of the first 50 positive integers?", "1275"),
        ("A ball is thrown upward at 20 m/s. How high does it go? Use g = 10 m/s²", "20"),
        ("What is 7 modulo 3?", "1"),
        ("If f(x) = 2x² + 3x - 2, what is f(3)?", "25"),
        ("How many degrees are each interior angle of a regular pentagon?", "108"),
        ("What is the greatest common divisor of 48 and 72?", "24"),
        ("What is the least common multiple of 6 and 10?", "30"),
        ("If the probability of rain is 0.3, what is the probability of no rain 3 days in a row assuming independence?", "0.343"),
        ("A triangle has sides 3, 4, 5. What is its area?", "6"),
    ]
    for p, a in math_hard:
        cases.append(TestCase(_next_id("M"), p, "math", "hard", ["word-problem"], a, "numeric"))

    math_expert = [
        ("What is the derivative of f(x) = 3x³ - 2x² + 5x - 1?", "9x² - 4x + 5"),
        ("Integrate ∫(2x + 3) dx from 0 to 4.", "28"),
        ("If A = [[1,2],[3,4]], what is the determinant of A?", "-2"),
        ("What is the limit of (x² - 1)/(x - 1) as x approaches 1?", "2"),
        ("A rabbit population doubles every 3 months. Starting with 2 rabbits, how many after 2 years?", "2048"),
    ]
    for p, a in math_expert:
        cases.append(TestCase(_next_id("M"), p, "math", "expert", ["calculus"], a, "numeric"))

    # ── Logic ──
    logic_easy = [
        ("All birds fly. A penguin is a bird. Does a penguin fly?", "No, penguins are birds but cannot fly"),
        ("What comes next: 2, 4, 8, 16, ?", "32"),
        ("If it is raining, the ground is wet. The ground is wet. Did it rain?", "Not necessarily"),
        ("Which number does not belong: 2, 3, 5, 7, 10?", "10"),
        ("You see a house with all four windows facing south. A bear walks by. What color is the bear?", "White"),
        ("If A > B and B > C, then A ? C", "A > C"),
        ("What day comes 3 days after Tuesday?", "Friday"),
        ("A farmer has 15 sheep. All but 8 die. How many are left?", "8"),
    ]
    for p, a in logic_easy:
        cases.append(TestCase(_next_id("L"), p, "logic", "easy", ["deduction"], a, "text"))

    logic_medium = [
        ("All A are B. All B are C. Therefore, all A are C. Is this valid?", "Yes, it is valid"),
        ("If you have two coins that add up to 30 cents and one of them is not a nickel, what coins are they?", "Quarter and nickel"),
        ("A bat and a ball cost $1.10. The bat costs $1 more than the ball. How much does the ball cost?", "5 cents"),
        ("Three light switches control a bulb in another room. You can only enter the room once. How do you determine which switch controls the bulb?", "Turn one on, leave it. Turn another on briefly then off. Enter room. On = first, warm = second, off = third"),
        ("Mary's father has 5 daughters: Nana, Nene, Nini, Nono. What is the fifth daughter's name?", "Mary"),
        ("What is the next letter: J, F, M, A, M, J, ?", "J"),
        ("If you rearrange the letters CIFAIPC, you get the name of a what?", "Pacific"),
        ("A clock shows 3:15. What is the angle between the hour and minute hand?", "7.5 degrees"),
        ("You have a 3-gallon jug and a 5-gallon jug. How do you get 4 gallons?", "Fill 5, pour to 3 (2 left in 5). Empty 3, pour 2 into 3. Fill 5, pour to 3 (1 left in 5). 4 = 5-1"),
        ("Which switch turns on which light: Two switches, two lights. One switch toggles both, the other toggles only one.", "Turn both on. If both on, first switch toggles both. Turn one off - if that light goes off, second switch controls it."),
        ("If a doctor gives you three pills and tells you to take one every half hour, how long will they last?", "1 hour"),
        ("A rooster lays an egg on top of a barn. Which way does it roll?", "Roosters don't lay eggs"),
        ("You walk into a dark room with a match, a lamp, a candle, and a fireplace. What do you light first?", "The match"),
        ("What is the missing number: 1, 1, 2, 3, 5, 8, 13, ?", "21"),
    ]
    for p, a in logic_medium:
        cases.append(TestCase(_next_id("L"), p, "logic", "medium", ["puzzle"], a, "text"))

    logic_hard = [
        ("There are 12 coins. One is counterfeit (heavier). Using a balance scale with only 3 weighings, how do you find the fake coin?", "Divide into 3 groups of 4. Weigh 4 vs 4. Heavier side has fake. Divide that 4 into 2+2. Weigh them. Heavier pair. Weigh those 2 against each other. Heavier is fake."),
        ("A man pushes his car to a hotel and tells the owner he's bankrupt. Why?", "He's playing Monopoly"),
        ("You have 10 bags of coins. Each bag has 10 coins. One bag has all counterfeit coins weighing 9g each instead of 10g. Using one weighing, find the fake bag.", "Take 1 coin from bag 1, 2 from bag 2, etc. Weigh all. Expected = 550g. Difference in grams = bag number with fakes."),
        ("Five pirates must divide 100 gold coins. The senior pirate proposes a split. If at least half vote yes, it passes. Otherwise he dies and the next senior proposes. How does the senior maximize his share?", "98 for senior, 1 for 3rd senior, 1 for 5th. They vote yes because they'd get nothing if senior dies."),
        ("You are on an island with knights (always tell truth) and knaves (always lie). You meet A and B. A says: We are both knaves. What are A and B?", "A is knave, B is knight"),
        ("How many times a day do the hands of a clock overlap?", "22"),
        ("Three gods: True (truth), False (lie), Random (random). Three yes/no questions to determine who is who. How?", "Complex - ask 'Is it true that you are True if and only if X?' etc."),
        ("You have a 7-minute and 4-minute hourglass. How do you measure exactly 9 minutes?", "Start both. When 4 runs out, flip it (8 min total elapsed? Complex)"),
        ("There are 100 prisoners and a warden. A room has a switch. Prisoners can enter one at a time. How do all 100 declare they've been in the room at least once?", "Designate one counter. If counter sees switch down, flip up and count. Others flip down only once if they haven't before."),
        ("A census taker asks a woman: How many children? She says 3. Ages? Product is 36. Sum equals the house number. Census taker needs more info. She says the oldest plays piano. Ages?", "9, 2, 2"),
    ]
    for p, a in logic_hard:
        cases.append(TestCase(_next_id("L"), p, "logic", "hard", ["brainteaser"], a, "text"))

    logic_expert = [
        ("100 blue-eyed and 100 brown-eyed islanders. If someone knows their own eye color, they leave at midnight. An outsider says 'At least one of you has blue eyes.' What happens?", "All 100 blue-eyed leave on day 100"),
        ("There are two doors. One leads to treasure, one to certain death. Two guards. One always lies, one always tells truth. You can ask one question to one guard. What do you ask?", "Ask: 'What would the other guard say is the safe door?' Then choose the opposite."),
        ("You have 25 horses and a racetrack that can race 5 at a time. What is the minimum number of races to find the top 3 fastest?", "7 races"),
    ]
    for p, a in logic_expert:
        cases.append(TestCase(_next_id("L"), p, "logic", "expert", ["advanced"], a, "text"))

    # ── Trivia ──
    trivia_easy = [
        ("What planet is known as the Red Planet?", "Mars"),
        ("Who wrote Romeo and Juliet?", "William Shakespeare"),
        ("What is the largest ocean on Earth?", "Pacific Ocean"),
        ("How many continents are there?", "Seven"),
        ("What is the freezing point of water in Celsius?", "0"),
        ("What gas do plants absorb from the atmosphere?", "Carbon dioxide"),
        ("What is the chemical symbol for gold?", "Au"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
        ("What is the fastest land animal?", "Cheetah"),
        ("How many days in a leap year?", "366"),
    ]
    for p, a in trivia_easy:
        cases.append(TestCase(_next_id("T"), p, "trivia", "easy", ["general"], a, "text"))

    trivia_medium = [
        ("In what year was the Berlin Wall built?", "1961"),
        ("What is the capital of Mongolia?", "Ulaanbaatar"),
        ("What element has the chemical symbol Fe?", "Iron"),
        ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
        ("What is the longest river in the world?", "Nile"),
        ("Which country has the largest population?", "India"),
        ("What is the boiling point of water in Fahrenheit?", "212"),
        ("Who developed the theory of general relativity?", "Albert Einstein"),
        ("What year did World War II end?", "1945"),
        ("What is the smallest country in the world by area?", "Vatican City"),
        ("What language has the most native speakers?", "Mandarin Chinese"),
        ("Which animal is known as the King of the Jungle?", "Lion"),
        ("What is the speed of light approximately in km/s?", "299,792"),
        ("Who wrote The Great Gatsby?", "F. Scott Fitzgerald"),
        ("What is the largest desert in the world (by area)?", "Antarctic Desert"),
        ("What is the chemical formula for water?", "H2O"),
    ]
    for p, a in trivia_medium:
        cases.append(TestCase(_next_id("T"), p, "trivia", "medium", ["geography", "history"], a, "text"))

    trivia_hard = [
        ("Which African country has the most UNESCO World Heritage sites?", "Ethiopia"),
        ("What year was the first Nobel Prize awarded?", "1901"),
        ("What is the only letter not appearing in any U.S. state name?", "Q"),
        ("Which element was discovered first: uranium or plutonium?", "Uranium"),
        ("What is the deepest point in the ocean called?", "Mariana Trench"),
        ("Who invented the World Wide Web?", "Tim Berners-Lee"),
        ("What is the collective noun for a group of flamingos?", "Flamboyance"),
        ("What is the only planet that rotates clockwise?", "Venus"),
        ("What year was the United Nations founded?", "1945"),
        ("Which country consumes the most chocolate per capita?", "Switzerland"),
        ("What is the longest-running Broadway show?", "The Phantom of the Opera"),
    ]
    for p, a in trivia_hard:
        cases.append(TestCase(_next_id("T"), p, "trivia", "hard", ["niche"], a, "text"))

    trivia_expert = [
        ("Who is the only person to win a Nobel Prize in two different scientific fields?", "Marie Curie (Physics 1903, Chemistry 1911)"),
        ("What is the rarest blood type in the human population?", "AB-negative"),
        ("Which country has time zones 26 hours apart? (largest time zone difference within one country)", "Russia"),
    ]
    for p, a in trivia_expert:
        cases.append(TestCase(_next_id("T"), p, "trivia", "expert", ["obscure"], a, "text"))

    # ── Facts ──
    facts_easy = [
        ("What is the capital of France?", "Paris"),
        ("Is the sun a star?", "Yes"),
        ("How many legs does a spider have?", "Eight"),
        ("What is the largest mammal on Earth?", "Blue whale"),
        ("What is the primary source of energy for Earth?", "The Sun"),
        ("Is water wet?", "Yes"),
        ("What color is the sky on a clear day?", "Blue"),
        ("Do fish have eyelids?", "No"),
    ]
    for p, a in facts_easy:
        cases.append(TestCase(_next_id("F"), p, "facts", "easy", ["science"], a, "text"))

    facts_medium = [
        ("Which is larger: the Sahara Desert or the Gobi Desert?", "Sahara"),
        ("What is photosynthesis?", "Process where plants convert sunlight, water and CO2 into glucose and oxygen"),
        ("What causes the phases of the Moon?", "The changing angle of sunlight hitting the Moon as it orbits Earth"),
        ("How many bones are in the adult human body?", "206"),
        ("Which planet has the most moons?", "Saturn"),
        ("What is the difference between a virus and a bacterium?", "Bacteria are living cells; viruses need a host to replicate"),
        ("Why do leaves change color in autumn?", "Chlorophyll breaks down, revealing other pigments"),
        ("What is the powerhouse of the cell?", "Mitochondria"),
        ("How does a magnifying glass work?", "Convex lens bends light rays to converge at a focal point"),
        ("What is the Richter scale used for?", "Measuring earthquake magnitude"),
        ("What is the difference between weather and climate?", "Weather is short-term; climate is long-term patterns"),
        ("Why does ice float on water?", "Ice is less dense because water expands when frozen"),
        ("What is the function of red blood cells?", "Carry oxygen from lungs to body tissues"),
        ("Which gas makes up most of Earth's atmosphere?", "Nitrogen"),
        ("What is a mammal?", "Warm-blooded animal with hair/fur that produces milk"),
        ("What causes a rainbow?", "Light refraction, dispersion and reflection in water droplets"),
        ("Why do we have leap years?", "To correct for the extra ~6 hours in Earth's orbit"),
        ("What is the smallest bone in the human body?", "Stapes (in the ear)"),
    ]
    for p, a in facts_medium:
        cases.append(TestCase(_next_id("F"), p, "facts", "medium", ["explanation"], a, "text"))

    facts_hard = [
        ("How does the water cycle affect hurricane formation?", "Warm water evaporates, rises, condenses releasing latent heat, lowering pressure, drawing more air"),
        ("Explain the difference between bacterial and viral infections in terms of treatment.", "Bacterial: treated with antibiotics. Viral: treated with antivirals or vaccines, antibiotics don't work"),
        ("How does a nuclear reactor generate electricity?", "Nuclear fission heats water → steam → turns turbine → generator produces electricity"),
        ("What is the greenhouse effect and how does it differ from the enhanced greenhouse effect?", "Natural: atmosphere traps some heat. Enhanced: human emissions increase trapping, causing global warming"),
        ("Why does the sky appear blue and sunsets appear red?", "Rayleigh scattering: blue light scatters more. At sunset, light travels further through atmosphere, blue scatters away, red reaches us"),
        ("How does vaccination work at the cellular level?", "Introduces antigen → B cells produce antibodies → memory cells created → faster response on real infection"),
        ("What is the difference between DNA and RNA?", "DNA: double-stranded, deoxyribose, thymine. RNA: single-stranded, ribose, uracil"),
        ("How do batteries produce electricity?", "Chemical reaction between anode and cathode via electrolyte creates electron flow through external circuit"),
        ("What causes tides on Earth?", "Gravitational pull of the Moon and Sun, combined with Earth's rotation"),
        ("Explain the Doppler effect.", "Change in frequency/wavelength of wave as source moves relative to observer"),
        ("How does a touchscreen work? (capacitive)", "Human body conducts electricity, touching screen changes capacitance at that point, detected by grid of sensors"),
    ]
    for p, a in facts_hard:
        cases.append(TestCase(_next_id("F"), p, "facts", "hard", ["science", "explanation"], a, "text"))

    facts_expert = [
        ("Explain how CRISPR gene editing works at the molecular level.", "Cas9 enzyme guided by RNA to target DNA sequence, cuts both strands, cell repairs using template"),
        ("What is the difference between fusion and fission as energy sources?", "Fission: splits heavy atoms. Fusion: combines light atoms. Fusion produces more energy, less waste, but harder to sustain"),
        ("How does quantum entanglement challenge classical physics?", "Entangled particles instantaneously affect each other regardless of distance, violating local realism"),
    ]
    for p, a in facts_expert:
        cases.append(TestCase(_next_id("F"), p, "facts", "expert", ["advanced"], a, "text"))

    # ── Coding ──
    coding_easy = [
        ("Write a Python function called greet that takes a name and returns 'Hello, {name}!'", "def greet(name): return f'Hello, {name}!'"),
        ("How do you print 'Hello World' in Python?", "print('Hello World')"),
        ("Write a Python expression to check if a number x is even.", "x % 2 == 0"),
        ("How do you create a list with numbers 1 to 5 in Python?", "[1, 2, 3, 4, 5]"),
    ]
    for p, a in coding_easy:
        cases.append(TestCase(_next_id("C"), p, "coding", "easy", ["python", "syntax"], a, "code"))

    coding_medium = [
        ("Write a Python function that returns the factorial of n.", "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"),
        ("Write a function that reverses a string.", "def reverse(s): return s[::-1]"),
        ("Write code to find the largest number in a list.", "max(lst)"),
        ("Write a function that checks if a string is a palindrome.", "def is_palindrome(s): return s == s[::-1]"),
        ("Write a function that counts vowels in a string.", "def count_vowels(s): return sum(1 for c in s.lower() if c in 'aeiou')"),
        ("Write a function that returns the nth Fibonacci number.", "def fib(n): a, b = 0, 1; [a, b := b, a+b for _ in range(n)]; return a"),
        ("Write a function that removes duplicates from a list.", "def dedup(lst): return list(dict.fromkeys(lst))"),
        ("Write a function that merges two sorted lists.", "def merge(a, b): i=j=0; r=[]; while i<len(a) and j<len(b): r.append(a[i] if a[i]<b[j] else b[j]); exec('i+=1' if a[i]<b[j] else 'j+=1'); return r + a[i:] + b[j:]"),
        ("Write a function that converts a string to title case.", "def title_case(s): return ' '.join(w.capitalize() for w in s.split())"),
        ("Write a Python class for a simple BankAccount with deposit and withdraw methods.", "class BankAccount: def __init__(self): self.balance=0\ndef deposit(self,a): self.balance+=a\ndef withdraw(self,a): self.balance-=a"),
        ("Write a function that finds all prime numbers up to n.", "def sieve(n): p=[True]*(n+1); p[0]=p[1]=False; [p[i*i::i] for i in range(2,int(n**0.5)+1) if p[i]]; return [i for i in range(n+1) if p[i]]"),
        ("Write a function that counts word frequency in a string.", "from collections import Counter; def word_freq(s): return Counter(s.lower().split())"),
    ]
    for p, a in coding_medium:
        cases.append(TestCase(_next_id("C"), p, "coding", "medium", ["function"], a, "code"))

    coding_hard = [
        ("Implement binary search in Python.", "def binary_search(arr, x): lo, hi = 0, len(arr)-1; while lo <= hi: mid = (lo+hi)//2; ..."),
        ("Write a Stack class with push, pop, and peek methods.", "class Stack: def __init__(self): self.items=[]; def push(self,i): self.items.append(i); def pop(self): return self.items.pop(); def peek(self): return self.items[-1]"),
        ("Write a function that finds the longest common prefix of a list of strings.", "def lcp(strs): if not strs: return ''; for i, c in enumerate(strs[0]): if any(i>=len(s) or s[i]!=c for s in strs): return strs[0][:i]; return strs[0]"),
        ("Write a function that detects a cycle in a linked list.", "def has_cycle(head): slow=fast=head; while fast and fast.next: slow=slow.next; fast=fast.next.next; if slow==fast: return True; return False"),
        ("Write a function that serializes a binary tree to a string and deserializes it back.", "class Codec: ..."),
        ("Write a function returning all permutations of a list.", "def permute(lst): if len(lst)<=1: return [lst]; res=[]; for i,e in enumerate(lst): [res.append([e]+p) for p in permute(lst[:i]+lst[i+1:])]; return res"),
        ("Write LRU cache implementation.", "from collections import OrderedDict; class LRU: def __init__(self, cap): self.cache=OrderedDict(); self.cap=cap; def get(self,k): if k not in self.cache: return -1; self.cache.move_to_end(k); return self.cache[k]; def put(self,k,v): self.cache[k]=v; self.cache.move_to_end(k); if len(self.cache)>self.cap: self.cache.popitem(last=False)"),
        ("Implement a function to find the kth largest element in an array.", "def kth_largest(nums, k): return sorted(nums, reverse=True)[k-1]"),
        ("Write a function to convert Roman numerals to integers.", "def roman_to_int(s): vals={'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}; total=0; prev=0; for c in reversed(s): curr=vals[c]; total+=curr if curr>=prev else -curr; prev=curr; return total"),
        ("Write a function that generates all valid parentheses combinations for n pairs.", "def gen_parens(n): res=[]; def bt(s,o,c): if len(s)==2*n: res.append(s); if o<n: bt(s+'(',o+1,c); if c<o: bt(s+')',o,c+1); bt('',0,0); return res"),
        ("Implement a function that solves the Two Sum problem (return indices).", "def two_sum(nums, t): m={}; for i,n in enumerate(nums): if t-n in m: return [m[t-n],i]; m[n]=i"),
        ("Write a function to find the longest substring without repeating characters.", "def longest_unique(s): used={}; start=0; max_len=0; for i,c in enumerate(s): if c in used and used[c]>=start: start=used[c]+1; used[c]=i; max_len=max(max_len,i-start+1); return max_len"),
        ("Write a function that checks if a binary tree is balanced.", "def is_balanced(root): def dfs(n): if not n: return (True,0); lb,lh=dfs(n.left); rb,rh=dfs(n.right); return (lb and rb and abs(lh-rh)<=1, max(lh,rh)+1); return dfs(root)[0]"),
    ]
    for p, a in coding_hard:
        cases.append(TestCase(_next_id("C"), p, "coding", "hard", ["algorithm"], a, "code"))

    coding_expert = [
        ("Implement merge sort that handles duplicate values efficiently.", "def merge_sort(arr): if len(arr)<=1: return arr; mid=len(arr)//2; left=merge_sort(arr[:mid]); right=merge_sort(arr[mid:]); ... (standard merge)"),
        ("Write a function to find the longest palindromic substring.", "def longest_pal(s): n=len(s); def expand(l,r): while l>=0 and r<n and s[l]==s[r]: l-=1; r+=1; return s[l+1:r]; return max((expand(i,i) for i in range(n)), key=len, default='')"),
        ("Implement Dijkstra's shortest path algorithm.", "def dijkstra(graph, start): import heapq; dist={node:float('inf') for node in graph}; dist[start]=0; pq=[(0,start)]; while pq: d,u=heapq.heappop(pq); if d>dist[u]: continue; for v,w in graph[u].items(): if dist[u]+w<dist[v]: dist[v]=dist[u]+w; heapq.heappush(pq,(dist[v],v)); return dist"),
        ("Write a regex matcher supporting '.' and '*' wildcards.", "def is_match(s, p): import re; return bool(re.fullmatch(p, s))"),
        ("Design a rate limiter that allows N requests per second.", "from collections import deque; class RateLimiter: def __init__(self, n): self.n=n; self.times=deque(); def allow(self): now=time.time(); while self.times and self.times[0]<now-1: self.times.popleft(); if len(self.times)>=self.n: return False; self.times.append(now); return True"),
        ("Implement topological sort for a directed acyclic graph.", "def topo_sort(graph): visited=set(); order=[]; def dfs(u): visited.add(u); for v in graph[u]: if v not in visited: dfs(v); order.append(u); [dfs(n) for n in graph if n not in visited]; return order[::-1]"),
    ]
    for p, a in coding_expert:
        cases.append(TestCase(_next_id("C"), p, "coding", "expert", ["advanced"], a, "code"))

    if count and count < len(cases):
        cases = cases[:count]

    return cases


# ───────────────────────────────────────────────────────────
#  Runner
# ───────────────────────────────────────────────────────────


def _configure_parallel_remote_limit(max_workers: int) -> None:
    """Align Fireworks concurrency with test-suite worker count."""
    os.environ["SWARM_MAX_CONCURRENT"] = str(max(1, max_workers))


def _ensure_multi_model_allowlist(remote_model: str | None = None) -> None:
    """
    Build ALLOWED_MODELS from the caller request, existing env, and validated models.

    Prefer models confirmed accessible via validated_model_list.json when present.
    Do not inject undeployed tier failovers (e.g. qwen3p7-max) that only burn 404s.
    """
    from app import REMOTE_FAILOVER_MODELS, default_allowed_models_csv, normalize_model_id
    from my_routing_agent.remote.validate_models import load_validated_models

    preferred: list[str] = []
    for part in (remote_model or "").split(","):
        mid = normalize_model_id(part.strip()) if part.strip() else ""
        if mid and mid not in preferred:
            preferred.append(mid)

    existing = [
        normalize_model_id(part)
        for part in os.environ.get("ALLOWED_MODELS", "").split(",")
        if part.strip()
    ]
    for mid in existing:
        if mid and mid not in preferred:
            preferred.append(mid)

    validated = [normalize_model_id(m) for m in load_validated_models() if normalize_model_id(m)]
    if validated:
        validated_set = set(validated)
        preferred = [m for m in preferred if m in validated_set]
        for mid in validated:
            if mid not in preferred:
                preferred.append(mid)
    else:
        for mid in REMOTE_FAILOVER_MODELS:
            if mid and mid not in preferred:
                preferred.append(mid)

    preferred = list(dict.fromkeys(m for m in preferred if m))
    os.environ["ALLOWED_MODELS"] = ",".join(preferred) if preferred else default_allowed_models_csv()


def run_suite(
    cases: list[TestCase],
    *,
    api_key: str | None = None,
    remote_model: str | None = None,
    local_model: str | None = None,
    max_workers: int = 2,
    verbose: bool = False,
    progress_callback: Any = None,
) -> list[TaskResult]:
    _ensure_multi_model_allowlist(remote_model)
    _configure_parallel_remote_limit(max_workers)

    from app import (
        DEFAULT_LOCAL_MODEL,
        DEFAULT_REMOTE_MODEL,
        DEFAULT_COMPLEXITY_THRESHOLD,
        configure_allowed_models,
        get_fireworks_api_key,
        process_user_request,
    )

    resolved_api_key = api_key or get_fireworks_api_key()
    if not resolved_api_key:
        logger.error("No Fireworks API key found. Set FIREWORKS_API_KEY env var or pass --api-key")
        return []

    # Refresh validation against Fireworks metadata when possible, then re-apply allowlist.
    try:
        from my_routing_agent.remote.validate_models import (
            load_validated_models,
            validate_remote_models,
        )
        from app import REMOTE_FAILOVER_MODELS, normalize_model_id

        current_allowed = [
            normalize_model_id(part)
            for part in os.environ.get("ALLOWED_MODELS", "").split(",")
            if part.strip()
        ]
        candidates = list(dict.fromkeys(current_allowed + list(REMOTE_FAILOVER_MODELS)))
        validate_remote_models(
            candidates,
            resolved_api_key,
            output_path=Path(__file__).resolve().parent / "validated_model_list.json",
        )
        validated = [normalize_model_id(m) for m in load_validated_models() if normalize_model_id(m)]
        if validated:
            # Keep requested order but drop inaccessible models.
            filtered = [m for m in current_allowed if m in set(validated)] or validated
            os.environ["ALLOWED_MODELS"] = ",".join(filtered)
            _ensure_multi_model_allowlist(remote_model or ",".join(filtered))
    except Exception as exc:
        logger.warning("Remote model validation skipped: %s", exc)

    resolved_remote_model = remote_model
    try:
        resolved_remote_model = configure_allowed_models(strict=False)
    except RuntimeError:
        resolved_remote_model = remote_model or DEFAULT_REMOTE_MODEL

    resolved_local_model = local_model or DEFAULT_LOCAL_MODEL
    resolved_threshold = DEFAULT_COMPLEXITY_THRESHOLD

    results: list[TaskResult] = []
    completed = 0
    total = len(cases)
    errors = 0

    def _process_one(tc: TestCase) -> TaskResult:
        try:
            t0 = time.perf_counter()
            result = process_user_request(
                tc.prompt,
                resolved_threshold,
                resolved_api_key,
                resolved_local_model,
                resolved_remote_model,
            )
            elapsed = time.perf_counter() - t0
            answer = result.answer.strip() if result.answer else ""
            router_success = result.success
            success, failure_reason = evaluate_task_outcome(
                router_success=router_success,
                answer=answer,
                route=result.route,
                tokens=result.tokens,
                category=tc.category,
                answer_type=tc.answer_type,
            )
            return TaskResult(
                task_id=tc.task_id,
                prompt=tc.prompt,
                category=tc.category,
                difficulty=tc.difficulty,
                answer=answer,
                route=result.route,
                tokens=result.tokens,
                latency_ms=result.latency_ms or round(elapsed * 1000, 1),
                success=success,
                model_used=result.model_used,
                routing_reason=result.routing_reason or "",
                complexity_score=result.complexity_score or 0,
                router_success=router_success,
                failure_reason=failure_reason,
            )
        except Exception as exc:
            nonlocal errors
            errors += 1
            return TaskResult(
                task_id=tc.task_id,
                prompt=tc.prompt,
                category=tc.category,
                difficulty=tc.difficulty,
                answer="",
                route="ERROR",
                tokens=0,
                latency_ms=0.0,
                success=False,
                model_used="",
                routing_reason=f"exception: {exc}",
                complexity_score=0,
                error=str(exc),
            )

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, tc): tc for tc in cases}
        for future in as_completed(futures):
            tc = futures[future]
            try:
                tr = future.result()
                results.append(tr)
            except Exception as exc:
                tr = TaskResult(tc.task_id, tc.prompt, tc.category, tc.difficulty,
                                "", "ERROR", 0, 0.0, False, "", f"unhandled: {exc}", 0, str(exc))
                results.append(tr)

            completed += 1
            if verbose or progress_callback:
                mark = "✓" if tr.success else "✗"
                line = f"  [{completed:3d}/{total}] {mark} {tr.task_id:6s} | {tr.route:15s} | {tr.tokens:5d} tok | {tr.latency_ms:8.1f}ms"
                if progress_callback:
                    progress_callback(line)
                elif verbose:
                    print(line)

    # Preserve input order
    order = {tc.task_id: i for i, tc in enumerate(cases)}
    results.sort(key=lambda r: order.get(r.task_id, 9999))

    if errors:
        logger.warning(f"{errors} task(s) raised exceptions")

    # One retry for transient remote failures (semaphore/API contention under load).
    retry_ids = {
        r.task_id
        for r in results
        if not r.success and r.route in {"TEXT_REMOTE", "FALLBACK_REMOTE"} and not r.router_success
    }
    if retry_ids:
        _configure_parallel_remote_limit(1)
        case_by_id = {tc.task_id: tc for tc in cases}
        for task_id in sorted(retry_ids):
            tc = case_by_id.get(task_id)
            if tc is None:
                continue
            time.sleep(0.75)
            retry_result = _process_one(tc)
            for idx, existing in enumerate(results):
                if existing.task_id == task_id:
                    results[idx] = retry_result
                    break

    return results


# ───────────────────────────────────────────────────────────
#  Report generation
# ───────────────────────────────────────────────────────────


@dataclass
class SuiteReport:
    summary: dict[str, Any] = field(default_factory=dict)
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_route: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    cost_estimate: dict[str, Any] = field(default_factory=dict)
    latency_summary: dict[str, Any] = field(default_factory=dict)
    task_results: list[dict[str, Any]] = field(default_factory=list)


def _agg_stats(results: list[TaskResult]) -> dict[str, Any]:
    n = len(results)
    passed = sum(1 for r in results if r.success)
    total_tok = sum(r.tokens for r in results)
    total_lat = sum(r.latency_ms for r in results)
    empty_answers = sum(1 for r in results if not (r.answer or "").strip())
    false_positives = sum(
        1 for r in results if r.router_success and not r.success
    )
    return {
        "total": n,
        "passed": passed,
        "failed": n - passed,
        "success_rate": round(passed / n * 100, 1) if n else 0.0,
        "avg_tokens": round(total_tok / n, 1) if n else 0.0,
        "avg_latency_ms": round(total_lat / n, 1) if n else 0.0,
        "total_tokens": total_tok,
        "total_latency_ms": round(total_lat, 1),
        "empty_answers": empty_answers,
        "false_positives": false_positives,
    }


def _route_dist(results: list[TaskResult]) -> dict[str, int]:
    d: dict[str, int] = {}
    for r in results:
        d[r.route] = d.get(r.route, 0) + 1
    return dict(sorted(d.items()))


def generate_report(results: list[TaskResult]) -> SuiteReport:
    report = SuiteReport()

    # Overall
    report.summary = _agg_stats(results)
    report.summary["route_distribution"] = _route_dist(results)

    # By category
    cats: dict[str, list[TaskResult]] = {}
    for r in results:
        cats.setdefault(r.category or "unknown", []).append(r)
    for cat, items in sorted(cats.items()):
        report.by_category[cat] = _agg_stats(items)
        report.by_category[cat]["route_distribution"] = _route_dist(items)

    # By difficulty
    diffs: dict[str, list[TaskResult]] = {}
    for r in results:
        diffs.setdefault(r.difficulty or "unknown", []).append(r)
    for diff, items in sorted(diffs.items()):
        report.by_difficulty[diff] = _agg_stats(items)
        report.by_difficulty[diff]["route_distribution"] = _route_dist(items)

    # By route
    routes: dict[str, list[TaskResult]] = {}
    for r in results:
        routes.setdefault(r.route, []).append(r)
    for route, items in sorted(routes.items()):
        stats = _agg_stats(items)
        stats.pop("route_distribution", None)
        report.by_route[route] = stats

    # Failures
    report.failures = [
        {
            "task_id": r.task_id,
            "category": r.category,
            "difficulty": r.difficulty,
            "prompt": r.prompt[:120],
            "route": r.route,
            "model_used": r.model_used,
            "tokens": r.tokens,
            "answer_preview": (r.answer or "")[:240],
            "router_success": r.router_success,
            "failure_reason": r.failure_reason or r.error or r.routing_reason,
        }
        for r in results if not r.success
    ]

    report.task_results = [task_result_to_dict(r) for r in results]

    # Cost estimate (Fireworks approx: $2/1M input tokens)
    total_tok = report.summary["total_tokens"]
    report.cost_estimate = {
        "total_tokens_sent": total_tok,
        "estimated_cost_usd": round(total_tok / 1_000_000 * 2.0, 6),
        "assumptions": "$2.00 per 1M tokens input (Fireworks standard pricing)",
    }

    # Latency summary
    latencies = sorted(r.latency_ms for r in results)
    if latencies:
        fastest_r = min(results, key=lambda r: r.latency_ms)
        slowest_r = max(results, key=lambda r: r.latency_ms)
        n = len(latencies)
        report.latency_summary = {
            "fastest": {"task_id": fastest_r.task_id, "latency_ms": fastest_r.latency_ms},
            "slowest": {"task_id": slowest_r.task_id, "latency_ms": slowest_r.latency_ms},
            "median_ms": latencies[n // 2] if n else 0.0,
            "p95_ms": latencies[int(n * 0.95)] if n else 0.0,
            "p99_ms": latencies[int(n * 0.99)] if n else 0.0,
            "mean_ms": round(sum(latencies) / n, 1) if n else 0.0,
        }
    else:
        report.latency_summary = {}

    return report


def _pct_box(total: int, count: int) -> str:
    if not total:
        return "  0.0%"
    return f"{count/total*100:5.1f}%"


def print_terminal_report(report: SuiteReport, results: list[TaskResult]) -> None:
    s = report.summary
    W = 58

    def _center(text: str, w: int = W) -> str:
        pad = w - len(text)
        if pad <= 0:
            return text
        left = pad // 2
        right = pad - left
        return " " * left + text + " " * right

    def _bar(label: str, pct: float, width: int = 30) -> str:
        filled = int(pct / 100 * width)
        bar = "█" * filled + "░" * (width - filled)
        return f"  {label:12s} {bar} {pct:5.1f}%"

    print()
    print("╔" + "═" * W + "╗")
    print("║" + _center("TEST SUITE REPORT") + "║")
    print("╠" + "═" * W + "╣")
    print(f"║  Success Rate:    {s['success_rate']:6.1f}%  ({s['passed']}/{s['total']})"
          f"{'':>15s} ║")
    if s.get("false_positives"):
        fp = s["false_positives"]
        print(f"║  False Positives: {fp:6d}  (router ok, bad output){'':>8s} ║")
    if s.get("empty_answers"):
        ea = s["empty_answers"]
        print(f"║  Empty Answers:   {ea:6d}{'':>24s} ║")
    print(f"║  Total Tokens:    {s['total_tokens']:>6,}{'':>24s} ║")
    print(f"║  Avg Latency:     {s['avg_latency_ms']:>8.0f} ms{'':>21s} ║")
    total_time = s['total_latency_ms'] / 1000
    print(f"║  Total Time:      {total_time:>6.1f} s{'':>21s} ║")

    print("╠" + "═" * W + "╣")
    print("║" + _center("By Category") + "║")
    print("╠" + "═" * W + "╣")
    print("║  Category      │  Pass%  │    Tokens │  Latency ms ║")
    print("║" + "─" * (W - 2) + "║")
    for cat, cs in sorted(report.by_category.items()):
        rate = f"{cs['success_rate']:.0f}%"
        toks = f"{cs['avg_tokens']:.0f}"
        lat = f"{cs['avg_latency_ms']:.0f}"
        print(f"║  {cat.capitalize():12s} │ {rate:>7s} │ {toks:>9s} │ {lat:>10s} ║")

    print("╠" + "═" * W + "╣")
    print("║" + _center("By Difficulty") + "║")
    print("╠" + "═" * W + "╣")
    for diff, ds in sorted(report.by_difficulty.items()):
        rate = f"{ds['success_rate']:.0f}%"
        toks = f"{ds['avg_tokens']:.0f}"
        lat = f"{ds['avg_latency_ms']:.0f}"
        print(f"║  {diff.capitalize():12s} │ {rate:>7s} │ {toks:>9s} │ {lat:>10s} ║")

    print("╠" + "═" * W + "╣")
    print("║" + _center("Route Distribution") + "║")
    print("╠" + "═" * W + "╣")
    route_dist = s.get("route_distribution", {})
    for route, count in sorted(route_dist.items(), key=lambda x: -x[1]):
        pct = count / s["total"] * 100
        bar = "█" * int(pct / 100 * 40) + "░" * (40 - int(pct / 100 * 40))
        pct_str = f"{pct:.1f}%"
        print(f"║  {route:15s} {bar}  {count:3d} / {s['total']} ({pct_str}){'':>10s} ║")

    print("╠" + "═" * W + "╣")
    print("║" + _center("Cost Estimate") + "║")
    print("╠" + "═" * W + "╣")
    ce = report.cost_estimate
    cost_str = f"${ce['estimated_cost_usd']:.4f} (based on {ce['total_tokens_sent']:,} tokens)"
    print("║  " + cost_str + " " * (W - 4 - len(cost_str)) + "║")

    print("╠" + "═" * W + "╣")
    print("║" + _center("Latency") + "║")
    print("╠" + "═" * W + "╣")
    ls = report.latency_summary
    if ls:
        f_id = ls['fastest']['task_id']
        f_lat = ls['fastest']['latency_ms']
        s_id = ls['slowest']['task_id']
        s_lat = ls['slowest']['latency_ms']
        print(f"║  Fastest: {f_lat:>7.1f} ms  ({f_id}){'':>20s} ║")
        print(f"║  Slowest: {s_lat:>7.1f} ms  ({s_id}){'':>20s} ║")
        print(f"║  Median:  {ls['median_ms']:>7.1f} ms{'':>28s} ║")
        print(f"║  P95:     {ls['p95_ms']:>7.1f} ms{'':>28s} ║")

    if report.failures:
        print("╠" + "═" * W + "╣")
        print("║" + _center("Failures") + "║")
        print("╠" + "═" * W + "╣")
        for f in report.failures[:10]:
            prompt_short = f["prompt"][:40]
            reason = f.get("failure_reason", "")[:28]
            print(f"║  ✗ {f['task_id']:6s} [{f['route']:15s}] {prompt_short:<28s} ║")
            answer_preview = (f.get("answer_preview") or "")[:48]
            if answer_preview:
                print(f"║     {answer_preview:<54s} ║")
            if reason:
                print(f"║     reason: {reason:<46s} ║")
        remaining = len(report.failures) - 10
        if remaining > 0:
            msg = f"... and {remaining} more"
            print("║  " + msg + " " * (W - 4 - len(msg)) + "║")

    print("╚" + "═" * W + "╝")
    print()


# ───────────────────────────────────────────────────────────
#  CLI
# ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="AMD Hackathon — Test Suite Runner")
    parser.add_argument("input", nargs="?", help="Path to test_cases.json")
    parser.add_argument("--output", "-o", default="test_report.json", help="Output report path")
    parser.add_argument("--api-key", help="Fireworks API key (override env)")
    parser.add_argument("--model", help="Remote model id")
    parser.add_argument("--local-model", help="Local model id")
    parser.add_argument("--generate", type=int, default=0, help="Generate N sample cases instead of loading file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print per-task progress")
    parser.add_argument("--workers", type=int, default=2, help="Parallel workers (default 2)")
    parser.add_argument("--streamlit", action="store_true", help="Launch Streamlit dashboard")
    args = parser.parse_args()

    if args.streamlit:
        import subprocess
        script = Path(__file__).resolve()
        subprocess.run(["streamlit", "run", str(script.parent / "run_test_suite_streamlit.py")], check=True)
        return 0

    # Load or generate cases
    if args.generate:
        from scripts.generate_test_cases import generate_test_cases, validate_cases

        print(f"Generating {args.generate} sample test cases...")
        raw_cases = generate_test_cases(args.generate)
        validate_cases(raw_cases)
        out_path = Path("test_cases.json")
        out_path.write_text(
            json.dumps(raw_cases, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(raw_cases)} cases to {out_path}")
        if not args.input:
            args.input = str(out_path)

    if not args.input:
        parser.print_help()
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"File not found: {input_path}", file=sys.stderr)
        return 1

    print(f"Loading test cases from {input_path}...")
    cases = load_test_cases(input_path)
    print(f"Loaded {len(cases)} test case(s)")

    if not cases:
        print("No test cases found", file=sys.stderr)
        return 1

    results = run_suite(
        cases,
        api_key=args.api_key,
        remote_model=args.model,
        local_model=args.local_model,
        max_workers=args.workers,
        verbose=args.verbose,
    )

    if not results:
        print("No results — check API key and network", file=sys.stderr)
        return 1

    report = generate_report(results)

    # Write JSON report
    report_dict = {
        "summary": report.summary,
        "by_category": report.by_category,
        "by_difficulty": report.by_difficulty,
        "by_route": report.by_route,
        "failures": report.failures,
        "cost_estimate": report.cost_estimate,
        "latency_summary": report.latency_summary,
        "task_results": report.task_results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report_dict, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nReport written to {output_path}")

    print_terminal_report(report, results)

    return 0 if report.summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
