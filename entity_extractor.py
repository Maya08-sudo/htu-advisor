from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Optional

from chatbot_config import MAJOR_ALIASES, CERTIFICATE_ALIASES


_AR = re.compile(r"[\u0600-\u06FF]")


def normalize(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[ًٌٍَُِّْـ]", "", text)
    text = re.sub(r"[^\w\s%+.-]", " ", text)
    return re.sub(r"\s+", " ", text)


def is_arabic(text: str) -> bool:
    return bool(_AR.search(text or ""))


@dataclass
class Entities:
    language: str
    major: Optional[str] = None
    certificate: Optional[str] = None
    degree: Optional[str] = None
    average: Optional[float] = None
    math: Optional[float] = None
    physics: Optional[float] = None
    major_explicit: bool = False


def _fuzzy_major(text: str) -> str | None:
    """
    Resolve common speech-to-text and typing errors such as:
    'computer since' -> Computer Science.
    """
    n = normalize(text)

    common_corrections = {
        "computer since": "Computer Science",
        "computer sciences": "Computer Science",
        "computer scince": "Computer Science",
        "computer sienc": "Computer Science",
        "cyber securty": "Cyber Security",
        "cyber secuirty": "Cyber Security",
        "data since": "Data Science and Artificial Intelligence",
        "artificial inteligence": "Data Science and Artificial Intelligence",
        "electrical enginering": "Electrical Engineering",
        "mechanical enginering": "Mechanical Engineering",
        "energy enginering": "Energy Engineering",
        "industrial enginering": "Industrial Engineering",
        "architectural enginering": "Architectural Engineering",
    }
    for phrase, canonical in common_corrections.items():
        if phrase in n:
            return canonical

    words = n.split()
    candidates: list[str] = []

    # Compare 1–4 word windows to all meaningful aliases.
    for width in range(1, min(4, len(words)) + 1):
        for start in range(0, len(words) - width + 1):
            candidates.append(" ".join(words[start : start + width]))

    best_major = None
    best_score = 0.0
    for canonical, aliases in MAJOR_ALIASES.items():
        for alias in aliases:
            alias_n = normalize(alias)
            if len(alias_n) < 5 or not alias_n.isascii():
                continue
            for candidate in candidates:
                score = SequenceMatcher(None, candidate, alias_n).ratio()
                # Multi-word major names need strong similarity.
                threshold = 0.82 if " " in alias_n else 0.90
                if score >= threshold and score > best_score:
                    best_major = canonical
                    best_score = score

    return best_major


def extract_entities(text: str, context: dict | None = None) -> Entities:
    n = normalize(text)
    context = context or {}

    major = None
    major_explicit = False

    for canonical, aliases in MAJOR_ALIASES.items():
        matched = False
        for alias in aliases:
            alias_n = normalize(alias)
            if alias_n.isascii() and len(alias_n) <= 3:
                # Avoid interpreting ordinary words such as "me" in
                # "tell me more" as Mechanical Engineering. "cs" is a
                # common natural abbreviation; other two-letter engineering
                # abbreviations must be written in uppercase by the user.
                if alias_n == "cs":
                    matched = bool(
                        re.search(
                            rf"(?<!\\w){re.escape(alias_n)}(?!\\w)",
                            n,
                        )
                    )
                else:
                    matched = bool(
                        re.search(
                            rf"(?<!\\w){re.escape(alias.upper())}(?!\\w)",
                            text or "",
                        )
                    )
            else:
                matched = alias_n in n
            if matched:
                break

        if matched:
            major = canonical
            major_explicit = True
            break

    if major is None:
        fuzzy = _fuzzy_major(text)
        if fuzzy:
            major = fuzzy
            major_explicit = True

    certificate = None
    for canonical, aliases in CERTIFICATE_ALIASES.items():
        if any(normalize(alias) in n for alias in aliases):
            certificate = canonical
            break

    degree = None
    if any(
        phrase in n
        for phrase in [
            "technician",
            "درجة فنية",
            "درجة الفني",
            "فني",
        ]
    ):
        degree = "technician"
    elif any(
        phrase in n
        for phrase in [
            "technical degree",
            "technical computer science",
            "technical cyber",
            "technical data science",
            "technical electrical",
            "technical energy",
            "technical industrial",
            "technical architectural",
            "technical in",
            "degree technical",
            "درجة تقنية",
            "الدرجة التقنية",
            "تقني",
        ]
    ):
        degree = "tech"
    elif any(
        phrase in n
        for phrase in ["bsc", "bachelor", "بكالوريوس"]
    ):
        degree = "bachelor"

    numbers = [
        float(value)
        for value in re.findall(
            r"(?<!\d)(\d{1,3}(?:\.\d+)?)\s*%?",
            n,
        )
    ]
    average = math = physics = None

    labels = [
        ("average", "average"),
        ("معدل", "average"),
        ("math", "math"),
        ("رياضيات", "math"),
        ("physics", "physics"),
        ("فيزياء", "physics"),
    ]
    for label, attribute in labels:
        match = re.search(
            rf"{label}[^\d]{{0,12}}(\d{{1,3}}(?:\.\d+)?)",
            n,
        )
        if match:
            value = float(match.group(1))
            if attribute == "average":
                average = value
            elif attribute == "math":
                math = value
            else:
                physics = value

    if (
        len(numbers) == 3
        and average is None
        and math is None
        and physics is None
    ):
        average, math, physics = numbers

    return Entities(
        language="ar" if is_arabic(text) else "en",
        major=major or context.get("major"),
        certificate=certificate or context.get("certificate"),
        degree=(
            degree
            if degree is not None
            else (
                None
                if major_explicit
                else context.get("degree")
            )
        ),
        average=average,
        math=math,
        physics=physics,
        major_explicit=major_explicit,
    )
