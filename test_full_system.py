from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from entity_extractor import extract_entities
from intent_classifier import classify_intent
from rag import AdmissionRAG


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_tests() -> None:
    bot = AdmissionRAG()
    context = {}

    cases = [
        (
            "What is required in the personal introductory video?",
            "video",
            ["one minute", "hobbies"],
            [],
        ),
        (
            "Do extracurricular activities add bonus points?",
            "bonus",
            ["10 bonus points"],
            ["BTEC"],
        ),
        (
            "How do I start an application to HTU?",
            "application",
            ["application process", "Create a username"],
            [],
        ),
        (
            "How much is the registration fee?",
            "fees",
            ["Registration"],
            ["Bachelor's Degree fee category"],
        ),
        (
            "What is Game Design and Development about?",
            "major_info",
            ["Game Design and Development", "Technician"],
            ["Cyber Security"],
        ),
        (
            "What courses are included in Computer Science BSc?",
            "courses",
            ["### BSc", "Data Structures"],
            ["### Technical"],
        ),
        (
            "What courses are included in the Technical Computer Science degree?",
            "courses",
            ["### Technical", "Programming"],
            ["### BSc"],
        ),
        (
            "How many credit hours are in Computer Science?",
            "study_plan",
            ["BSc", "Technical"],
            [],
        ),
        (
            "How many credit hours are in Technical Electrical Engineering?",
            "study_plan",
            ["Technical", "132"],
            ["BSc"],
        ),
        (
            "What are the prerequisites for Computer Science BSc courses?",
            "prerequisites",
            ["Course prerequisites", "Programming"],
            ["### Technical"],
        ),
        (
            "What does the admission test assess?",
            "interview",
            ["logical reasoning", "does not require prior preparation"],
            [],
        ),
        (
            "Where is HTU located?",
            "university_info",
            ["King Hussein Business Park", "Building 23"],
            [],
        ),
        (
            "What degrees can HTU award?",
            "academic_rules",
            ["bachelor", "intermediate university"],
            [],
        ),
        (
            "شو المواد الموجودة في بكالوريوس الهندسة الكهربائية؟",
            "courses",
            ["### BSc", "Electrical Engineering"],
            ["### Technical"],
        ),
        (
            "كم عدد ساعات الدرجة التقنية في هندسة الطاقة؟",
            "study_plan",
            ["Technical", "132"],
            ["BSc"],
        ),
        (
            "شو المطلوب بالفيديو التعريفي؟",
            "video",
            ["دقيقة واحدة", "هواياتك"],
            [],
        ),
    ]

    for question, expected_intent, required_text, forbidden_text in cases:
        result = bot.answer(question, context)
        context = result.get("context", context)

        require(
            result["intent"] == expected_intent,
            f"{question!r}: intent={result['intent']!r}, "
            f"expected={expected_intent!r}",
        )
        require(
            result["found"],
            f"{question!r}: expected a grounded answer",
        )

        answer = result["answer"]
        for text in required_text:
            require(
                text.lower() in answer.lower(),
                f"{question!r}: missing {text!r}\nAnswer:\n{answer}",
            )
        for text in forbidden_text:
            require(
                text.lower() not in answer.lower(),
                f"{question!r}: included unrelated {text!r}\nAnswer:\n{answer}",
            )

    # Leading zeroes must remain visible in course codes.
    result = bot.answer("What courses are included in Computer Science BSc?")
    require(
        "0040201201" in result["answer"],
        "Course numbers lost their leading zeroes.",
    )

    # Entity extraction must distinguish technical from technician.
    technical = extract_entities("Technical Computer Science")
    technician = extract_entities("Technician Game Design")
    require(technical.degree == "tech", "Technical degree extraction failed.")
    require(
        technician.degree == "technician",
        "Technician degree extraction failed.",
    )

    # Unknown questions must not crash.
    result = bot.answer("What food is served next Thursday?")
    require("answer" in result, "Unknown-question fallback failed.")

    print(f"PASS: {len(cases) + 4} chatbot checks")


if __name__ == "__main__":
    run_tests()
