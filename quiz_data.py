# -*- coding: utf-8 -*-
"""
Quiz data and admission-checking helpers for the HTU Advisor Streamlit app.

Place this file beside app.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple
from pathlib import Path
import csv


# Must match the feature columns used when training the model.
FEATURE_COLUMNS: List[str] = [
    "assembling", "bikes", "car", "cardboard", "tinkering", "tools",
    "circuits", "bulbs", "wires", "windmills", "connecting", "climate",
    "tasks", "time", "toys", "teamwork", "work", "reading", "building",
    "likes_computers", "likes_coding", "likes_drawing", "likes_science",
    "likes_math", "likes_problem_solving", "likes_robots", "likes_circuits",
    "likes_design", "likes_games", "likes_data", "likes_ai",
    "likes_security", "likes_teamwork", "likes_business",
]


# Each option contributes values to one or more model features.
SCENARIOS: List[Dict[str, Any]] = [
    {
        "id": "broken_device",
        "en": "A device at home stops working. What would you most likely do?",
        "ar": "تعطّل جهاز في المنزل. ما الشيء الذي ستفعله غالباً؟",
        "options": [
            {
                "en": "Open it carefully and inspect the parts",
                "ar": "أفتحه بحذر وأفحص القطع",
                "features": {"tinkering": 1, "tools": 1, "assembling": 1},
            },
            {
                "en": "Check its wires, power and circuit",
                "ar": "أفحص الأسلاك والطاقة والدائرة",
                "features": {"circuits": 1, "wires": 1, "likes_circuits": 1},
            },
            {
                "en": "Search for a software or computer-based solution",
                "ar": "أبحث عن حل برمجي أو باستخدام الكمبيوتر",
                "features": {"likes_computers": 1, "likes_coding": 1, "likes_problem_solving": 1},
            },
            {
                "en": "Redesign its appearance or user experience",
                "ar": "أعيد تصميم شكله أو تجربة استخدامه",
                "features": {"likes_design": 1, "likes_drawing": 1, "cardboard": 1},
            },
        ],
    },
    {
        "id": "school_project",
        "en": "You can choose one project for a school exhibition. Which one attracts you most?",
        "ar": "يمكنك اختيار مشروع واحد لمعرض مدرسي. أي مشروع يجذبك أكثر؟",
        "options": [
            {
                "en": "Build a small robot or smart machine",
                "ar": "بناء روبوت صغير أو آلة ذكية",
                "features": {"likes_robots": 1, "building": 1, "assembling": 1},
            },
            {
                "en": "Create a game or interactive experience",
                "ar": "إنشاء لعبة أو تجربة تفاعلية",
                "features": {"likes_games": 1, "likes_design": 1, "likes_coding": 1},
            },
            {
                "en": "Analyse a dataset and discover patterns",
                "ar": "تحليل بيانات واكتشاف الأنماط",
                "features": {"likes_data": 1, "likes_math": 1, "likes_ai": 1},
            },
            {
                "en": "Design an efficient production or delivery system",
                "ar": "تصميم نظام إنتاج أو توصيل أكثر كفاءة",
                "features": {"tasks": 1, "time": 1, "likes_business": 1},
            },
        ],
    },
    {
        "id": "future_city",
        "en": "Imagine designing a future city. Which part would you work on?",
        "ar": "تخيّل أنك تصمم مدينة مستقبلية. على أي جزء ستعمل؟",
        "options": [
            {
                "en": "Renewable energy and sustainable power",
                "ar": "الطاقة المتجددة والطاقة المستدامة",
                "features": {"windmills": 1, "climate": 1, "likes_science": 1},
            },
            {
                "en": "Buildings, spaces and visual plans",
                "ar": "المباني والمساحات والمخططات البصرية",
                "features": {"likes_drawing": 1, "likes_design": 1, "building": 1},
            },
            {
                "en": "Secure digital services and networks",
                "ar": "الخدمات الرقمية والشبكات الآمنة",
                "features": {"likes_security": 1, "likes_computers": 1, "connecting": 1},
            },
            {
                "en": "Traffic flow, logistics and operations",
                "ar": "حركة المرور واللوجستيات والعمليات",
                "features": {"tasks": 1, "time": 1, "likes_business": 1},
            },
        ],
    },
    {
        "id": "free_afternoon",
        "en": "You have a free afternoon. Which activity sounds best?",
        "ar": "لديك وقت فراغ بعد الظهر. أي نشاط تفضّل؟",
        "options": [
            {
                "en": "Fix a bicycle, car part or mechanical object",
                "ar": "إصلاح دراجة أو قطعة سيارة أو شيء ميكانيكي",
                "features": {"bikes": 1, "car": 1, "tools": 1},
            },
            {
                "en": "Program a small app",
                "ar": "برمجة تطبيق صغير",
                "features": {"likes_coding": 1, "likes_computers": 1, "likes_problem_solving": 1},
            },
            {
                "en": "Draw, model or create a visual design",
                "ar": "الرسم أو النمذجة أو إنشاء تصميم بصري",
                "features": {"likes_drawing": 1, "likes_design": 1, "cardboard": 1},
            },
            {
                "en": "Read about science, AI or new technology",
                "ar": "القراءة عن العلوم أو الذكاء الاصطناعي أو التكنولوجيا",
                "features": {"reading": 1, "likes_science": 1, "likes_ai": 1},
            },
        ],
    },
    {
        "id": "team_role",
        "en": "In a team project, which role fits you best?",
        "ar": "في مشروع جماعي، أي دور يناسبك أكثر؟",
        "options": [
            {
                "en": "Build and test the physical prototype",
                "ar": "بناء النموذج العملي واختباره",
                "features": {"teamwork": 1, "building": 1, "assembling": 1},
            },
            {
                "en": "Write the software and solve technical problems",
                "ar": "كتابة البرنامج وحل المشاكل التقنية",
                "features": {"likes_teamwork": 1, "likes_coding": 1, "likes_problem_solving": 1},
            },
            {
                "en": "Plan tasks, time and resources",
                "ar": "تخطيط المهام والوقت والموارد",
                "features": {"teamwork": 1, "tasks": 1, "time": 1, "likes_business": 1},
            },
            {
                "en": "Create the concept, visuals and presentation",
                "ar": "إنشاء الفكرة والتصميم والعرض",
                "features": {"likes_teamwork": 1, "likes_design": 1, "likes_drawing": 1},
            },
        ],
    },
    {
        "id": "technology_interest",
        "en": "Which technology topic makes you most curious?",
        "ar": "أي موضوع تقني يثير فضولك أكثر؟",
        "options": [
            {
                "en": "Artificial intelligence and data",
                "ar": "الذكاء الاصطناعي والبيانات",
                "features": {"likes_ai": 1, "likes_data": 1, "likes_math": 1},
            },
            {
                "en": "Cyberattacks and digital protection",
                "ar": "الهجمات السيبرانية والحماية الرقمية",
                "features": {"likes_security": 1, "likes_computers": 1, "likes_problem_solving": 1},
            },
            {
                "en": "Electricity, electronics and control",
                "ar": "الكهرباء والإلكترونيات والتحكم",
                "features": {"bulbs": 1, "circuits": 1, "likes_circuits": 1},
            },
            {
                "en": "Machines, engines and manufacturing",
                "ar": "الآلات والمحركات والتصنيع",
                "features": {"car": 1, "tools": 1, "tinkering": 1},
            },
        ],
    },
]


def scenarios_to_feature_vector(
    answers: Mapping[str, int],
    feature_columns: Sequence[str] = FEATURE_COLUMNS,
) -> List[int]:
    """Convert selected scenario-option indexes into the model feature order."""
    values = {feature: 0 for feature in feature_columns}

    for scenario in SCENARIOS:
        scenario_id = scenario["id"]
        if scenario_id not in answers:
            continue

        try:
            selected_index = int(answers[scenario_id])
            selected = scenario["options"][selected_index]
        except (TypeError, ValueError, IndexError):
            continue

        for feature, value in selected.get("features", {}).items():
            if feature in values:
                values[feature] = max(values[feature], int(bool(value)))

    return [values[column] for column in feature_columns]


# Keys must match the labels produced by the trained model.
MAJOR_NAMES_AR: Dict[str, str] = {
    "Mechanical Engineering": "الهندسة الميكانيكية",
    "Electrical Engineering": "الهندسة الكهربائية",
    "Energy Engineering": "هندسة الطاقة",
    "CS": "علوم الحاسوب",
    "Computer Science": "علوم الحاسوب",
    "Cybersecurity": "الأمن السيبراني",
    "Cyber Security": "الأمن السيبراني",
    "Data Science and Artificial Intelligence": "علم البيانات والذكاء الاصطناعي",
    "Industrial Engineering": "الهندسة الصناعية",
    "Architecture Engineering": "الهندسة المعمارية",
    "Architectural Engineering": "الهندسة المعمارية",
    "Game Design & Development": "تصميم وتطوير الألعاب",
    "Game Design and Development": "تصميم وتطوير الألعاب",
}


# Admission family is used by app.py to populate the major selector.
MAJOR_ADMISSION_FAMILY: Dict[str, str] = {
    "Mechanical Engineering": "engineering",
    "Electrical Engineering": "engineering",
    "Energy Engineering": "engineering",
    "Industrial Engineering": "engineering",
    "Architecture Engineering": "engineering",
    "Architectural Engineering": "engineering",
    "CS": "computing",
    "Computer Science": "computing",
    "Cybersecurity": "computing",
    "Cyber Security": "computing",
    "Data Science and Artificial Intelligence": "computing",
    "Data Science & Artificial Intelligence": "computing",
    "Game Design & Development": "computing",
    "Game Design and Development": "computing",
}

MAJOR_POLICY_NAME: Dict[str, str] = {
    "CS": "Computer Science",
    "Cybersecurity": "Cyber Security",
    "Data Science and Artificial Intelligence":
        "Data Science & Artificial Intelligence",
    "Game Design and Development": "Game Design & Development",
    "Architecture Engineering": "Architectural Engineering",
}

DEGREE_POLICY_NAME: Dict[str, str] = {
    "bachelor": "BSc",
    "tech": "Technical",
    "applied": "Technician",
    "technician": "Technician",
}


def requires_physics(major: str) -> bool:
    """Return whether the selected major requires a physics threshold."""
    return MAJOR_ADMISSION_FAMILY.get(major) == "engineering"


def _grade_rank(value: Any) -> int:
    ranks = {
        "A*": 6,
        "A": 5,
        "B": 4,
        "C": 3,
        "D": 2,
        "E": 1,
        "DISTINCTION": 3,
        "D": 3,
        "MERIT": 2,
        "M": 2,
        "PASS": 1,
        "P": 1,
    }
    return ranks.get(str(value).strip().upper(), 0)


def _american_subject_pass(value: Any, family: str) -> bool:
    if not isinstance(value, Mapping):
        return False

    act = value.get("act")
    ap = value.get("ap")
    sat2 = value.get("sat2")

    # Prototype thresholds. Verify against the latest official HTU policy
    # before public deployment.
    if family == "computing_bsc":
        return bool(
            (act is not None and float(act) >= 27)
            or (ap is not None and float(ap) >= 4)
            or (sat2 is not None and float(sat2) >= 700)
        )
    if family == "engineering_bsc":
        return bool(
            (act is not None and float(act) >= 24)
            or (ap is not None and float(ap) >= 3)
            or (sat2 is not None and float(sat2) >= 600)
        )
    return bool(
        (act is not None and float(act) >= 20)
        or (ap is not None and float(ap) >= 3)
        or (sat2 is not None and float(sat2) >= 540)
    )


def check_eligibility(
    major: str,
    certificate_type: str,
    degree_tier: str,
    math_grade: Any,
    physics_grade: Any,
    overall_average: float,
) -> Tuple[bool | None, str]:
    """
    Return (eligible, detailed_reason).

    Tawjihi/numeric rules are loaded from the structured official-policy
    dataset. Physics is required for engineering and architecture programmes,
    but not for School of Computing and Informatics programmes.
    """
    certificate_type = certificate_type.lower().strip()

    # The structured dataset currently provides the verified numeric/Tawjihi
    # thresholds used by this form. Other certificate systems continue through
    # the certificate-specific logic below.
    if certificate_type == "tawjihi":
        policy_path = Path(__file__).resolve().parent / "knowledge_base" / "admission_requirements.csv"
        if not policy_path.exists():
            return None, "The admission-requirements dataset is missing."

        policy_major = MAJOR_POLICY_NAME.get(major, major)
        policy_degree = DEGREE_POLICY_NAME.get(degree_tier, degree_tier)

        with policy_path.open("r", encoding="utf-8-sig", newline="") as file:
            policy_rows = list(csv.DictReader(file))

        matching = [
            row
            for row in policy_rows
            if row.get("major", "").strip() == policy_major
            and row.get("degree", "").strip() == policy_degree
            and "Tawjihi" in row.get("certificate", "")
        ]

        if not matching:
            return None, (
                f"No verified Tawjihi rule is available for "
                f"{policy_major} — {policy_degree}."
            )

        rule = matching[0]

        try:
            average = float(overall_average)
            mathematics = float(math_grade)
        except (TypeError, ValueError):
            return None, "Overall average or mathematics grade is invalid."

        physics = None
        if rule.get("physics_min", "").strip():
            try:
                physics = float(physics_grade)
            except (TypeError, ValueError):
                return None, "Physics grade is required and must be numeric."

        minimum_average = float(rule["overall_min"])
        minimum_math = float(rule["math_min"])
        minimum_physics = (
            float(rule["physics_min"])
            if rule.get("physics_min", "").strip()
            else None
        )

        checks = [
            {
                "label": "Overall average",
                "value": average,
                "minimum": minimum_average,
                "passed": average >= minimum_average,
            },
            {
                "label": "Mathematics",
                "value": mathematics,
                "minimum": minimum_math,
                "passed": mathematics >= minimum_math,
            },
        ]

        if minimum_physics is not None:
            checks.append(
                {
                    "label": "Physics",
                    "value": physics,
                    "minimum": minimum_physics,
                    "passed": physics >= minimum_physics,
                }
            )

        failed = [check for check in checks if not check["passed"]]
        physics_note = (
            ""
            if minimum_physics is not None
            else " Physics is not a separate minimum requirement for this computing programme."
        )

        if not failed:
            details = "; ".join(
                f"{check['label']} {check['value']:.1f}% "
                f"(minimum {check['minimum']:.1f}%)"
                for check in checks
            )
            return True, (
                f"All programme-specific numeric minimums are met: {details}."
                f"{physics_note} All Tawjihi subjects must also be passed "
                f"with a final grade above 50%."
            )

        failures = "; ".join(
            f"{check['label']} is {check['value']:.1f}% "
            f"but the minimum is {check['minimum']:.1f}%"
            for check in failed
        )
        return False, (
            f"Not eligible based on the entered grades: {failures}."
            f"{physics_note} All Tawjihi subjects must also be passed "
            f"with a final grade above 50%."
        )

    family = MAJOR_ADMISSION_FAMILY.get(major)
    if family is None:
        return None, "No admission family is configured for this major."

    policy_degree = DEGREE_POLICY_NAME.get(degree_tier)
    if policy_degree is None:
        return None, "The selected degree type is not supported."

    # Certificate-specific prototype thresholds, retained until matching
    # structured official tables are added for each international certificate.
    thresholds = {
        "engineering": {
            "BSc": {"avg": 80.0, "ib": 5.0, "igcse": "B", "btec": "Merit"},
            "Technical": {"avg": 70.0, "ib": 4.0, "igcse": "C", "btec": "Pass"},
            "Technician": {"avg": 70.0, "ib": 3.0, "igcse": "D", "btec": "Pass"},
        },
        "computing": {
            "BSc": {"avg": 80.0, "ib": 5.0, "igcse": "B", "btec": "Merit"},
            "Technical": {"avg": 70.0, "ib": 4.0, "igcse": "C", "btec": "Pass"},
            "Technician": {"avg": 70.0, "ib": 3.0, "igcse": "D", "btec": "Pass"},
        },
    }

    if policy_degree not in thresholds[family]:
        return None, "No rule is configured for this degree type."

    rule = thresholds[family][policy_degree]

    try:
        average = float(overall_average)
    except (TypeError, ValueError):
        return None, "Overall average is missing or invalid."

    failures = []
    if average < rule["avg"]:
        failures.append(
            f"Overall average is {average:.1f}% but the minimum is {rule['avg']:.1f}%"
        )

    if certificate_type == "ib":
        try:
            mathematics = float(math_grade)
            if mathematics < rule["ib"]:
                failures.append(
                    f"IB Mathematics is {mathematics:g} but the minimum is {rule['ib']:g}"
                )

            if family == "engineering":
                physics = float(physics_grade)
                if physics < rule["ib"]:
                    failures.append(
                        f"IB Physics is {physics:g} but the minimum is {rule['ib']:g}"
                    )
        except (TypeError, ValueError):
            return None, "The required IB subject grade is invalid."

    elif certificate_type == "igcse":
        required_rank = _grade_rank(rule["igcse"])
        if _grade_rank(math_grade) < required_rank:
            failures.append(
                f"IGCSE/A-level Mathematics is below {rule['igcse']}"
            )
        if family == "engineering" and _grade_rank(physics_grade) < required_rank:
            failures.append(
                f"IGCSE/A-level Physics is below {rule['igcse']}"
            )

    elif certificate_type == "btec":
        required_rank = _grade_rank(rule["btec"])
        if _grade_rank(math_grade) < required_rank:
            failures.append(
                f"BTEC Mathematics is below {rule['btec']}"
            )

    elif certificate_type == "american":
        if not _american_subject_pass(math_grade, family):
            failures.append("No entered Mathematics score meets the configured minimum")
        if family == "engineering" and not _american_subject_pass(
            physics_grade,
            family,
        ):
            failures.append("No entered Physics score meets the configured minimum")
    else:
        return None, "This certificate type is not configured."

    if failures:
        return False, "Not eligible based on the entered values: " + "; ".join(failures) + "."

    return True, (
        "The entered values meet the configured minimums for this certificate "
        "and programme. Final eligibility still depends on document verification "
        "and the complete HTU admission process."
    )
