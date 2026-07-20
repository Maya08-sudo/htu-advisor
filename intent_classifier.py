from __future__ import annotations

from entity_extractor import normalize


def _contains_any(text: str, phrases: list[str]) -> bool:
    return any(normalize(phrase) in text for phrase in phrases)


def classify_intent(text: str) -> str:
    """
    Classify the user's actual task.

    Specific action phrases are evaluated before broad words such as
    "tell me", "about", and "can I apply". This prevents application
    questions from being misclassified as eligibility questions.
    """
    n = normalize(text)

    # Application workflow: "how can I apply" is about the process, not
    # whether the student is academically eligible.
    if _contains_any(
        n,
        [
            "how can i apply",
            "how do i apply",
            "how to apply",
            "start an application",
            "application steps",
            "application process",
            "apply to htu",
            "apply to the htu",
            "documents required",
            "required documents",
            "edit my application",
            "application status",
            "كيف اقدم",
            "كيف بدي اقدم",
            "كيف ابدأ التقديم",
            "خطوات التقديم",
            "طلب الالتحاق",
            "وثائق التقديم",
            "حالة الطلب",
        ],
    ):
        return "application"

    if _contains_any(
        n,
        [
            "introductory video",
            "personal video",
            "statement video",
            "فيديو تعريفي",
            "الفيديو التعريفي",
            "فيديو شخصي",
        ],
    ):
        return "video"

    if _contains_any(
        n,
        [
            "extracurricular",
            "additional certificates",
            "bonus points",
            "activities add",
            "انشطة",
            "الانشطة",
            "شهادات اضافية",
            "نقاط اضافية",
        ],
    ):
        return "bonus"

    if _contains_any(
        n,
        [
            "prerequisite",
            "prerequisites",
            "course prerequisite",
            "متطلب سابق",
            "المتطلبات السابقة",
        ],
    ):
        return "prerequisites"

    if _contains_any(
        n,
        [
            "credit hours",
            "how many credits",
            "total credits",
            "عدد الساعات",
            "كم ساعة",
            "كم عدد ساعات",
            "ساعات التخصص",
        ],
    ):
        return "study_plan"

    if _contains_any(
        n,
        [
            "compare",
            "difference between",
            " versus ",
            " vs ",
            "مقارنة",
            "الفرق بين",
        ],
    ):
        return "comparison"

    if _contains_any(
        n,
        [
            "registration fee",
            "application fee",
            "credit hour fee",
            "tuition",
            "how much does",
            "how much is",
            "cost",
            "رسوم",
            "تكلفة",
        ],
    ):
        return "fees"

    if _contains_any(
        n,
        [
            "scholarship",
            "financial aid",
            "tuition assistance",
            "discount",
            "منحة",
            "المنح",
            "مساعدة مالية",
            "خصم",
            "اعفاء",
        ],
    ):
        return "scholarships"

    if _contains_any(
        n,
        [
            "deadline",
            "application period",
            "when does application",
            "when are results",
            "موعد",
            "متى يبدأ",
            "متى ينتهي",
            "متى تظهر",
        ],
    ):
        return "deadlines"

    if _contains_any(
        n,
        [
            "contact",
            "email",
            "phone",
            "working hours",
            "where is htu",
            "htu location",
            "university address",
            "تواصل",
            "هاتف",
            "ايميل",
            "موقع الجامعة",
            "عنوان الجامعة",
            "اين تقع الجامعة",
            "ساعات الدوام",
        ],
    ):
        return "contacts"

    if _contains_any(
        n,
        [
            "interview",
            "admission test",
            "placement test",
            "competitive score",
            "مقابلة",
            "اختبار قبول",
            "اختبار تحديد مستوى",
            "علامة القبول التنافسية",
        ],
    ):
        return "interview"

    if _contains_any(
        n,
        [
            "degree types",
            "degrees awarded",
            "academic regulation",
            "graduation rules",
            "what degrees",
            "انواع الدرجات",
            "الدرجات العلمية",
            "الشهادات العلمية",
            "تعليمات اكاديمية",
        ],
    ):
        return "academic_rules"

    if _contains_any(
        n,
        [
            "about htu",
            "what is htu",
            "عن الجامعة",
            "ما هي جامعة الحسين التقنية",
        ],
    ):
        return "university_info"

    if _contains_any(
        n,
        [
            "am i eligible",
            "eligible for",
            "do i qualify",
            "can i enter",
            "my average is",
            "my grade is",
            "هل انا مؤهل",
            "هل بقدر انقبل",
            "مؤهل",
            "علامتي",
            "معدلي",
        ],
    ):
        return "eligibility"

    if _contains_any(
        n,
        [
            "admission requirement",
            "entry requirement",
            "minimum average",
            "minimum grade",
            "requirements for",
            "شروط القبول",
            "متطلبات القبول",
            "الحد الادنى",
        ],
    ):
        return "requirements"

    if _contains_any(
        n,
        [
            "course",
            "courses",
            "subjects",
            "curriculum",
            "study plan courses",
            "مواد",
            "مساقات",
            "مواد الخطة",
        ],
    ):
        return "courses"

    if _contains_any(
        n,
        [
            "recommend",
            "best major",
            "which major",
            "انسب تخصص",
            "اقترح تخصص",
            "توصية",
        ],
    ):
        return "recommendation"

    # Broad "tell me more/about/what is" is intentionally late.
    if _contains_any(
        n,
        [
            "tell me more",
            "tell me about",
            "what is",
            "what does",
            "describe",
            "more about",
            "احكيلي عن",
            "ما هو",
            "طبيعة تخصص",
            "شو تخصص",
        ],
    ):
        return "major_info"

    # Generic mentions of applying without a process phrase usually mean
    # eligibility, but only after the application phrases above were checked.
    if _contains_any(n, ["can i apply", "هل بقدر اقدم"]):
        return "eligibility"

    return "general"
