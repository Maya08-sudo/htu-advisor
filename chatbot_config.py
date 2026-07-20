from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "knowledge_base"
PROCESSED_KB = KB_DIR / "processed" / "htu_knowledge_base.csv"
ADMISSION_RULES = BASE_DIR / "admission_rules.json"
COURSES_CSV = BASE_DIR / "courses_seed.csv"
SEED_CHUNKS_CSV = BASE_DIR / "dept_labeled_chunks_cleaned_Policy.csv"

TOP_K = 5
MIN_RETRIEVAL_SCORE = 0.08
MAX_HISTORY_MESSAGES = 10

MAJOR_ALIASES = {
    "Computer Science": ["computer science", "cs", "علم الحاسوب", "علوم الحاسوب"],
    "Cyber Security": ["cyber security", "cybersecurity", "امن سيبراني", "الأمن السيبراني", "امن المعلومات"],
    "Data Science and Artificial Intelligence": ["data science", "artificial intelligence", "ai", "علم البيانات", "الذكاء الاصطناعي", "علم البيانات والذكاء الاصطناعي"],
    "Electrical Engineering": ["electrical engineering", "ee", "هندسة كهربائية", "الهندسة الكهربائية", "كهرباء"],
    "Mechanical Engineering": ["mechanical engineering", "me", "هندسة ميكانيكية", "الهندسة الميكانيكية", "ميكانيك"],
    "Energy Engineering": ["energy engineering", "هندسة الطاقة", "طاقة"],
    "Industrial Engineering": ["industrial engineering", "ie", "هندسة صناعية", "الهندسة الصناعية"],
    "Architectural Engineering": ["architectural engineering", "architecture engineering", "architecture", "هندسة العمارة", "الهندسة المعمارية", "الهندسة المعماريه", "عمارة"],
    "Game Design and Development": ["game design", "game development", "تصميم وتطوير الالعاب", "تصميم الألعاب", "العاب"],
}

CERTIFICATE_ALIASES = {
    "tawjihi": ["tawjihi", "توجيهي", "الثانوية العامة"],
    "igcse": ["igcse", "british", "بريطاني", "الشهادة البريطانية"],
    "ib": ["international baccalaureate", "ib", "البكالوريا الدولية"],
    "american": ["american", "sat", "act", "ap", "امريكي", "أمريكي"],
    "btec": ["btec", "بي تك", "بيتك"],
}
