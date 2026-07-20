from rag import AdmissionRAG

QUESTIONS = [
    "What are the admission requirements for Electrical Engineering?",
    "How do I start an application to HTU?",
    "كم رسوم الساعة المعتمدة؟",
    "ما المواد الموجودة في تخصص الهندسة الكهربائية؟",
    "ما المنح المتاحة؟",
]

bot = AdmissionRAG()
for question in QUESTIONS:
    result = bot.answer(question, {})
    print("=" * 80)
    print("QUESTION:", question)
    print("FOUND:", result["found"])
    print("INTENT:", result["intent"])
    print("CONFIDENCE:", result["confidence"])
    print("ANSWER:", result["answer"][:1000])
