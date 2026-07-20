from rag import AdmissionRAG
from entity_extractor import extract_entities
from intent_classifier import classify_intent


bot = AdmissionRAG()

# Application question must return the application workflow.
question = "tell me how can I apply to the htu"
result = bot.answer(question, context={"major": "Electrical Engineering"})
assert result["intent"] == "application", result
assert "application process" in result["answer"].lower(), result["answer"]
assert "electrical engineering prepares" not in result["answer"].lower()
assert "computer science" not in result["answer"].lower()

# Speech/typing typo should resolve Computer Science and override old context.
question = "tell me molre about computer since"
result = bot.answer(question, context={"major": "Mechanical Engineering"})
assert result["intent"] == "major_info", result
assert result["entities"]["major"] == "Computer Science", result
assert "computer science" in result["answer"].lower(), result["answer"]
assert "mechanical engineering" not in result["answer"].lower()

# Correctly typed follow-up.
result = bot.answer("tell me more about computer science")
assert result["intent"] == "major_info"
assert "software development" in result["answer"].lower()

# A request that needs a major should clarify instead of dumping random rows.
result = bot.answer("tell me the courses", context={})
assert result["intent"] == "courses"
assert result["found"] is False
assert "which major" in result["answer"].lower()

# Raw retrieval metadata must never be displayed.
result = bot.answer("how can i apply")
assert "category:" not in result["answer"].lower()
assert "source page:" not in result["answer"].lower()
assert "question:" not in result["answer"].lower()

# Explicit major replaces prior context.
entities = extract_entities(
    "tell me more about computer since",
    {"major": "Electrical Engineering"},
)
assert entities.major == "Computer Science"
assert entities.major_explicit is True

assert classify_intent("how can I apply to HTU") == "application"
assert classify_intent("can I apply with an average of 75") == "eligibility"

print("PASS: real chatbot conversation checks")
