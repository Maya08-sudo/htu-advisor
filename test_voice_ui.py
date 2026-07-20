from pathlib import Path

app = Path("app.py").read_text(encoding="utf-8")

required_fragments = [
    "st.audio_input(",
    "transcribe_recording(",
    "recognize_google(",
    "Send voice question",
    "Question input method",
    "trust-strip",
    "chat-empty",
    "answer-meta",
]

for fragment in required_fragments:
    assert fragment in app, f"Missing UI fragment: {fragment}"

assert "from rag import AdmissionRAG, is_arabic" not in app
assert "from rag import AdmissionRAG" in app

print("PASS: microphone and UI checks")
