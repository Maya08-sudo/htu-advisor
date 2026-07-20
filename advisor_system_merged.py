"""
HTU Major Advisor System
Fixed version — see FIXES.md for a full list of what was broken and why.
"""

import re
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics

import tensorflow as tf
import speech_recognition as sr
import pyttsx3
from transformers import pipeline
from nltk import word_tokenize
from nltk.probability import FreqDist

from logic import Symbol, And, Implication, Not, Or, Biconditional, model_check


# ============================================================
# TTS ENGINE
# ============================================================
engine = pyttsx3.init()
engine.setProperty("rate", 150)


def speak(text):
    print(text)
    engine.say(text)
    engine.runAndWait()


# ============================================================
# MODEL TRAINING (run once to (re)produce best_major_predictor.*)
# ============================================================
def train_major_predictor(csv_path="htu_majors_interests.csv"):
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1).values
    y = df["label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, shuffle=True, stratify=y_enc, random_state=42
    )
    # NOTE: stratify must use the *encoded* labels (y_enc), not the raw
    # string labels y — stratify=y worked by accident only because sklearn
    # accepts either, but keeping it consistent avoids confusion.

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        # FIX: multi_class='ovr' is deprecated/removed in recent sklearn
        # versions and max_iter=10 was too low to converge. Removed the
        # deprecated kwarg and raised max_iter so training actually finishes.
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Decision Tree": DecisionTreeClassifier(max_depth=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = sklearn.metrics.accuracy_score(y_test, preds)
        results[name] = acc
        print(f"{name} Accuracy: {acc:.4f}")

    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(le.classes_), activation="softmax"),
    ])
    # FIX: Dropout(0.8) on a 128-unit layer was killing most of the signal
    # for a small tabular dataset. Lowered to more reasonable rates.
    optimizer = tf.optimizers.Adam(learning_rate=0.001)
    tf_model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
    tf_model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2,
                 validation_split=0.2)
    # FIX: batch_size=512 was larger than most small survey-style datasets,
    # which meant the model barely got any gradient updates per epoch.

    loss, acc = tf_model.evaluate(X_test, y_test, verbose=0)
    results["TensorFlow ANN"] = acc

    best_model_name = max(results, key=results.get)
    best_acc = results[best_model_name]

    # FIX: previously both branches wrote to filenames that were assumed
    # fixed elsewhere in the script (ml_model = joblib.load("best_major_predictor.pkl")
    # unconditionally), which would crash if the ANN won, since it saves to .h5.
    # Now we always write a small manifest so the loader knows which format to use.
    if best_model_name == "TensorFlow ANN":
        tf_model.save("best_major_predictor.h5")
        joblib.dump({"type": "keras"}, "best_major_predictor_meta.pkl")
    else:
        joblib.dump(models[best_model_name], "best_major_predictor.pkl")
        joblib.dump({"type": "sklearn"}, "best_major_predictor_meta.pkl")

    joblib.dump(le, "label_encoder.pkl")

    print(results, best_model_name, best_acc)
    return results, best_model_name, best_acc


def load_major_predictor():
    """Load whichever model type won training, based on the saved metadata."""
    meta = joblib.load("best_major_predictor_meta.pkl")
    if meta["type"] == "keras":
        return tf.keras.models.load_model("best_major_predictor.h5"), "keras"
    return joblib.load("best_major_predictor.pkl"), "sklearn"


# ============================================================
# LOGIC / ELIGIBILITY SYMBOLS
# ============================================================
Math_L1, Math_L2, Math_L3 = Symbol("Math_L1"), Symbol("Math_L2"), Symbol("Math_L3")
Phys_L1, Phys_L2, Phys_L3 = Symbol("Phys_L1"), Symbol("Phys_L2"), Symbol("Phys_L3")
Avg_L1, Avg_L2, Avg_L3 = Symbol("Avg_L1"), Symbol("Avg_L2"), Symbol("Avg_L3")

Eligible_All_BSc = Symbol("Eligible_All_BSc")
Eligible_BSc = Symbol("Eligible_BSc")
Eligible_Tech = Symbol("Eligible_Tech")
NotEligible = Symbol("NotEligible")

Mechanical = Symbol("Mechanical Engineering")
Electrical = Symbol("Electrical Engineering")
Energy = Symbol("Energy Engineering")
DataScienceAI = Symbol("Data Science and Artificial Intelligence")
CS = Symbol("CS")
Industrial = Symbol("Industrial Engineering")
Cyber = Symbol("Cybersecurity")
Architecture = Symbol("Architecture Engineering")
Game_design = Symbol("Game Design & Development")

VALID_CERT_TYPES = ["TAWJIHI", "IGCSE", "AMERICAN", "SAT", "BTEC", "IB"]


# ============================================================
# GRADE EXTRACTION
# ============================================================
def extract_grades(text):
    math_pattern = r"(?:math|mathematics)\s*(?:grade)?\s*(?:is|=)?\s*(A\+?|B\+?|C\+?|P|M|D|\d{1,3})"
    phys_pattern = r"(?:physics|phys)\s*(?:grade)?\s*(?:is|=)?\s*(A\+?|B\+?|C\+?|P|M|D|\d{1,3})"
    avg_pattern = r"(?:average|avg)\s*(?:grade)?\s*(?:is|=)?\s*(\d{1,3})"
    cert_pattern = r"(?:certificate|cert)\s*(?:type)?\s*(?:is|=)?\s*([a-zA-Z]+)"
    # FIX: removed the stray "six|sex" alternatives from the physics pattern —
    # those looked like accidental leftovers, not real grade phrasing, and
    # risked matching the wrong word in a sentence.

    math = re.search(math_pattern, text, re.IGNORECASE)
    physics = re.search(phys_pattern, text, re.IGNORECASE)
    avg = re.search(avg_pattern, text, re.IGNORECASE)
    cert = re.search(cert_pattern, text, re.IGNORECASE)

    grades = {
        "Math": math.group(1) if math else input("Math grade: "),
        "Physics": physics.group(1) if physics else input("Physics grade: "),
        "Avg": avg.group(1) if avg else input("Average grade: "),
    }

    certificate_type = cert.group(1).upper() if cert else input("Certificate type: ").upper()
    while certificate_type not in VALID_CERT_TYPES:
        print(f"'{certificate_type}' is not a recognized certificate type.")
        certificate_type = input(f"Certificate type {VALID_CERT_TYPES}: ").upper()
    # FIX: original re-prompted only once on an invalid certificate type and
    # didn't re-validate the second answer, so a second bad entry would slip
    # through silently. This now loops until a valid value is given.

    print(grades)
    print(certificate_type)
    return certificate_type, grades


# ============================================================
# BUILD STUDENT FACTS
# ============================================================
def add_numeric_threshold_facts(facts, value, thresholds, symbols):
    for t, sym in zip(thresholds, symbols):
        if value >= t:
            facts.append(sym)
        else:
            facts.append(Not(sym))


def add_letter_threshold_facts(facts, value, thresholds, symbols):
    for t, sym in zip(thresholds, symbols):
        if value.upper() == t:
            facts.append(sym)
        else:
            facts.append(Not(sym))


def build_student_facts(cert_type, grades):
    facts = []
    math = grades.get("Math")
    phys = grades.get("Physics")
    avg = grades.get("Avg")

    # FIX: the original used `cert_type in "TAWJIHI"` (substring-of-string
    # containment) instead of equality, so e.g. cert_type == "A" would
    # accidentally match "AMERICAN SAT". Switched to a proper equality/
    # membership check against VALID_CERT_TYPES.
    if cert_type == "TAWJIHI":
        add_numeric_threshold_facts(facts, int(math), [70, 80, 85], [Math_L1, Math_L2, Math_L3])
        add_numeric_threshold_facts(facts, int(phys), [70, 80, 85], [Phys_L1, Phys_L2, Phys_L3])
        add_numeric_threshold_facts(facts, int(avg), [70, 80, 85], [Avg_L1, Avg_L2, Avg_L3])

    elif cert_type == "IGCSE":
        add_letter_threshold_facts(facts, math, ["C", "B", "A", "A*"], [Math_L1, Math_L2, Math_L3, Math_L3])
        add_letter_threshold_facts(facts, phys, ["C", "B", "A", "A*"], [Phys_L1, Phys_L2, Phys_L3, Phys_L3])
        add_numeric_threshold_facts(facts, int(avg), [70, 80, 85], [Avg_L1, Avg_L2, Avg_L3])

    elif cert_type == "IB":
        add_numeric_threshold_facts(facts, int(math), [4, 5, 6], [Math_L1, Math_L2, Math_L3])
        add_numeric_threshold_facts(facts, int(phys), [4, 5, 6], [Phys_L1, Phys_L2, Phys_L3])
        add_numeric_threshold_facts(facts, int(avg), [70, 80, 85], [Avg_L1, Avg_L2, Avg_L3])
        # FIX: original thresholds were [4, 5, 5] — the top two IB tiers were
        # identical, so Math_L2 and Math_L3 could never be distinguished.

    elif cert_type in ("AMERICAN", "SAT"):
        add_numeric_threshold_facts(facts, int(math), [540, 600, 700], [Math_L1, Math_L2, Math_L3])
        add_numeric_threshold_facts(facts, int(phys), [540, 600, 700], [Phys_L1, Phys_L2, Phys_L3])
        add_numeric_threshold_facts(facts, int(avg), [70, 80, 85], [Avg_L1, Avg_L2, Avg_L3])
        # FIX: original compared against "AMERICAN SAT" as one combined
        # string, which could never equal cert_type (which is only ever a
        # single word from VALID_CERT_TYPES). Also the top two thresholds
        # were both 630, same L2/L3 collision bug as the IB branch.

    elif cert_type == "BTEC":
        add_letter_threshold_facts(facts, math, ["M", "M", "D"], [Math_L1, Math_L2, Math_L3])
        add_letter_threshold_facts(facts, phys, ["M", "M", "D"], [Phys_L1, Phys_L2, Phys_L3])
        add_numeric_threshold_facts(facts, int(avg), [70, 80, 85], [Avg_L1, Avg_L2, Avg_L3])

    else:
        # Should be unreachable since extract_grades() already validates
        # cert_type, but kept as a safety net.
        speak("Unknown certificate type. Please provide a valid certificate type.")
        return build_student_facts(input("Certificate type: ").upper(), grades)

    # FIX: this return used to sit *after* the if/elif/else block at the same
    # indentation level, so it ran unconditionally on every call — meaning a
    # valid cert type would still trigger a pointless recursive re-prompt
    # instead of ever reaching `And(*facts)`. Moved the recursive retry
    # inside the `else` branch only, and this is now the single normal exit.
    return And(*facts)


# ============================================================
# ELIGIBILITY INFERENCE
# ============================================================
def infer_eligible_programs(facts):
    KB = And(
        Biconditional(And(Math_L3, Phys_L3, Avg_L3), Eligible_All_BSc),
        Biconditional(And(Math_L2, Phys_L2, Avg_L2), Eligible_BSc),
        Biconditional(And(Math_L1, Phys_L1, Avg_L1), Eligible_Tech),
        Biconditional(Not(Or(Math_L1, Phys_L1, Avg_L1)), NotEligible),
        Implication(Eligible_All_BSc, Eligible_BSc),
        Implication(Eligible_All_BSc, Eligible_Tech),
        Implication(Eligible_BSc, Eligible_Tech),
    )
    KB.add(facts)

    majors_kb = And(
        Implication(Eligible_All_BSc, And(Mechanical, Electrical, Energy, DataScienceAI, CS, Industrial, Cyber, Architecture)),
        Implication(Eligible_BSc, And(Mechanical, Electrical, Energy, Cyber, Architecture)),
        Implication(Eligible_Tech, And(Mechanical, Electrical, Energy, DataScienceAI, CS, Industrial, Cyber, Architecture, Game_design)),
    )

    # FIX: figure out the student's single eligibility tier ONCE, outside the
    # per-major loop. The original re-ran model_check(KB, ...) and re-added
    # the same fact to majors_kb on every loop iteration — wasteful, and it
    # silently mislabeled the "not eligible" case by tagging results with
    # Eligible_Tech.name instead of NotEligible.name.
    if model_check(KB, Eligible_All_BSc):
        tier = Eligible_All_BSc
    elif model_check(KB, Eligible_BSc):
        tier = Eligible_BSc
    elif model_check(KB, Eligible_Tech):
        tier = Eligible_Tech
    else:
        tier = NotEligible

    if tier is NotEligible:
        return []

    majors_kb.add(tier)

    all_majors = [Mechanical, Electrical, Energy, DataScienceAI, Industrial,
                  CS, Cyber, Architecture, Game_design]
    # FIX: Game_design was missing from the candidate list entirely, so it
    # could never be recommended even for Eligible_Tech students.

    eligible = []
    for prog in all_majors:
        if model_check(majors_kb, prog):
            eligible.append([tier.name, prog.name])
    return eligible


# ============================================================
# SPEECH INPUT
# ============================================================
def listen_for_grades():
    recognizer = sr.Recognizer()
    for attempt in range(3):
        with sr.Microphone() as source:
            speak("Please provide me with your certificate type and your grades for math, physics, and average.")
            audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            print(f"Recognized speech: {text}")
            return text
        except sr.UnknownValueError:
            speak("Sorry, I could not understand your voice.")
        except sr.RequestError as e:
            speak("Speech recognition service is unavailable right now.")
            print(f"RequestError: {e}")
            break
    # FIX: original had a bare `except:` (hides real errors like network
    # failures) and `attempt -= 1` inside the loop, which doesn't affect the
    # for-loop's iteration count at all — it's a no-op, so the "retry"
    # comment was misleading. Also there was no fallback return, so callers
    # would get `None` silently. Now falls through to a typed input prompt.
    print("Falling back to manual text entry.")
    return input("Please type your certificate type and grades: ")


# ============================================================
# INTEREST QUESTIONS -> FEATURE VECTOR
# ============================================================
FEATURE_QUESTIONS = {
    "assembling": "Do you enjoy putting parts together to create something new?",
    "bikes": "When you were younger, did you like fixing or adjusting your bicycle?",
    "car": "Do you find car engines or vehicles fascinating to explore?",
    "cardboard": "Did you enjoy building models or projects using simple materials like paper or cardboard?",
    "tinkering": "Do you like opening gadgets to see how they work inside?",
    "tools": "Do you feel comfortable using tools like screwdrivers, pliers, or wrenches?",
    "circuits": "Have you ever enjoyed connecting wires or playing with small electronic kits?",
    "bulbs": "When you see a light bulb, do you get curious about how electricity makes it glow?",
    "wires": "Do you like experimenting with wires or setting up small electric connections?",
    "windmills": "Do renewable energy ideas, like wind or solar, catch your interest?",
    "connecting": "Do you enjoy linking things together to make them work, like cables or devices?",
    "climate": "Are you curious about weather changes, like why it rains or gets hotter?",
    "tasks": "Do you enjoy planning your day and completing tasks in an organized way?",
    "time": "Are you usually careful with managing your time when working on projects?",
    "toys": "When you were a child, did you prefer creating or modifying your toys?",
    "teamwork": "Do you enjoy working in groups to solve a problem?",
    "work": "Do you like doing projects that require planning and effort over time?",
    "reading": "Do you enjoy learning by reading books or articles in your free time?",
    "building": "Do you like creating things from scratch, whether physical or digital?",
    "likes_computers": "Do you enjoy working with computers or learning about how they work?",
    "likes_coding": "Do you like writing code or solving programming challenges?",
    "likes_drawing": "Do you enjoy drawing or creating visual designs?",
    "likes_science": "Are you curious about how the world works and enjoy science experiments?",
    "likes_math": "Do you find solving math problems fun and satisfying?",
    "likes_problem_solving": "Do you enjoy solving logical or real-life problems?",
    "likes_robots": "Are you fascinated by robots and how they move or think?",
    "likes_circuits": "Do you enjoy playing with electronic kits, wires, or circuit boards?",
    "likes_design": "Do you like designing things, such as posters, models, or systems?",
    "likes_games": "Do you like playing or thinking about how to design games?",
    "likes_data": "Do you enjoy analyzing data, patterns, or statistics?",
    "likes_ai": "Are you interested in artificial intelligence and how machines can learn?",
    "likes_security": "Do you find cybersecurity and protecting systems interesting?",
    "likes_teamwork": "Do you enjoy working in teams and collaborating with others?",
    "likes_business": "Do you enjoy planning, organizing, or thinking about how businesses work?",
}


def ask_indirect_questions(expected_features):
    features = {}
    for feat in expected_features:
        if feat in FEATURE_QUESTIONS:
            ans = input(f"{FEATURE_QUESTIONS[feat]} (yes/no): ")
            features[feat] = 1 if ans.strip().lower() in ["yes", "y"] else 0
        else:
            features[feat] = 0
    return [[features[f] for f in expected_features]]


# ============================================================
# MAJOR RECOMMENDATION
# ============================================================
def recommend_major(eligible_list, feature_vector, ml_model, model_kind, label_encoder):
    if model_kind == "keras":
        probs = ml_model.predict(np.array(feature_vector), verbose=0)
        pred_idx = int(np.argmax(probs, axis=1)[0])
    else:
        pred_idx = ml_model.predict(feature_vector)[0]
    # FIX: original always called ml_model.predict(feature_vector)[0] as if
    # it were an sklearn model. If the TensorFlow ANN had won training, this
    # would either crash or silently return raw probabilities instead of a
    # class index, since Keras' .predict() output shape/semantics differ.

    pred_major = label_encoder.inverse_transform([pred_idx])[0]
    print(eligible_list)

    eligible_majors = [prog for _tier, prog in eligible_list]
    if pred_major in eligible_majors:
        return f"✅ Recommended Major: {pred_major}, which is eligible based on your grades ({eligible_majors})."
    elif eligible_list:
        return f"⚠️ Your strongest interest match is {pred_major}, but your eligible majors are {eligible_majors}."
    else:
        return f"❌ Your strongest interest match is {pred_major}, but unfortunately no majors are eligible based on your grades."
    # FIX: `any(pred_major in prog for prog in eligible_list)` compared a
    # string against a list of [tier_name, major_name] pairs, which is
    # almost never true even for a genuine match (substring check against a
    # 2-element list, not the major name itself). Now compares against the
    # extracted major names directly.


# ============================================================
# RAG-STYLE Q&A OVER POLICY DOCUMENTS
# ============================================================
def build_bow_corpus(chunks):
    corpus = {}
    for i, chunk in enumerate(chunks):
        tokens = word_tokenize(chunk.lower())
        corpus[i] = dict(FreqDist(tokens))
    return pd.DataFrame.from_records(corpus).fillna(0).astype(int).T


def retrieve(query, chunks, bow_df, k=4):
    tokens = word_tokenize(query.lower())
    freq = FreqDist(tokens)
    q_vec = np.array([freq.get(word, 0) for word in bow_df.columns])

    scores = []
    for idx, row in bow_df.iterrows():
        score = np.dot(q_vec, row.values)
        scores.append((chunks[idx], score, "HTU_policy"))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:k]
    # FIX: original rebuilt the entire BoW matrix from scratch on every
    # single call to retrieve() (i.e. on every question asked), which is
    # extremely slow. The BoW matrix is now built once and passed in.


def answer(query, df_chunks, bow_df, qa_pipeline, nli_pipeline, k=4):
    ctx = retrieve(query, df_chunks["text"].tolist(), bow_df, k)
    context = "\n\n".join(c for c, _s, _src in ctx if isinstance(c, str) and c.strip())

    out = qa_pipeline(question=query, context=context)
    ans = out.get("answer", "").strip()

    raw = nli_pipeline({"text": context[:1500], "text_pair": ans})
    ent = {d["label"]: float(d["score"]) for d in raw[0]} if raw and isinstance(raw[0], list) else \
          {d["label"]: float(d["score"]) for d in raw}
    # FIX: with top_k=None, the `text-classification` pipeline returns a
    # list-of-lists (one list of label scores per input). The original code
    # assumed a flat list, which would raise a KeyError/TypeError on the
    # dict comprehension. This handles both shapes defensively.

    return ans, context, ent


# ============================================================
# MAIN INTERACTIVE LOOP
# ============================================================
def advisor_system():
    df = pd.read_csv("htu_majors_interests.csv")
    expected_features = [c for c in df.columns if c != "label"]

    ml_model, model_kind = load_major_predictor()
    label_encoder = joblib.load("label_encoder.pkl")

    df_chunks = pd.read_csv("dept_labeled_chunks_cleaned_Policy.csv")
    bow_df = build_bow_corpus(df_chunks["text"].tolist())

    qa_pipeline = pipeline("question-answering", model="deepset/tinyroberta-squad2", device=-1)
    nli_pipeline = pipeline("text-classification", model="facebook/bart-large-mnli",
                             top_k=None, device=-1)
    # FIX: loading the ML model, label encoder, policy chunks and both
    # HuggingFace pipelines used to happen at *import time*, at the module's
    # top level. That meant simply `import`ing this file (e.g. to reuse a
    # helper function) would try to load multi-GB transformer models and
    # crash if any of the CSV/pkl files were missing. Now everything lives
    # inside advisor_system() and only runs when you actually start the app.

    while True:
        print("🎓 Welcome to the HTU Advisor System 🎓")
        mode = input("Choose mode: (1) Advising Flow \t (2) Free Q&A \t (3) Exit: ")

        if mode == "1":
            text = listen_for_grades()
            cert_type, grades = extract_grades(text)
            student_facts = build_student_facts(cert_type, grades)
            eligible_list = infer_eligible_programs(student_facts)

            fv = ask_indirect_questions(expected_features)
            result = recommend_major(eligible_list, fv, ml_model, model_kind, label_encoder)
            print(result)
            speak(result)

        elif mode == "2":
            q = input("Ask your advising question: ")
            ans, evidence, ent = answer(q, df_chunks, bow_df, qa_pipeline, nli_pipeline)
            print("\n" + ans)
            print("\n**Evidence:** " + evidence)
            print("\nEntailment Scores:", ent)

        elif mode == "3":
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid mode selected. Please choose 1, 2, or 3.")


if __name__ == "__main__":
    advisor_system()
