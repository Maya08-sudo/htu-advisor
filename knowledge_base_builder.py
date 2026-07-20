from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
from chatbot_config import BASE_DIR, PROCESSED_KB, SEED_CHUNKS_CSV, COURSES_CSV

MAJOR_FROM_FILE = {
    "Electrical_StudyPlan 1.txt": "Electrical Engineering",
    "Energy_StudyPlan 1.txt": "Energy Engineering",
    "IE_StudyPlan 1.txt": "Industrial Engineering",
    "Mechanical_StudyPlan 1.txt": "Mechanical Engineering",
}

def chunks(text: str, size: int = 900, overlap: int = 120):
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    out=[]; start=0
    while start < len(text):
        end=min(len(text), start+size)
        if end < len(text):
            cut=text.rfind(". ", start, end)
            if cut > start+size//2: end=cut+1
        out.append(text[start:end].strip())
        if end == len(text): break
        start=max(start+1, end-overlap)
    return out

def build() -> Path:
    records=[]
    seed=pd.read_csv(SEED_CHUNKS_CSV)
    for i,row in seed.iterrows():
        records.append({
            "chunk_id":f"seed_{i:04d}", "category":row.get("department","general"),
            "title":str(row.get("department","General")).replace("_"," ").title(),
            "text":str(row["text"]), "language":"en", "major":"",
            "source":str(row.get("source","HTU source")), "status":str(row.get("status","prototype"))
        })
    courses=pd.read_csv(COURSES_CSV)
    for major,g in courses.groupby("major"):
        text="Courses currently indexed for " + major + ": " + "; ".join(
            f"{r.course_code} — {r.course_name} ({r.credit_hours} credit hours; prerequisite: {r.prerequisite})"
            for r in g.itertuples())
        records.append({"chunk_id":f"courses_{len(records):04d}","category":"courses","title":f"{major} courses",
                        "text":text,"language":"en","major":major,"source":"courses_seed.csv","status":"prototype"})
    source_dir=BASE_DIR/"knowledge_base"/"source"
    for filename,major in MAJOR_FROM_FILE.items():
        path=source_dir/filename
        if path.exists():
            for j,ch in enumerate(chunks(path.read_text(encoding="utf-8",errors="ignore"))):
                records.append({"chunk_id":f"study_{len(records):04d}","category":"study_plan","title":f"{major} study plan",
                                "text":ch,"language":"en","major":major,"source":filename,"status":"source"})
    # Add a compact verified admission-policy overview from the supplied policy.
    policy_notes = [
        ("admission","General entry and scoring","BSc minimum entry is generally 80%, while technical and technician degrees generally use 70%. Competitive assessment may include high-school GPA, mathematics, physics, interview, introductory video, admission tests, and bonus points for documented extracurricular activities.","Untitled document.pdf, pages 1-2"),
        ("admission","Engineering Tawjihi requirements","For Mechanical Engineering, Energy Engineering, Electrical Engineering and Industrial Engineering BSc, the listed Tawjihi minimums are 80% in mathematics, 80% in physics and 80% overall, with all three conditions required.","Untitled document.pdf, page 1"),
        ("admission","IGCSE engineering requirements","For the listed engineering BSc programmes, IGCSE requirements include grade B in mathematics, grade B in physics and 80% Tawjihi equivalency; permitted A-level/AS-level combinations are also specified.","Untitled document.pdf, page 2"),
        ("admission","IB engineering requirements","For the listed engineering BSc programmes, IB requirements are mathematics 5, physics 5 and 80% Tawjihi equivalency, with acceptable HL/SL combinations stated in the policy.","Untitled document.pdf, page 3"),
    ]
    for cat,title,text,source in policy_notes:
        records.append({"chunk_id":f"policy_{len(records):04d}","category":cat,"title":title,"text":text,"language":"en","major":"","source":source,"status":"verified_extract"})
    df=pd.DataFrame(records).drop_duplicates(subset=["text"]).reset_index(drop=True)
    PROCESSED_KB.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(PROCESSED_KB,index=False,encoding="utf-8-sig")
    return PROCESSED_KB

if __name__ == "__main__":
    path=build(); print(f"Built {path}")
