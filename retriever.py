from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from chatbot_config import KB_DIR, PROCESSED_KB, TOP_K, MIN_RETRIEVAL_SCORE
from entity_extractor import normalize


CATEGORY_MAP = {
    "admission_requirements": "admission",
    "certificate_equivalency": "admission",
    "application_process": "application",
    "interview_process": "interview",
    "major_information": "major_info",
    "courses": "courses",
    "faq": "general",
    "contacts": "contacts",
    "scholarships": "scholarships",
    "tuition": "fees",
    "study_plans": "study_plan",
    "degree_rules": "academic_rules",
    "university_information": "university_info",
    "financial_rules": "fees",
    "deadlines": "deadlines",
}


def _row_text(row: pd.Series, ignored: set[str]) -> str:
    parts: list[str] = []
    for col, value in row.items():
        if col in ignored or pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() not in {"nan", "none"}:
            parts.append(f"{col.replace('_', ' ')}: {text}")
    return " | ".join(parts)


def _load_structured_csv(path: Path) -> list[dict[str, Any]]:
    name = path.stem
    category = CATEGORY_MAP.get(name, "general")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    docs: list[dict[str, Any]] = []

    for index, row in df.iterrows():
        major = ""
        for col in ("major", "programme_title", "program", "programme"):
            if col in df.columns and str(row.get(col, "")).strip():
                major = str(row[col]).strip()
                break

        title = name.replace("_", " ").title()
        for col in ("title", "question", "course_title", "fee_name", "event", "scholarship", "component"):
            if col in df.columns and str(row.get(col, "")).strip():
                title = str(row[col]).strip()
                break

        source = str(row.get("source", path.name)).strip() or path.name
        source_page = str(row.get("source_page", "")).strip()
        if source_page:
            source = f"{source} — page {source_page}"

        text = _row_text(row, {"source"})
        if text:
            docs.append({
                "chunk_id": f"{name}_{index:04d}",
                "category": category,
                "title": title,
                "text": text,
                "major": major,
                "source": source,
            })
    return docs


class HybridRetriever:
    """Offline keyword retriever over both processed chunks and structured CSV files."""

    def __init__(self):
        docs: list[dict[str, Any]] = []

        if PROCESSED_KB.exists():
            base = pd.read_csv(PROCESSED_KB, dtype=str, keep_default_na=False)
            for index, row in base.iterrows():
                docs.append({
                    "chunk_id": str(row.get("chunk_id", f"processed_{index}")),
                    "category": str(row.get("category", "general")),
                    "title": str(row.get("title", "HTU information")),
                    "text": str(row.get("text", "")),
                    "major": str(row.get("major", "")),
                    "source": str(row.get("source", "HTU knowledge base")),
                })

        for path in sorted(KB_DIR.glob("*.csv")):
            docs.extend(_load_structured_csv(path))

        self.df = pd.DataFrame(docs).fillna("").drop_duplicates(subset=["category", "title", "text"])
        if self.df.empty:
            raise RuntimeError("The knowledge base is empty. Add CSV files inside knowledge_base/.")

        corpus = (
            self.df["title"].astype(str) + " "
            + self.df["text"].astype(str) + " "
            + self.df["major"].astype(str) + " "
            + self.df["category"].astype(str)
        ).map(normalize)

        self.word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
        self.word_matrix = self.word.fit_transform(corpus)
        self.char_matrix = self.char.fit_transform(corpus)

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        major: str | None = None,
        top_k: int = TOP_K,
    ) -> list[dict[str, Any]]:
        q = normalize(query)
        word_score = cosine_similarity(self.word.transform([q]), self.word_matrix).ravel()
        char_score = cosine_similarity(self.char.transform([q]), self.char_matrix).ravel()
        scores = 0.55 * word_score + 0.45 * char_score

        mask = np.ones(len(self.df), dtype=bool)
        category_aliases = {
            "eligibility": {"admission"},
            "requirements": {"admission"},
            "application": {"application", "deadlines", "contacts", "general"},
            "fees": {"fees", "scholarships"},
            "interview": {"interview", "admission"},
            "major_info": {"major_info", "courses"},
            "comparison": {"major_info", "courses"},
            "courses": {"courses"},
            "prerequisites": {"courses"},
            "study_plan": {"study_plan", "courses"},
            "academic_rules": {"academic_rules", "general"},
            "university_info": {"university_info", "contacts", "general"},
            "contacts": {"contacts"},
            "scholarships": {"scholarships", "fees"},
            "deadlines": {"deadlines", "application"},
        }
        if category and category != "general":
            allowed = category_aliases.get(category, {category})
            candidate_mask = self.df["category"].isin(allowed).to_numpy()
            if candidate_mask.any():
                mask &= candidate_mask

        if major:
            major_norm = normalize(major)
            major_mask = (
                self.df["major"].astype(str).map(normalize).str.contains(major_norm, regex=False)
                | self.df["text"].astype(str).map(normalize).str.contains(major_norm, regex=False)
                | self.df["title"].astype(str).map(normalize).str.contains(major_norm, regex=False)
            ).to_numpy()
            if major_mask.any():
                mask &= major_mask

        ranked = np.argsort(np.where(mask, scores, -1))[::-1]
        results: list[dict[str, Any]] = []
        for index in ranked:
            if scores[index] < MIN_RETRIEVAL_SCORE:
                continue
            row = self.df.iloc[index]
            results.append({
                "text": str(row["text"]),
                "title": str(row["title"]),
                "source": str(row["source"]),
                "category": str(row["category"]),
                "major": str(row["major"]),
                "score": float(scores[index]),
            })
            if len(results) >= top_k:
                break
        return results
