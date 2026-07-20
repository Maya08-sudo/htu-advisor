from __future__ import annotations

from typing import Dict

import pandas as pd

from chatbot_config import KB_DIR
from entity_extractor import extract_entities, normalize
from intent_classifier import classify_intent
from retriever import HybridRetriever
from rule_engine import RuleEngine


DEGREE_MAP = {
    "bachelor": "BSc",
    "bsc": "BSc",
    "tech": "Technical",
    "technical": "Technical",
    "technician": "Technician",
}


class AdmissionRAG:
    """Deterministic HTU advisory engine with retrieval fallback."""

    def __init__(self):
        self.retriever = HybridRetriever()
        self.rules = RuleEngine()
        self.tables: dict[str, pd.DataFrame] = {}

        table_names = [
            "application_process",
            "tuition",
            "major_information",
            "courses",
            "study_plans",
            "scholarships",
            "deadlines",
            "contacts",
            "interview_process",
            "admission_requirements",
            "certificate_equivalency",
            "faq",
            "degree_rules",
            "university_information",
            "financial_rules",
        ]

        for name in table_names:
            path = KB_DIR / f"{name}.csv"
            if path.exists():
                # dtype=str preserves leading zeroes in HTU course numbers.
                self.tables[name] = pd.read_csv(
                    path,
                    dtype=str,
                    keep_default_na=False,
                )
            else:
                self.tables[name] = pd.DataFrame()

    @staticmethod
    def _source(page: str, source: str = "HTU source") -> str:
        page = str(page or "").strip()
        source = str(source or "HTU source").strip()
        return f"{source}, page {page}" if page else source

    @staticmethod
    def _same_major(left: str, right: str) -> bool:
        def canonical(value: str) -> str:
            value = normalize(value).replace("&", " and ")
            aliases = {
                "cybersecurity": "cyber security",
                "data science & artificial intelligence":
                    "data science and artificial intelligence",
                "game design & development":
                    "game design and development",
            }
            value = aliases.get(value, value)
            return " ".join(value.split())

        left_value = canonical(left)
        right_value = canonical(right)
        return (
            left_value == right_value
            or left_value in right_value
            or right_value in left_value
        )

    @staticmethod
    def _degree_name(degree: str | None) -> str | None:
        if not degree:
            return None
        return DEGREE_MAP.get(str(degree).lower(), degree)

    @staticmethod
    def _current_rows(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty or "plan_status" not in rows.columns:
            return rows

        current = rows[
            rows["plan_status"].astype(str).str.lower().eq("current")
        ]
        return current if not current.empty else rows

    def _application_answer(self, language: str):
        table = self.tables["application_process"].copy()
        if table.empty:
            return None

        table["step_number"] = pd.to_numeric(
            table.get("step", ""),
            errors="coerce",
        )
        table = table.sort_values("step_number")

        heading = (
            "خطوات التقديم من البداية حتى القبول:"
            if language == "ar"
            else "The application process from start to admission is:"
        )
        lines = [heading]

        for _, row in table.iterrows():
            number = str(row.get("step", "")).strip()
            title = str(row.get("title", "")).strip()
            description = str(row.get("description", "")).strip()
            action = str(row.get("required_action", "")).strip()

            if language == "ar":
                lines.append(
                    f"{number}. **{title}** — {description} "
                    f"الإجراء المطلوب: {action}"
                )
            else:
                lines.append(
                    f"{number}. **{title}** — {description} "
                    f"Required action: {action}"
                )

        sources = list(
            dict.fromkeys(
                self._source(page)
                for page in table.get("source_page", [])
            )
        )
        return "\n\n".join(lines), sources

    def _video_answer(self, language: str):
        if language == "ar":
            answer = (
                "يجب أن تعرّف بنفسك، وتتحدث عن هواياتك، "
                "وأنشطتك اللامنهجية، وسبب اختيارك لجامعة الحسين "
                "التقنية، وطموحاتك المستقبلية. مدة الفيديو دقيقة واحدة."
            )
        else:
            answer = (
                "Introduce yourself and speak about your hobbies, "
                "extracurricular activities, why you selected HTU, "
                "and your future aspirations. The video must be one minute."
            )
        return answer, ["HTU Admission Guide, page 21"]

    def _bonus_answer(self, language: str):
        if language == "ar":
            answer = (
                "نعم. يمكن للأنشطة اللامنهجية والشهادات الإضافية "
                "أن تضيف **حتى 10 نقاط إضافية** ضمن التقييم، بشرط "
                "رفع الشهادات مع مرفقات طلب الالتحاق."
            )
        else:
            answer = (
                "Yes. Extracurricular activities and additional certificates "
                "can add **up to 10 bonus points**, provided the supporting "
                "certificates are uploaded with the application."
            )
        return answer, ["HTU Admission Guide, page 17"]

    def _fees_answer(self, question: str, language: str):
        table = self.tables["tuition"].copy()
        if table.empty:
            return None

        normalized = normalize(question)
        selected = table

        if any(
            phrase in normalized
            for phrase in ["registration fee", "رسوم التسجيل", "تسجيل"]
        ):
            selected = table[
                table["fee_name"].str.contains(
                    "Registration",
                    case=False,
                    na=False,
                )
            ]
        elif any(
            phrase in normalized
            for phrase in ["application fee", "رسوم الطلب", "رسوم التقديم"]
        ):
            selected = table[
                table["fee_name"].str.contains(
                    "Application",
                    case=False,
                    na=False,
                )
            ]
        elif any(
            phrase in normalized
            for phrase in [
                "credit hour",
                "per credit",
                "رسوم الساعة",
                "ساعة معتمدة",
            ]
        ):
            selected = table[
                table["fee_category"].str.contains(
                    "Credit",
                    case=False,
                    na=False,
                )
            ]

        if selected.empty:
            selected = table.head(6)

        lines = [
            "الرسوم ذات الصلة:"
            if language == "ar"
            else "Relevant fees:"
        ]

        for _, row in selected.iterrows():
            name = row.get("fee_name", "")
            regular = row.get("regular_program_jod", "")
            international = row.get("international_program_jod", "")
            unit = row.get("unit", "")

            if language == "ar":
                lines.append(
                    f"- **{name}**: البرنامج العادي {regular} د.أ، "
                    f"والبرنامج الدولي {international} د.أ ({unit})."
                )
            else:
                lines.append(
                    f"- **{name}**: JOD {regular} for the regular programme; "
                    f"JOD {international} for the international programme "
                    f"({unit})."
                )

        sources = list(
            dict.fromkeys(
                self._source(row.get("source_page", ""), row.get("source", ""))
                for _, row in selected.iterrows()
            )
        )
        return "\n".join(lines), sources

    def _major_answer(self, major: str | None, language: str):
        if not major:
            return None

        table = self.tables["major_information"]
        rows = table[
            table["major"].map(
                lambda value: self._same_major(str(value), major)
            )
        ]
        if rows.empty:
            return None

        row = rows.iloc[0]
        school = row.get("school", "")
        levels = row.get("degree_levels", "")
        overview = row.get("overview", "")

        if language == "ar":
            answer = (
                f"**{major}** يتبع {school}. "
                f"الدرجات المتاحة: {levels}.\n\n{overview}"
            )
        else:
            answer = (
                f"**{major}** is offered by the {school}. "
                f"Available degree levels: {levels}.\n\n{overview}"
            )

        return answer, [
            self._source(
                row.get("source_page", ""),
                row.get("source", "HTU study plan"),
            )
        ]

    def _courses_answer(
        self,
        major: str | None,
        degree: str | None,
        language: str,
        prerequisites_only: bool = False,
    ):
        if not major:
            return None

        table = self.tables["courses"]
        rows = table[
            table["major"].map(
                lambda value: self._same_major(str(value), major)
            )
        ].copy()
        rows = self._current_rows(rows)

        requested_degree = self._degree_name(degree)
        if requested_degree and "degree_level" in rows.columns:
            degree_rows = rows[
                rows["degree_level"].astype(str).str.lower().eq(
                    requested_degree.lower()
                )
            ]
            if not degree_rows.empty:
                rows = degree_rows

        if rows.empty:
            return None

        # Keep distinct plans separate. If the user did not identify a degree and
        # several levels exist, present each level in its own section.
        sort_columns = [
            column
            for column in [
                "degree_level",
                "requirement_group",
                "course_code",
            ]
            if column in rows.columns
        ]
        rows = rows.drop_duplicates(
            subset=["degree_level", "course_code"],
            keep="first",
        ).sort_values(sort_columns)

        if prerequisites_only:
            rows = rows[
                rows["prerequisites"].astype(str).str.strip().ne("")
            ]
            if rows.empty:
                message = (
                    "لا توجد متطلبات سابقة موثقة لهذا الاختيار في البيانات الحالية."
                    if language == "ar"
                    else "No documented prerequisites were found for this selection."
                )
                return message, []

            heading = (
                f"المتطلبات السابقة لمواد **{major}**:"
                if language == "ar"
                else f"Course prerequisites for **{major}**:"
            )
            lines = [heading]
        else:
            heading = (
                f"المواد الموجودة في الخطط الرسمية لتخصص **{major}**:"
                if language == "ar"
                else f"Courses in the official **{major}** study plans:"
            )
            lines = [heading]

        active_degree = None
        active_group = None
        shown = 0

        for _, row in rows.iterrows():
            row_degree = str(row.get("degree_level", "")).strip()
            group = str(row.get("requirement_group", "Study Plan")).strip()
            code = str(row.get("course_code", "")).strip()
            title = str(row.get("course_title", "")).strip()
            credit = str(row.get("credit_hours", "")).strip()
            prerequisite = str(row.get("prerequisites", "")).strip()

            if row_degree != active_degree:
                lines.append(f"\n### {row_degree or 'Study Plan'}")
                active_degree = row_degree
                active_group = None

            if group != active_group:
                lines.append(f"\n**{group}**")
                active_group = group

            if prerequisites_only:
                lines.append(
                    f"- **{code} — {title}**: {prerequisite}"
                )
            else:
                credit_text = f" — {credit} CH" if credit else ""
                lines.append(f"- {code} — {title}{credit_text}")

            shown += 1
            if shown >= 60:
                lines.append(
                    "\n_The response is limited to 60 records. "
                    "Ask for a specific degree or requirement group for a shorter list._"
                )
                break

        sources = list(
            dict.fromkeys(
                str(value)
                for value in rows["source"].head(10)
            )
        )
        return "\n".join(lines), sources

    def _study_plan_answer(
        self,
        major: str | None,
        degree: str | None,
        language: str,
    ):
        if not major:
            return None

        table = self.tables["study_plans"]
        rows = table[
            table["programme_title"].map(
                lambda value: self._same_major(str(value), major)
            )
        ].copy()
        rows = self._current_rows(rows)

        requested_degree = self._degree_name(degree)
        if requested_degree:
            degree_rows = rows[
                rows["degree_level"].astype(str).str.lower().eq(
                    requested_degree.lower()
                )
            ]
            if not degree_rows.empty:
                rows = degree_rows

        if rows.empty:
            return None

        heading = (
            f"الساعات المعتمدة الرسمية لتخصص **{major}**:"
            if language == "ar"
            else f"Official credit-hour totals for **{major}**:"
        )
        lines = [heading]

        rows = rows.drop_duplicates(
            subset=["degree_level", "total_credit_hours"],
            keep="first",
        )
        for _, row in rows.iterrows():
            lines.append(
                f"- **{row.get('degree_level', '')}**: "
                f"{row.get('total_credit_hours', '')} credit hours."
            )

        return "\n".join(lines), list(
            dict.fromkeys(str(value) for value in rows["source"])
        )

    def _interview_answer(self, question: str, language: str):
        normalized = normalize(question)
        if (
            "admission test" in normalized
            or "اختبار قبول" in normalized
        ):
            answer = (
                "اختبار القبول يقيس التفكير المنطقي بطرق مختلفة "
                "ولا يحتاج إلى تحضير مسبق."
                if language == "ar"
                else
                "The admissions test evaluates logical reasoning through "
                "different methods and does not require prior preparation."
            )
            return answer, ["HTU Admission Guide, page 21"]

        table = self.tables["interview_process"]
        if table.empty:
            return None

        lines = [
            "مكونات التقييم والمقابلة:"
            if language == "ar"
            else "Interview and assessment components:"
        ]

        for _, row in table.iterrows():
            lines.append(
                f"- **{row.get('programme_group', '')} — "
                f"{row.get('component', '')} "
                f"({row.get('weight_points', '')} points)**: "
                f"{row.get('assessment_focus', '')}"
            )

        sources = list(
            dict.fromkeys(
                self._source(
                    row.get("source_page", ""),
                    row.get("source", "HTU Admission Guide"),
                )
                for _, row in table.iterrows()
            )
        )
        return "\n".join(lines), sources

    def _simple_table(self, table_name: str, language: str):
        table = self.tables[table_name]
        if table.empty:
            return None

        if table_name == "scholarships":
            heading = (
                "المنح والمساعدات المالية:"
                if language == "ar"
                else "Scholarships and financial aid:"
            )
            lines = [heading]
            for _, row in table.iterrows():
                lines.append(
                    f"- **{row.get('scholarship', '')}**: "
                    f"{row.get('eligibility', '')} "
                    f"{row.get('coverage', '')} "
                    f"{row.get('application_method', '')}"
                )
        elif table_name == "deadlines":
            heading = (
                "المواعيد والفترات المتاحة في الوثائق:"
                if language == "ar"
                else "Dates and periods available in the documents:"
            )
            lines = [heading]
            for _, row in table.iterrows():
                lines.append(
                    f"- **{row.get('event', '')}**: "
                    f"{row.get('date_or_period', '')}. "
                    f"{row.get('notes', '')}"
                )
        else:
            heading = (
                "جهات التواصل:"
                if language == "ar"
                else "Contacts:"
            )
            lines = [heading]
            for _, row in table.iterrows():
                contact = row.get("contact", "") or "Not listed"
                lines.append(
                    f"- **{row.get('office', '')}** — "
                    f"{row.get('contact_type', '')}: {contact}. "
                    f"{row.get('purpose', '')}"
                )

        sources = list(
            dict.fromkeys(
                self._source(
                    row.get("source_page", ""),
                    row.get("source", "HTU source"),
                )
                for _, row in table.iterrows()
            )
        )
        return "\n".join(lines), sources

    def _bilingual_table(self, table_name: str, language: str):
        table = self.tables[table_name]
        if table.empty:
            return None

        if table_name == "degree_rules":
            heading = (
                "الدرجات والشهادات الأكاديمية:"
                if language == "ar"
                else "Academic degrees and certificates:"
            )
            arabic_column = "rule_ar"
            english_column = "rule_en"
        elif table_name == "university_information":
            heading = (
                "معلومات عن جامعة الحسين التقنية:"
                if language == "ar"
                else "Information about HTU:"
            )
            arabic_column = "information_ar"
            english_column = "information_en"
        else:
            heading = (
                "القواعد المالية ذات الصلة:"
                if language == "ar"
                else "Relevant financial rules:"
            )
            arabic_column = "rule_ar"
            english_column = "rule_en"

        lines = [heading]
        for _, row in table.iterrows():
            text = (
                row.get(arabic_column, "")
                if language == "ar"
                else row.get(english_column, "")
            )
            lines.append(
                f"- **{row.get('topic', '')}**: {text}"
            )

        sources = list(
            dict.fromkeys(
                self._source(
                    row.get("source_page", ""),
                    row.get("source", "HTU regulation"),
                )
                for _, row in table.iterrows()
            )
        )
        return "\n".join(lines), sources

    def _degree_comparison(self, language: str):
        if language == "ar":
            answer = (
                "**البكالوريوس:** الدرجة الجامعية الأولى، وعادة تكون "
                "الخطة الأطول والأوسع.\n\n"
                "**الدرجة التقنية:** درجة جامعية متوسطة ذات تركيز "
                "تطبيقي وتقني أكبر.\n\n"
                "**درجة الفني:** شهادة أقصر تركز على المهارات العملية "
                "الأساسية.\n\n"
                "تختلف الساعات حسب التخصص، لذلك يجب تحديد التخصص "
                "للحصول على الأرقام الدقيقة."
            )
        else:
            answer = (
                "**BSc:** the first university degree and normally the "
                "broadest and longest study plan.\n\n"
                "**Technical degree:** an intermediate university degree "
                "with a stronger applied technical focus.\n\n"
                "**Technician degree:** a shorter qualification focused "
                "on core practical skills.\n\n"
                "Credit totals vary by programme, so specify the major "
                "for exact figures."
            )

        return answer, [
            "HTU degree-awarding regulation",
            "Official HTU study plans",
        ]

    def answer(
        self,
        question: str,
        context: dict | None = None,
    ) -> Dict[str, object]:
        context = context or {}
        entities = extract_entities(question, context)
        intent = classify_intent(question)

        # Natural follow-ups such as "tell me more about computer since"
        # should become a major-information request even when speech
        # recognition contains small mistakes.
        if intent == "general" and entities.major_explicit:
            intent = "major_info"

        updated_context = dict(context)
        if entities.major_explicit and entities.major:
            updated_context = {"major": entities.major}
            if entities.degree:
                updated_context["degree"] = entities.degree
            if entities.certificate:
                updated_context["certificate"] = entities.certificate
        else:
            for key in ("major", "certificate", "degree"):
                value = getattr(entities, key)
                if value:
                    updated_context[key] = value

        structured = None

        if intent == "video":
            structured = self._video_answer(entities.language)
        elif intent == "bonus":
            structured = self._bonus_answer(entities.language)
        elif intent == "application":
            structured = self._application_answer(entities.language)
        elif intent == "fees":
            structured = self._fees_answer(question, entities.language)
        elif intent == "scholarships":
            structured = self._simple_table(
                "scholarships",
                entities.language,
            )
        elif intent == "deadlines":
            structured = self._simple_table(
                "deadlines",
                entities.language,
            )
        elif intent == "contacts":
            structured = self._simple_table(
                "contacts",
                entities.language,
            )
        elif intent == "major_info":
            structured = self._major_answer(
                entities.major,
                entities.language,
            )
        elif intent == "study_plan":
            structured = self._study_plan_answer(
                entities.major,
                entities.degree,
                entities.language,
            )
        elif intent == "courses":
            structured = self._courses_answer(
                entities.major,
                entities.degree,
                entities.language,
                prerequisites_only=False,
            )
        elif intent == "prerequisites":
            structured = self._courses_answer(
                entities.major,
                entities.degree,
                entities.language,
                prerequisites_only=True,
            )
        elif intent == "interview":
            structured = self._interview_answer(
                question,
                entities.language,
            )
        elif intent == "academic_rules":
            structured = self._bilingual_table(
                "degree_rules",
                entities.language,
            )
        elif intent == "university_info":
            structured = self._bilingual_table(
                "university_information",
                entities.language,
            )
        elif (
            intent == "comparison"
            and any(
                token in normalize(question)
                for token in [
                    "bsc",
                    "bachelor",
                    "technical",
                    "technician",
                    "بكالوريوس",
                    "تقنية",
                    "فني",
                ]
            )
        ):
            structured = self._degree_comparison(entities.language)

        if (
            intent in {"major_info", "courses", "study_plan", "prerequisites"}
            and not entities.major
        ):
            message = (
                "أي تخصص تقصد؟ اكتب اسم التخصص، مثل علم الحاسوب، "
                "الأمن السيبراني، الهندسة الكهربائية، أو هندسة الطاقة."
                if entities.language == "ar"
                else
                "Which major do you mean? For example: Computer Science, "
                "Cyber Security, Electrical Engineering, or Energy Engineering."
            )
            return {
                "found": False,
                "answer": message,
                "snippets": [],
                "sources": [],
                "scores": [],
                "confidence": "low",
                "intent": intent,
                "entities": entities.__dict__,
                "context": updated_context,
            }

        if structured:
            answer, sources = structured
            return {
                "found": True,
                "answer": answer,
                "snippets": [],
                "sources": sources,
                "scores": [1.0],
                "confidence": "high",
                "intent": intent,
                "entities": entities.__dict__,
                "context": updated_context,
            }

        if (
            intent in {"requirements", "eligibility"}
            and entities.major
            and entities.certificate in (None, "tawjihi")
        ):
            if (
                intent == "eligibility"
                and None not in (
                    entities.average,
                    entities.math,
                    entities.physics,
                )
            ):
                ruled = self.rules.check(
                    entities.major,
                    entities.average,
                    entities.math,
                    entities.physics,
                    entities.language,
                )
            else:
                ruled = self.rules.requirements(
                    entities.major,
                    entities.language,
                )

            if ruled:
                return {
                    "found": True,
                    "answer": ruled["answer"],
                    "snippets": [],
                    "sources": ["HTU admission policy"],
                    "scores": [1.0],
                    "confidence": "high",
                    "intent": intent,
                    "entities": entities.__dict__,
                    "context": updated_context,
                }

        results = self.retriever.search(
            question,
            category=intent,
            major=entities.major,
            top_k=3,
        )
        if not results:
            results = self.retriever.search(
                question,
                major=entities.major,
                top_k=3,
            )

        if not results:
            message = (
                "لم أجد إجابة موثوقة في البيانات الحالية. "
                "حدّد التخصص أو نوع الشهادة، أو تواصل مع القبول والتسجيل."
                if entities.language == "ar"
                else
                "I could not find a verified answer in the current data. "
                "Specify the major or certificate type, or contact "
                "Admissions and Registration."
            )
            return {
                "found": False,
                "answer": message,
                "snippets": [],
                "sources": [],
                "scores": [],
                "confidence": "low",
                "intent": intent,
                "entities": entities.__dict__,
                "context": updated_context,
            }

        # Retrieval is a last resort. Do not concatenate unrelated rows or
        # expose raw CSV field labels to the student.
        top = results[0]
        if top["score"] < 0.18:
            message = (
                "لم أجد تطابقاً موثوقاً كافياً. أعد صياغة السؤال وحدد "
                "التخصص أو موضوع القبول المطلوب."
                if entities.language == "ar"
                else
                "I did not find a sufficiently reliable match. Rephrase the "
                "question and specify the major or admission topic."
            )
            return {
                "found": False,
                "answer": message,
                "snippets": [],
                "sources": [],
                "scores": [top["score"]],
                "confidence": "low",
                "intent": intent,
                "entities": entities.__dict__,
                "context": updated_context,
            }

        raw_text = str(top["text"])
        # Structured rows use "field: value | field: value". Present the
        # meaningful values without dumping database column names.
        values = []
        for part in raw_text.split(" | "):
            value = part.split(": ", 1)[1] if ": " in part else part
            value = value.strip()
            if value and value not in values:
                values.append(value)

        clean_text = " ".join(values[:4]).strip()
        answer = (
            f"**{top['title']}**\n\n{clean_text}"
            if clean_text
            else str(top["title"])
        )

        return {
            "found": True,
            "answer": answer,
            "snippets": [clean_text],
            "sources": [top["source"]],
            "scores": [top["score"]],
            "confidence": "medium",
            "intent": intent,
            "entities": entities.__dict__,
            "context": updated_context,
        }
