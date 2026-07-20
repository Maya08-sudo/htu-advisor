from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from chatbot_config import ADMISSION_RULES

class RuleEngine:
    def __init__(self, path: Path = ADMISSION_RULES):
        self.data = json.loads(path.read_text(encoding="utf-8"))

    def tawjihi_rule(self, major: str) -> Optional[dict]:
        for rule in self.data.get("tawjihi", []):
            if major in rule.get("programs", []):
                return rule
        return None

    def requirements(self, major: str, language: str = "en") -> Optional[dict]:
        rule = self.tawjihi_rule(major)
        if not rule:
            return None
        if language == "ar":
            answer = (f"متطلبات التوجيهي لـ {major}: المعدل العام {rule['average_min']}%، "
                      f"الرياضيات {rule['math_min']}%، والفيزياء {rule['physics_min']}%. "
                      "يجب تحقيق الشروط الثلاثة.")
        else:
            answer = (f"Tawjihi requirements for {major}: overall average {rule['average_min']}%, "
                      f"mathematics {rule['math_min']}%, and physics {rule['physics_min']}%. "
                      "All three conditions must be met.")
        return {"answer": answer, "rule": rule}

    def check(self, major: str, average: float, math: float, physics: float, language: str = "en") -> Optional[dict]:
        rule = self.tawjihi_rule(major)
        if not rule:
            return None
        failures = []
        if average < rule["average_min"]: failures.append(("overall average", rule["average_min"], average))
        if math < rule["math_min"]: failures.append(("mathematics", rule["math_min"], math))
        if physics < rule["physics_min"]: failures.append(("physics", rule["physics_min"], physics))
        eligible = not failures
        if language == "ar":
            if eligible:
                answer = f"نعم، حسب القيم المدخلة أنت تحقق الشروط الأكاديمية الأولية لـ {major}. هذا لا يضمن القبول النهائي التنافسي."
            else:
                details = "، ".join(f"{name}: المطلوب {req} وأدخلت {got}" for name, req, got in failures)
                answer = f"لا تحقق جميع الشروط الأولية لـ {major}. {details}."
        else:
            if eligible:
                answer = f"Yes. Based on the values provided, you meet the initial academic requirements for {major}. This does not guarantee final competitive admission."
            else:
                details = "; ".join(f"{name}: required {req}, provided {got}" for name, req, got in failures)
                answer = f"You do not meet every initial requirement for {major}. {details}."
        return {"eligible": eligible, "answer": answer, "failures": failures, "rule": rule}
