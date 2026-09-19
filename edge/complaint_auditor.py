"""Ollama-Powered Customer Complaint Classifier.

Analyzes customer transcripts (Khmer + English) using the local Ollama LLM
to reliably identify customer grievances, complaints, and dissatisfaction.
Outputs validated, structured JSON metrics.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger("complaint_auditor")

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

COMPLAINT_KEYWORDS_EN = list(map(str.lower, [
    "not good", "bad", "broken", "fix it again", "terrible", "wrong",
    "not satisfied", "expensive", "too long", "wait too long", "waste",
    "angry", "dirty", "damaged", "scratched", "leaking", "noise still",
    "not clean", "unclean", "not yet cleaned", "still dirty",
]))

COMPLAINT_KEYWORDS_KM = [
    "មិនពេញចិត្ត", "ខូច", "មិនល្អ", "ថ្លៃ", "យូរ", "ខឹង",
    "បាត់", "កោស", "លេច", "មិនស្អាត", "ខ្វក់", "កខ្វក់",
]


def _has_khmer_script(text: str) -> bool:
    """Return True if text contains at least one Khmer Unicode character."""
    return any("\u1780" <= ch <= "\u17FF" for ch in text)


def _khmer_coverage(text: str) -> float:
    """Fraction of characters that belong to the Khmer Unicode block."""
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalnum()]
    if not letters:
        return 0.0
    khmer = [ch for ch in letters if "\u1780" <= ch <= "\u17FF"]
    return len(khmer) / len(letters)


def _latin_coverage(text: str) -> float:
    """Fraction of characters that belong to Latin/English script.

    Used to let English-speaking (foreign) customers through the same
    complaint gate as Khmer speakers, so their complaints are NOT dropped.
    """
    if not text:
        return 0.0
    letters = [ch for ch in text if ch.isalnum()]
    if not letters:
        return 0.0
    latin = [ch for ch in letters if ("A" <= ch <= "Z") or ("a" <= ch <= "z")]
    return len(latin) / len(letters)


@dataclasses.dataclass
class ComplaintAnalysis:
    """Structured complaint analysis result."""
    is_complaint: bool
    category: str
    severity: str
    summary: str
    raw_response: str = ""
    english_translation: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_complaint": self.is_complaint,
            "category": self.category,
            "severity": self.severity,
            "summary": self.summary,
            "english_translation": self.english_translation,
        }


class OllamaComplaintAuditor:
    """Audits customer speech to determine if it represents a complaint."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        timeout: float = 180.0,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout

    def analyze(self, khmer_text: str, english_text: str) -> ComplaintAnalysis:
        """Analyze customer dialogue to classify complaints."""
        if not khmer_text and not english_text:
            return ComplaintAnalysis(
                is_complaint=False,
                category="none",
                severity="none",
                summary="Empty transcript.",
            )

        prompt = f"""You are a Quality Assurance Auditor for an automotive service garage and store.
Analyze the following customer statement (with both Khmer and English transcripts) and decide whether the customer is expressing a genuine COMPLAINT or DISSATISFACTION.

RULES:
1. Routine customer inquiries (e.g. asking about prices, asking about duration/wait time for a job, requesting a new service, polite greetings, small talk, thanks) are NOT complaints. Set "is_complaint": false, "category": "inquiry", "severity": "none".
2. Mark as a complaint ("is_complaint": true) ONLY if the customer is clearly expressing dissatisfaction: reporting poor/incorrect work, stating the car is still dirty ("ឡាន់នៅតែក៏ខ្វក់", "car is still dirty", "not cleaned properly", "uncleaned"), damaged vehicle during service, rude staff, unresolved repeat issue, or overcharging.
3. When uncertain, prefer "is_complaint": false. Only rate a complaint "high" or "critical" for serious grievances (vehicle damage, threats, yelling, rework). Mild frustration is "low" or "medium".
4. If the transcript is gibberish, unrelated words, or is not Khmer/English speech, set "is_complaint": false, "category": "inquiry", "severity": "none", summary: "Unintelligible or non-speech audio."

Customer Statement:
- Khmer: {khmer_text}
- English: {english_text}

Respond ONLY with valid JSON conforming exactly to this structure:
{{
  "is_complaint": true or false,
  "category": "cleanliness | repair_quality | pricing | wait_time | staff_behavior | vehicle_damage | inquiry | other",
  "severity": "none | low | medium | high | critical",
  "summary": "Concise 1-sentence explanation."
}}
"""

        try:
            resp = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 256,
                    },
                },
                timeout=self.timeout,
            )

            if resp.status_code != 200:
                logger.error(f"[ComplaintAuditor] Ollama error HTTP {resp.status_code}: {resp.text}")
                return self._fallback_rule_based(khmer_text, english_text)

            data = resp.json()
            raw_text = data.get("response", "").strip()

            parsed = self._extract_json(raw_text)
            if parsed:
                is_comp = bool(parsed.get("is_complaint", False))
                cat = str(parsed.get("category") or "other").strip().lower()
                sev = str(parsed.get("severity") or "medium").strip().lower()
                summary = str(parsed.get("summary") or english_text or khmer_text).strip()

                logger.info(f"[ComplaintAuditor] Verdict: is_complaint={is_comp}, cat={cat}, sev={sev}, summary='{summary}'")
                return ComplaintAnalysis(
                    is_complaint=is_comp,
                    category=cat,
                    severity=sev,
                    summary=summary,
                    raw_response=raw_text,
                )

        except Exception as exc:
            logger.error(f"[ComplaintAuditor] Request failed: {exc}")

        return self._fallback_rule_based(khmer_text, english_text)

    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON from LLM output."""
        try:
            return json.loads(text)
        except Exception:
            pass

        # Try regex extract
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return None

    def _fallback_rule_based(self, khmer_text: str, english_text: str) -> ComplaintAnalysis:
        """Fallback keyword scanner in case Ollama server is briefly unreachable."""
        combined = f"{khmer_text} {english_text}".lower()

        # Only trust Khmer-scripted text for the keyword path; garbage will be
        # rejected earlier in the pipeline, but this is a belt-and-suspenders guard.
        kw_khmer = [k for k in COMPLAINT_KEYWORDS_KM if k in khmer_text]
        kw_english = [k for k in COMPLAINT_KEYWORDS_EN if k in english_text.lower()]
        if not kw_khmer and not kw_english:
            if not _has_khmer_script(khmer_text):
                return ComplaintAnalysis(
                    is_complaint=False,
                    category="inquiry",
                    severity="none",
                    summary="Unintelligible or non-speech audio.",
                )
            # Khmer speech but no complaint keywords -> normal dialogue
            return ComplaintAnalysis(
                is_complaint=False,
                category="inquiry",
                severity="none",
                summary="Normal customer dialogue.",
            )

        cat = "cleanliness" if any(k in combined for k in (
            "មិនស្អាត", "ខ្វក់", "កខ្វក់", "clean", "dirty", "unclean"
        )) else "repair_quality"
        return ComplaintAnalysis(
            is_complaint=True,
            category=cat,
            severity="medium",
            summary=f"Customer complaint ({cat}): {english_text or khmer_text}",
        )
