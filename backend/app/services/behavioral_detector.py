"""
Behavioral Detection Module
Rule-based scam pattern detection for job descriptions.
Returns a behavioral risk score 0.0 (clean) → 1.0 (highly suspicious).
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Rule definitions
# Each rule has: name, patterns, weight, description
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    name:        str
    patterns:    list[str]
    weight:      float          # contribution to final score
    description: str
    matched:     list[str] = field(default_factory=list)

    def compile(self) -> "CompiledRule":
        return CompiledRule(
            name=self.name,
            regexes=[re.compile(p, re.IGNORECASE) for p in self.patterns],
            weight=self.weight,
            description=self.description,
        )


@dataclass
class CompiledRule:
    name:        str
    regexes:     list[re.Pattern]
    weight:      float
    description: str

    def check(self, text: str) -> list[str]:
        """Return list of matched snippets (empty = no match)."""
        found = []
        for rx in self.regexes:
            for m in rx.finditer(text):
                snippet = text[max(0, m.start()-20): m.end()+20].strip()
                found.append(snippet)
        return found


# ---------------------------------------------------------------------------
# Rule registry
# ---------------------------------------------------------------------------

RULES: list[Rule] = [

    # --- Payment / financial requests ---
    Rule(
        name="payment_request",
        weight=0.40,
        description="Asks candidate to pay money upfront",
        patterns=[
            r"pay(ment)?\s+(a\s+)?(fee|deposit|registration|training|kit|starter)",
            r"(registration|processing|admin|application)\s+fee",
            r"send\s+(money|cash|\$|funds)",
            r"(wire|transfer)\s+(money|funds|payment)",
            r"(refundable|non-?refundable)\s+deposit",
            r"purchase\s+(your\s+)?(kit|equipment|laptop|materials)",
            r"buy\s+(your\s+)?(starter|training|work)\s+kit",
        ],
    ),

    # --- Unrealistic salary / income promises ---
    Rule(
        name="unrealistic_salary",
        weight=0.30,
        description="Promises unrealistically high or vague income",
        patterns=[
            r"\$\s*\d{3,4}\s*(per\s+)?(day|hour|hr)",          # $500/day, $300/hr
            r"\$\s*[5-9]\d{3,}|\$\s*[1-9]\d{4,}\s*(per\s+)?(week|month)",  # $5000+/week
            r"earn\s+(up\s+to\s+)?\$\s*\d+[,\d]*\s*(per\s+)?(week|day|hour)",
            r"(guaranteed|assured)\s+(income|salary|earnings|pay)",
            r"unlimited\s+(earning|income|potential)",
            r"make\s+(money|cash)\s+(fast|quick|easy|online)",
            r"(passive|residual)\s+income",
        ],
    ),

    # --- Urgency language ---
    Rule(
        name="urgency_language",
        weight=0.20,
        description="Creates artificial urgency to pressure candidates",
        patterns=[
            r"(urgent(ly)?|immediate(ly)?)\s+(hiring|opening|vacancy|position|start)",
            r"(apply|respond|contact)\s+(now|today|immediately|asap|urgently)",
            r"(limited|few|only\s+\d+)\s+(spots?|positions?|openings?)\s+(available|left|remaining)",
            r"(offer\s+)?(expires?|closing|ends?)\s+(today|tonight|soon|in\s+\d+\s+(hours?|days?))",
            r"don'?t\s+(miss|delay|wait)",
            r"(last|final)\s+(chance|opportunity|call)",
            r"positions?\s+(fill(ing)?\s+)?(fast|quickly|rapidly)",
        ],
    ),

    # --- Messaging app contact (WhatsApp / Telegram) ---
    Rule(
        name="messaging_app_contact",
        weight=0.25,
        description="Directs candidates to WhatsApp or Telegram instead of official channels",
        patterns=[
            r"(contact|reach|message|text|whatsapp|chat)\s+(us|me|hr|recruiter)?\s*(on|via|at|through)?\s*whatsapp",
            r"whatsapp\s*(number|no\.?|:)?\s*\+?\d[\d\s\-]{7,}",
            r"(contact|reach|message|join)\s+(us|me)?\s*(on|via|at|through)?\s*telegram",
            r"telegram\s*(channel|group|handle|:)?\s*@?\w+",
            r"(send|forward)\s+(cv|resume|application)\s+(to|via|on)\s*(whatsapp|telegram|wechat)",
            r"(wechat|viber|signal)\s*(id|number|contact)",
        ],
    ),

    # --- No interview / instant hiring ---
    Rule(
        name="no_interview",
        weight=0.20,
        description="Skips standard hiring process — no interview required",
        patterns=[
            r"no\s+(interview|screening|test|assessment|background\s+check)",
            r"(hired|selected|confirmed)\s+(immediately|instantly|on\s+the\s+spot|same\s+day)",
            r"(skip|bypass)\s+(the\s+)?(interview|hiring\s+process)",
            r"(start|begin)\s+(work|working|immediately)\s+(without|no)\s+interview",
            r"you('re|\s+are)\s+(already\s+)?(selected|hired|approved|chosen)",
            r"(no\s+)?(experience|qualification|skill)s?\s+(required|needed|necessary)",
        ],
    ),

    # --- Personal / financial info requests ---
    Rule(
        name="personal_info_request",
        weight=0.35,
        description="Requests sensitive personal or financial information upfront",
        patterns=[
            r"(send|provide|submit|share)\s+(your\s+)?(ssn|social\s+security|national\s+id|passport)",
            r"(bank\s+(account|details|info|number)|routing\s+number|account\s+number)",
            r"(credit|debit)\s+card\s+(number|details|info)",
            r"(date\s+of\s+birth|dob|mother'?s?\s+maiden\s+name)",
            r"(copy|scan|photo)\s+of\s+(your\s+)?(id|passport|license|ssn)",
        ],
    ),

    # --- Vague company identity ---
    Rule(
        name="vague_company",
        weight=0.15,
        description="Company identity is deliberately vague or anonymous",
        patterns=[
            r"(confidential|anonymous)\s+(company|employer|client|organization)",
            r"company\s+name\s+(will\s+be\s+)?(disclosed|revealed|shared)\s+(later|after|upon)",
            r"(leading|top|reputed|well-known)\s+company\s+(in\s+\w+\s+)?(industry|sector|field)",
            r"our\s+client\s+(is\s+)?(a\s+)?(leading|top|major|global)",
        ],
    ),

    # --- Work from home bait ---
    Rule(
        name="wfh_bait",
        weight=0.15,
        description="Uses work-from-home as primary selling point with suspicious framing",
        patterns=[
            r"(100%|fully|completely|entirely)\s+remote\s+(work|job|position|opportunity)",
            r"work\s+from\s+(home|anywhere)\s+(and\s+)?(earn|make|get\s+paid)",
            r"(stay\s+at\s+home|housewife|student|retired)\s+(can\s+)?(apply|earn|work)",
            r"(laptop|phone|internet)\s+(is\s+)?(all\s+you\s+need)",
        ],
    ),

    # --- Spelling / grammar red flags (common in scam posts) ---
    Rule(
        name="grammar_red_flags",
        weight=0.10,
        description="Patterns common in poorly written scam postings",
        patterns=[
            r"\bkindly\s+(send|forward|provide|contact|revert)\b",
            r"\brevert\s+back\b",
            r"\bdo\s+the\s+needful\b",
            r"\brespected\s+(sir|madam|candidate)\b",
            r"\bdear\s+(candidate|applicant|job\s+seeker)\b",
        ],
    ),
]

# Pre-compile all rules once at import time
_COMPILED_RULES: list[CompiledRule] = [r.compile() for r in RULES]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BehavioralAnalysisResult:
    text_snippet:   str                         # first 120 chars for reference
    triggered_rules: list[dict]                 # [{name, weight, matches}]
    raw_score:      float                       # sum of triggered weights (uncapped)
    risk_score:     float                       # normalized 0–1
    risk_level:     str                         # LOW / MEDIUM / HIGH / CRITICAL
    summary:        list[str]                   # human-readable findings


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _risk_level(score: float) -> str:
    if score < 0.20:
        return "LOW"
    elif score < 0.45:
        return "MEDIUM"
    elif score < 0.70:
        return "HIGH"
    else:
        return "CRITICAL"


def _normalize(raw: float, max_possible: float) -> float:
    """Sigmoid-like normalization so score never hard-clips at 1.0 artificially."""
    if max_possible == 0:
        return 0.0
    ratio = raw / max_possible
    # Soft cap: 0.95 ceiling even if all rules fire
    return round(min(ratio * 0.95, 0.99), 4)


MAX_POSSIBLE_SCORE = sum(r.weight for r in RULES)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_behavior(text: str) -> BehavioralAnalysisResult:
    """
    Run all rules against the job description text.

    Args:
        text: Raw or cleaned job description string

    Returns:
        BehavioralAnalysisResult with risk_score 0.0–1.0
    """
    text_lower = text.lower()
    triggered  = []
    raw_score  = 0.0
    summary    = []

    for rule in _COMPILED_RULES:
        matches = rule.check(text_lower)
        if matches:
            triggered.append({
                "rule":        rule.name,
                "weight":      rule.weight,
                "description": rule.description,
                "matches":     matches[:3],   # cap at 3 snippets per rule
            })
            raw_score += rule.weight
            summary.append(f"{rule.description} [{rule.name}]")

    risk_score = _normalize(raw_score, MAX_POSSIBLE_SCORE)

    return BehavioralAnalysisResult(
        text_snippet    = text[:120].strip(),
        triggered_rules = triggered,
        raw_score       = round(raw_score, 4),
        risk_score      = risk_score,
        risk_level      = _risk_level(risk_score),
        summary         = summary,
    )


def analyze_behavior_dict(text: str) -> dict:
    """Convenience wrapper returning a plain dict for API responses."""
    r = analyze_behavior(text)
    return {
        "risk_score":      r.risk_score,
        "risk_level":      r.risk_level,
        "triggered_rules": r.triggered_rules,
        "summary":         r.summary,
        "raw_score":       r.raw_score,
        "text_snippet":    r.text_snippet,
    }


def get_rule_names() -> list[str]:
    """List all available rule names — useful for debugging/admin."""
    return [r.name for r in RULES]
