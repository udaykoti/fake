"""
Company Validation Module
Checks company legitimacy and consistency with job description.
Returns a trust score 0.0 (untrustworthy) → 1.0 (trustworthy).
"""

import re
import json
import os
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known companies dataset (mock + extensible via JSON file)
# ---------------------------------------------------------------------------

# Curated list of well-known legitimate employers
# In production: replace/extend with a real dataset (OpenCorporates API, etc.)
_KNOWN_COMPANIES: dict[str, dict] = {
    # Tech
    "google":        {"industry": "technology", "size": "large",  "hq": "usa"},
    "microsoft":     {"industry": "technology", "size": "large",  "hq": "usa"},
    "amazon":        {"industry": "technology", "size": "large",  "hq": "usa"},
    "meta":          {"industry": "technology", "size": "large",  "hq": "usa"},
    "apple":         {"industry": "technology", "size": "large",  "hq": "usa"},
    "netflix":       {"industry": "technology", "size": "large",  "hq": "usa"},
    "salesforce":    {"industry": "technology", "size": "large",  "hq": "usa"},
    "ibm":           {"industry": "technology", "size": "large",  "hq": "usa"},
    "oracle":        {"industry": "technology", "size": "large",  "hq": "usa"},
    "adobe":         {"industry": "technology", "size": "large",  "hq": "usa"},
    "intel":         {"industry": "technology", "size": "large",  "hq": "usa"},
    "nvidia":        {"industry": "technology", "size": "large",  "hq": "usa"},
    # Finance
    "jpmorgan":      {"industry": "finance",    "size": "large",  "hq": "usa"},
    "goldman sachs": {"industry": "finance",    "size": "large",  "hq": "usa"},
    "morgan stanley":{"industry": "finance",    "size": "large",  "hq": "usa"},
    "citibank":      {"industry": "finance",    "size": "large",  "hq": "usa"},
    "bank of america":{"industry":"finance",    "size": "large",  "hq": "usa"},
    # Consulting
    "deloitte":      {"industry": "consulting", "size": "large",  "hq": "usa"},
    "mckinsey":      {"industry": "consulting", "size": "large",  "hq": "usa"},
    "accenture":     {"industry": "consulting", "size": "large",  "hq": "usa"},
    "pwc":           {"industry": "consulting", "size": "large",  "hq": "usa"},
    "kpmg":          {"industry": "consulting", "size": "large",  "hq": "usa"},
    # Healthcare
    "johnson & johnson": {"industry": "healthcare", "size": "large", "hq": "usa"},
    "pfizer":        {"industry": "healthcare", "size": "large",  "hq": "usa"},
    "unitedhealth":  {"industry": "healthcare", "size": "large",  "hq": "usa"},
    # Retail / Other
    "walmart":       {"industry": "retail",     "size": "large",  "hq": "usa"},
    "tesla":         {"industry": "automotive", "size": "large",  "hq": "usa"},
    "boeing":        {"industry": "aerospace",  "size": "large",  "hq": "usa"},
}

# Load extended list from JSON if available
_EXTENDED_DB_PATH = os.path.join(os.path.dirname(__file__), "../data/known_companies.json")

def _load_extended_db() -> dict:
    if os.path.isfile(_EXTENDED_DB_PATH):
        try:
            with open(_EXTENDED_DB_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load extended company DB: {e}")
    return {}

_KNOWN_COMPANIES.update(_load_extended_db())


# ---------------------------------------------------------------------------
# Suspicious company name patterns
# ---------------------------------------------------------------------------

_SUSPICIOUS_NAME_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"(global|international|worldwide)\s+(solutions?|opportunities?|ventures?|group)\s*(llc|inc|ltd)?$",
        r"(elite|premium|top|best|leading)\s+(staffing|recruitment|hr|jobs?)\s*(llc|inc|ltd)?$",
        r"(easy|quick|fast)\s+(jobs?|work|earn|money)",
        r"^\s*n/?a\s*$",                        # "N/A" as company name
        r"^\s*confidential\s*$",
        r"^\s*anonymous\s*$",
        r"\d{4,}",                              # lots of numbers in name
        r"(work\s+from\s+home|wfh)\s+",
    ]
]

# Industry keywords for consistency checking
_INDUSTRY_KEYWORDS: dict[str, list[str]] = {
    "technology":  ["software", "developer", "engineer", "data", "cloud", "devops",
                    "python", "java", "react", "machine learning", "ai", "cybersecurity"],
    "finance":     ["analyst", "banking", "investment", "trading", "accounting",
                    "financial", "audit", "compliance", "risk", "portfolio"],
    "healthcare":  ["nurse", "doctor", "clinical", "medical", "patient", "pharmacy",
                    "health", "hospital", "therapist", "diagnostic"],
    "consulting":  ["consultant", "strategy", "advisory", "management", "project",
                    "stakeholder", "deliverable", "engagement"],
    "retail":      ["sales", "store", "customer", "merchandise", "inventory",
                    "retail", "cashier", "buyer", "merchandising"],
    "automotive":  ["vehicle", "automotive", "manufacturing", "assembly", "mechanical",
                    "production", "quality control", "supply chain"],
    "aerospace":   ["aerospace", "aviation", "aircraft", "defense", "systems",
                    "engineering", "propulsion", "avionics"],
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompanyValidationResult:
    company_name:         str
    normalized_name:      str

    # Lookup
    found_in_db:          bool          = False
    db_record:            Optional[dict] = None

    # Name quality
    name_is_empty:        bool          = False
    name_is_suspicious:   bool          = False
    suspicious_pattern:   Optional[str] = None

    # Consistency
    industry_match:       Optional[bool] = None   # None = couldn't determine
    consistency_score:    float          = 0.5    # 0=inconsistent, 1=consistent

    # Final
    trust_score:          float          = 0.5
    trust_level:          str            = "UNKNOWN"
    findings:             list[str]      = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """Lowercase, strip legal suffixes and punctuation for matching."""
    name = name.lower().strip()
    name = re.sub(r"\b(inc|llc|ltd|corp|co|plc|gmbh|pvt|limited|incorporated)\.?\b", "", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return re.sub(r"\s+", " ", name).strip()


def _check_suspicious_name(name: str) -> Optional[str]:
    for rx in _SUSPICIOUS_NAME_PATTERNS:
        if rx.search(name):
            return rx.pattern
    return None


def _check_industry_consistency(
    industry: str,
    job_text: str,
) -> tuple[bool, float]:
    """
    Check if job description keywords match the company's known industry.
    Returns (matches: bool, score: float).
    """
    keywords = _INDUSTRY_KEYWORDS.get(industry, [])
    if not keywords:
        return True, 0.5   # unknown industry — neutral

    text_lower = job_text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    ratio = hits / len(keywords)

    # At least 2 keyword hits = consistent
    return hits >= 2, round(min(ratio * 3, 1.0), 4)


# ---------------------------------------------------------------------------
# Trust scoring
# ---------------------------------------------------------------------------

def _compute_trust(result: CompanyValidationResult) -> tuple[float, str]:
    score    = 0.5   # neutral baseline
    findings = result.findings

    # Empty name — strong negative
    if result.name_is_empty:
        return 0.05, "VERY_LOW"

    # Suspicious name pattern — strong negative
    if result.name_is_suspicious:
        score -= 0.30
        findings.append(f"suspicious_name_pattern")

    # Found in known DB — positive
    if result.found_in_db:
        score += 0.35
        findings.append("company_found_in_known_db")

        # Industry consistency bonus/penalty
        if result.industry_match is True:
            score += 0.15
            findings.append("job_description_matches_company_industry")
        elif result.industry_match is False:
            score -= 0.20
            findings.append("job_description_inconsistent_with_company_industry")
    else:
        # Not in DB — mild negative (could just be a smaller company)
        score -= 0.10
        findings.append("company_not_in_known_database")

    trust = round(max(0.01, min(score, 1.0)), 4)
    level = _trust_level(trust)
    return trust, level


def _trust_level(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    elif score >= 0.50:
        return "MEDIUM"
    elif score >= 0.25:
        return "LOW"
    else:
        return "VERY_LOW"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_company(company_name: str, job_description: str = "") -> CompanyValidationResult:
    """
    Validate a company name against the known database and check
    consistency with the job description.

    Args:
        company_name:    Raw company name from the job posting
        job_description: Full job description text (optional but improves accuracy)

    Returns:
        CompanyValidationResult with trust_score 0.0–1.0
    """
    result = CompanyValidationResult(
        company_name=company_name,
        normalized_name=_normalize_name(company_name),
    )

    # Empty name check
    if not company_name or not company_name.strip():
        result.name_is_empty = True
        result.trust_score   = 0.05
        result.trust_level   = "VERY_LOW"
        result.findings      = ["company_name_missing"]
        return result

    # Suspicious name pattern check
    pattern = _check_suspicious_name(company_name)
    if pattern:
        result.name_is_suspicious = True
        result.suspicious_pattern = pattern

    # DB lookup — try normalized name and partial matches
    norm = result.normalized_name
    db_record = _KNOWN_COMPANIES.get(norm)

    # Partial match: check if any known company name is contained in the input
    if not db_record:
        for known_name, record in _KNOWN_COMPANIES.items():
            if known_name in norm or norm in known_name:
                db_record = record
                break

    if db_record:
        result.found_in_db = True
        result.db_record   = db_record

        # Industry consistency check
        if job_description and db_record.get("industry"):
            match, cscore = _check_industry_consistency(
                db_record["industry"], job_description
            )
            result.industry_match    = match
            result.consistency_score = cscore

    # Compute final trust score
    result.trust_score, result.trust_level = _compute_trust(result)
    return result


def validate_company_dict(company_name: str, job_description: str = "") -> dict:
    """Convenience wrapper returning a plain dict for API responses."""
    r = validate_company(company_name, job_description)
    return {
        "company_name":      r.company_name,
        "normalized_name":   r.normalized_name,
        "trust_score":       r.trust_score,
        "trust_level":       r.trust_level,
        "found_in_db":       r.found_in_db,
        "name_is_suspicious": r.name_is_suspicious,
        "industry_match":    r.industry_match,
        "consistency_score": r.consistency_score,
        "findings":          r.findings,
        "db_record":         r.db_record,
    }
