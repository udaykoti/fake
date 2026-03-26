"""
Scoring Engine — combines all module outputs into a final scam probability.
Weights are tunable; explanation is human-readable.
"""

from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------
WEIGHTS = {
    "nlp":       0.35,   # text-based ML model — strongest signal
    "behavioral":0.30,   # rule-based pattern detection
    "domain":    0.20,   # domain age, SSL, suspicious TLD
    "company":   0.15,   # company validation (trust inverted to risk)
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


@dataclass
class ScoringInput:
    nlp_risk:       Optional[float] = None   # 0–1 from NLP model
    behavioral_risk:Optional[float] = None   # 0–1 from behavioral detector
    domain_risk:    Optional[float] = None   # 0–1 from domain analyzer
    company_trust:  Optional[float] = None   # 0–1 trust → converted to risk internally

    # Raw detail dicts from each module (for explanation)
    nlp_detail:       dict = field(default_factory=dict)
    behavioral_detail:dict = field(default_factory=dict)
    domain_detail:    dict = field(default_factory=dict)
    company_detail:   dict = field(default_factory=dict)


@dataclass
class ScoringResult:
    final_score:  float
    risk_level:   str
    explanation:  str
    breakdown:    dict
    flags:        list[str]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def compute_final_score(inp: ScoringInput) -> ScoringResult:
    """
    Weighted combination of all module scores.
    Missing modules are excluded and remaining weights renormalized.
    """
    scores = {
        "nlp":        inp.nlp_risk,
        "behavioral": inp.behavioral_risk,
        "domain":     inp.domain_risk,
        # company returns trust (high = good), invert to risk
        "company":    (1.0 - inp.company_trust) if inp.company_trust is not None else None,
    }

    # Filter out missing modules and renormalize weights
    available = {k: v for k, v in scores.items() if v is not None}
    if not available:
        return ScoringResult(0.0, "UNKNOWN", "No analysis data available.", {}, [])

    total_weight = sum(WEIGHTS[k] for k in available)
    weighted_sum = sum(WEIGHTS[k] * v for k, v in available.items())
    final_score  = round(weighted_sum / total_weight, 4)

    risk_level  = _risk_level(final_score)
    flags       = _collect_flags(inp)
    explanation = _generate_explanation(final_score, risk_level, inp, flags)

    breakdown = {k: round(v, 4) for k, v in available.items()}
    breakdown["final"] = final_score

    return ScoringResult(
        final_score=final_score,
        risk_level=risk_level,
        explanation=explanation,
        breakdown=breakdown,
        flags=flags,
    )


def _risk_level(score: float) -> str:
    if score < 0.25:  return "LOW"
    if score < 0.50:  return "MEDIUM"
    if score < 0.75:  return "HIGH"
    return "CRITICAL"


def _collect_flags(inp: ScoringInput) -> list[str]:
    flags = []

    # NLP flags
    if inp.nlp_risk and inp.nlp_risk > 0.7:
        flags.append("ML model flagged description as highly suspicious")

    # Behavioral flags
    for rule in inp.behavioral_detail.get("triggered_rules", []):
        flags.append(rule["description"])

    # Domain flags
    for factor in inp.domain_detail.get("risk_factors", []):
        flags.append(f"Domain: {factor}")

    # Company flags
    for finding in inp.company_detail.get("findings", []):
        flags.append(f"Company: {finding}")

    return flags


def _generate_explanation(
    score: float,
    level: str,
    inp: ScoringInput,
    flags: list[str],
) -> str:
    lines = []

    # Opening verdict
    verdicts = {
        "LOW":      "This job posting appears legitimate.",
        "MEDIUM":   "This job posting has some suspicious characteristics worth reviewing.",
        "HIGH":     "This job posting shows multiple red flags consistent with scam activity.",
        "CRITICAL": "This job posting is highly likely to be fraudulent. Do not apply or share personal information.",
    }
    lines.append(verdicts[level])
    lines.append(f"Overall scam probability: {round(score * 100, 1)}%.")

    # Module summaries
    if inp.nlp_risk is not None:
        nlp_label = inp.nlp_detail.get("label", "")
        lines.append(
            f"Text analysis: The job description language is {'consistent with known scam patterns' if inp.nlp_risk > 0.5 else 'mostly normal'}."
            + (f" (model confidence: {round(inp.nlp_detail.get('confidence', 0)*100)}%)" if inp.nlp_detail.get('confidence') else "")
        )

    if inp.behavioral_risk is not None and inp.behavioral_risk > 0:
        triggered = inp.behavioral_detail.get("triggered_rules", [])
        if triggered:
            rule_names = ", ".join(r["rule"] for r in triggered[:3])
            lines.append(f"Behavioral patterns detected: {rule_names}.")

    if inp.domain_risk is not None:
        age = inp.domain_detail.get("domain_age_days")
        ssl = inp.domain_detail.get("ssl_valid")
        if age is not None and age < 180:
            lines.append(f"The job URL domain is only {age} days old, which is unusual for legitimate employers.")
        if ssl is False:
            lines.append("The job URL does not have a valid SSL certificate.")

    if inp.company_trust is not None and inp.company_trust < 0.4:
        lines.append(
            "The company name could not be verified in known employer databases"
            + (" and matches suspicious naming patterns." if inp.company_detail.get("name_is_suspicious") else ".")
        )

    # Specific flags
    payment = [f for f in flags if "pay" in f.lower() or "fee" in f.lower() or "deposit" in f.lower()]
    if payment:
        lines.append("WARNING: This posting appears to request payment from the applicant — a hallmark of job scams.")

    personal = [f for f in flags if "personal" in f.lower() or "bank" in f.lower() or "ssn" in f.lower()]
    if personal:
        lines.append("WARNING: This posting requests sensitive personal or financial information.")

    # Advice
    if level in ("HIGH", "CRITICAL"):
        lines.append("Recommendation: Do not provide personal information, pay any fees, or contact via unofficial channels.")
    elif level == "MEDIUM":
        lines.append("Recommendation: Research the company independently before proceeding.")

    return " ".join(lines)


def score_from_dicts(
    nlp_result:       Optional[dict] = None,
    behavioral_result:Optional[dict] = None,
    domain_result:    Optional[dict] = None,
    company_result:   Optional[dict] = None,
) -> dict:
    """Convenience function — accepts raw module output dicts, returns final result dict."""
    inp = ScoringInput(
        nlp_risk        = nlp_result.get("prob_fake") or nlp_result.get("risk_score") if nlp_result else None,
        behavioral_risk = behavioral_result.get("risk_score") if behavioral_result else None,
        domain_risk     = domain_result.get("risk_score") if domain_result else None,
        company_trust   = company_result.get("trust_score") if company_result else None,
        nlp_detail      = nlp_result or {},
        behavioral_detail = behavioral_result or {},
        domain_detail   = domain_result or {},
        company_detail  = company_result or {},
    )
    result = compute_final_score(inp)
    return {
        "final_score": result.final_score,
        "risk_level":  result.risk_level,
        "explanation": result.explanation,
        "breakdown":   result.breakdown,
        "flags":       result.flags,
    }
