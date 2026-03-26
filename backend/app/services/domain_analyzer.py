"""
Domain Intelligence Module
Analyzes a job posting URL for trust signals:
  - Domain age via WHOIS
  - SSL certificate validity
  - Suspicious domain patterns

Returns a normalized risk score 0.0 (safe) → 1.0 (suspicious).
"""

import re
import ssl
import socket
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Optional

import whois

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suspicious pattern lists
# ---------------------------------------------------------------------------

# TLDs commonly abused in job scams
SUSPICIOUS_TLDS = {
    ".xyz", ".top", ".click", ".work", ".jobs", ".tk",
    ".ml", ".ga", ".cf", ".gq", ".pw", ".cc", ".biz",
}

# Legitimate job board domains — low risk baseline
TRUSTED_DOMAINS = {
    "linkedin.com", "indeed.com", "glassdoor.com", "monster.com",
    "ziprecruiter.com", "careerbuilder.com", "dice.com", "lever.co",
    "greenhouse.io", "workday.com", "bamboohr.com", "jobvite.com",
    "smartrecruiters.com", "icims.com", "taleo.net", "myworkdayjobs.com",
}

# Regex patterns that signal typosquatting or scam domains
SUSPICIOUS_PATTERNS = [
    r"jobs?\d{3,}",           # jobs123, job456
    r"career[s]?-\w+",        # careers-apply, career-now
    r"(apply|hiring)-now",    # apply-now, hiring-now
    r"work-?from-?home",      # work-from-home
    r"earn-?\d+",             # earn5000, earn-500
    r"\d{4,}",                # 4+ consecutive digits in domain
    r"(free|easy|fast)-?job", # free-job, easyjob
    r"(legit|real|genuine)",  # overclaiming legitimacy
]

_SUSPICIOUS_RE = [re.compile(p, re.IGNORECASE) for p in SUSPICIOUS_PATTERNS]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DomainAnalysisResult:
    url:              str
    domain:           str

    # WHOIS
    domain_age_days:  Optional[int]   = None
    whois_available:  bool            = False
    registrar:        Optional[str]   = None

    # SSL
    ssl_valid:        bool            = False
    ssl_expiry_days:  Optional[int]   = None
    ssl_error:        Optional[str]   = None

    # Pattern checks
    is_trusted_domain:    bool        = False
    suspicious_tld:       bool        = False
    suspicious_patterns:  list[str]   = field(default_factory=list)

    # Final score
    risk_score:       float           = 0.0
    risk_level:       str             = "UNKNOWN"
    risk_factors:     list[str]       = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_domain(url: str) -> str:
    """Extract root domain from any URL string."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path
    # Strip www.
    return re.sub(r"^www\.", "", host).lower()


def _get_domain_age(domain: str) -> tuple[Optional[int], bool, Optional[str]]:
    """
    Query WHOIS and return (age_in_days, available, registrar).
    Returns (None, False, None) on failure.
    """
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation is None:
            return None, True, w.registrar

        # Normalize to UTC-aware datetime
        if creation.tzinfo is None:
            creation = creation.replace(tzinfo=timezone.utc)

        age_days = (datetime.now(timezone.utc) - creation).days
        return age_days, True, w.registrar

    except Exception as e:
        logger.debug(f"WHOIS failed for {domain}: {e}")
        return None, False, None


def _check_ssl(domain: str) -> tuple[bool, Optional[int], Optional[str]]:
    """
    Verify SSL certificate.
    Returns (valid, days_until_expiry, error_message).
    """
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=5),
            server_hostname=domain,
        ) as s:
            cert = s.getpeercert()

        expiry_str = cert.get("notAfter", "")
        expiry_dt  = datetime.strptime(expiry_str, "%b %d %H:%M:%S %Y %Z")
        expiry_dt  = expiry_dt.replace(tzinfo=timezone.utc)
        days_left  = (expiry_dt - datetime.now(timezone.utc)).days

        return True, days_left, None

    except ssl.SSLCertVerificationError as e:
        return False, None, f"cert_invalid: {str(e)[:80]}"
    except ssl.SSLError as e:
        return False, None, f"ssl_error: {str(e)[:80]}"
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        return False, None, f"connection_error: {str(e)[:80]}"
    except Exception as e:
        return False, None, f"unknown: {str(e)[:80]}"


def _check_patterns(domain: str) -> list[str]:
    """Return list of suspicious pattern names matched."""
    matched = []
    for pat in _SUSPICIOUS_RE:
        if pat.search(domain):
            matched.append(pat.pattern)
    return matched


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

def _compute_risk_score(result: DomainAnalysisResult) -> tuple[float, list[str]]:
    """
    Weighted scoring across all signals.
    Each factor contributes a penalty; final score clamped to [0, 1].
    """
    score   = 0.0
    factors = []

    # Trusted domain → strong negative signal (low risk)
    if result.is_trusted_domain:
        return 0.05, ["trusted_job_board"]

    # SSL
    if not result.ssl_valid:
        score += 0.25
        factors.append(f"no_valid_ssl ({result.ssl_error})")
    elif result.ssl_expiry_days is not None and result.ssl_expiry_days < 14:
        score += 0.10
        factors.append(f"ssl_expiring_soon ({result.ssl_expiry_days}d)")

    # Domain age
    if not result.whois_available:
        score += 0.15
        factors.append("whois_unavailable")
    elif result.domain_age_days is None:
        score += 0.20
        factors.append("domain_age_unknown")
    elif result.domain_age_days < 30:
        score += 0.35
        factors.append(f"very_new_domain ({result.domain_age_days}d)")
    elif result.domain_age_days < 180:
        score += 0.20
        factors.append(f"new_domain ({result.domain_age_days}d)")
    elif result.domain_age_days < 365:
        score += 0.10
        factors.append(f"young_domain ({result.domain_age_days}d)")

    # Suspicious TLD
    if result.suspicious_tld:
        score += 0.20
        factors.append(f"suspicious_tld")

    # Pattern matches
    if result.suspicious_patterns:
        penalty = min(0.10 * len(result.suspicious_patterns), 0.30)
        score  += penalty
        factors.append(f"suspicious_patterns ({len(result.suspicious_patterns)} matched)")

    return round(min(score, 1.0), 4), factors


def _risk_level(score: float) -> str:
    if score < 0.25:
        return "LOW"
    elif score < 0.55:
        return "MEDIUM"
    else:
        return "HIGH"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_domain(url: str) -> DomainAnalysisResult:
    """
    Full domain analysis pipeline.

    Args:
        url: Job posting URL (with or without scheme)

    Returns:
        DomainAnalysisResult with risk_score 0.0–1.0
    """
    domain = _extract_domain(url)
    result = DomainAnalysisResult(url=url, domain=domain)

    # Trusted domain short-circuit
    root = ".".join(domain.split(".")[-2:])   # e.g. linkedin.com
    if root in TRUSTED_DOMAINS or domain in TRUSTED_DOMAINS:
        result.is_trusted_domain = True
        result.risk_score  = 0.05
        result.risk_level  = "LOW"
        result.risk_factors = ["trusted_job_board"]
        return result

    # TLD check
    tld = "." + domain.rsplit(".", 1)[-1]
    result.suspicious_tld = tld in SUSPICIOUS_TLDS

    # Pattern check
    result.suspicious_patterns = _check_patterns(domain)

    # WHOIS
    result.domain_age_days, result.whois_available, result.registrar = _get_domain_age(domain)

    # SSL
    result.ssl_valid, result.ssl_expiry_days, result.ssl_error = _check_ssl(domain)

    # Score
    result.risk_score, result.risk_factors = _compute_risk_score(result)
    result.risk_level = _risk_level(result.risk_score)

    return result


def analyze_domain_dict(url: str) -> dict:
    """Convenience wrapper — returns a plain dict for API responses."""
    r = analyze_domain(url)
    return {
        "url":              r.url,
        "domain":           r.domain,
        "risk_score":       r.risk_score,
        "risk_level":       r.risk_level,
        "risk_factors":     r.risk_factors,
        "domain_age_days":  r.domain_age_days,
        "ssl_valid":        r.ssl_valid,
        "ssl_expiry_days":  r.ssl_expiry_days,
        "is_trusted_domain": r.is_trusted_domain,
        "suspicious_tld":   r.suspicious_tld,
        "suspicious_patterns": r.suspicious_patterns,
        "registrar":        r.registrar,
    }
