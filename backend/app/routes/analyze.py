"""
POST /analyze — main detection endpoint.
Accepts text, image upload, or URL.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

from app.services.ocr_service import extract_text_from_upload
from app.services.behavioral_detector import analyze_behavior_dict
from app.services.domain_analyzer import analyze_domain_dict
from app.services.company_validator import validate_company_dict
from app.services.scoring_engine import score_from_dicts

router = APIRouter()


@router.post("/analyze")
async def analyze(
    text:         Optional[str]        = Form(None),
    url:          Optional[str]        = Form(None),
    company_name: Optional[str]        = Form(None),
    image:        Optional[UploadFile] = File(None),
):
    """
    Analyze a job posting for scam indicators.

    Inputs (at least one required):
      - text:         raw job description text
      - url:          job posting URL
      - company_name: company name from the posting
      - image:        screenshot/image of the job posting

    Returns:
      - final_score:  0.0–1.0 scam probability
      - risk_level:   LOW / MEDIUM / HIGH / CRITICAL
      - explanation:  human-readable summary
      - breakdown:    per-module scores
      - flags:        specific red flags found
      - ocr_text:     extracted text (if image was provided)
    """
    job_text = text or ""
    ocr_text = ""

    # --- OCR: extract text from image ---
    if image:
        file_bytes = await image.read()
        ocr_result = extract_text_from_upload(file_bytes, image.filename or "")
        if not ocr_result["success"]:
            raise HTTPException(status_code=422, detail=f"OCR failed: {ocr_result['error']}")
        ocr_text = ocr_result["text"]
        job_text = (job_text + " " + ocr_text).strip()

    if not job_text and not url:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of: text, image, or url."
        )

    # --- NLP scoring ---
    nlp_result = None
    from app.main import get_scorer
    scorer = get_scorer()
    if scorer and job_text:
        try:
            nlp_result = scorer.score(job_text)
        except Exception:
            nlp_result = None

    # --- Behavioral detection ---
    behavioral_result = analyze_behavior_dict(job_text) if job_text else None

    # --- Domain analysis (skip if URL is clearly invalid) ---
    domain_result = None
    if url and url.strip().startswith("http"):
        try:
            domain_result = analyze_domain_dict(url.strip())
        except Exception:
            domain_result = None

    # --- Company validation ---
    company_result = None
    if company_name and company_name.strip():
        try:
            company_result = validate_company_dict(company_name.strip(), job_text)
        except Exception:
            company_result = None

    # --- Final scoring ---
    final = score_from_dicts(
        nlp_result=nlp_result,
        behavioral_result=behavioral_result,
        domain_result=domain_result,
        company_result=company_result,
    )

    return JSONResponse({
        **final,
        "ocr_text":          ocr_text or None,
        "modules_used":      [
            m for m, v in {
                "nlp":       nlp_result,
                "behavioral":behavioral_result,
                "domain":    domain_result,
                "company":   company_result,
            }.items() if v is not None
        ],
    })
