"""
Phase 10 — 10 realistic job postings (mix of fake and real).
Tests the full pipeline without a trained model (behavioral + company + scoring).

Run:
    python -m pytest tests/test_pipeline.py -v
    # or without pytest:
    python tests/test_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.behavioral_detector import analyze_behavior_dict
from app.services.company_validator   import validate_company_dict
from app.services.scoring_engine      import score_from_dicts

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # ---- REAL postings ----
    {
        "id": 1, "expected": "real",
        "company": "Google",
        "text": (
            "Software Engineer, Google Cloud. We are looking for a software engineer "
            "with 3+ years of experience in distributed systems. You will design and "
            "build scalable backend services. Requirements: BS/MS in Computer Science, "
            "proficiency in Python or Go, experience with Kubernetes. Competitive salary, "
            "health benefits, 401k. Apply via careers.google.com."
        ),
    },
    {
        "id": 2, "expected": "real",
        "company": "Deloitte",
        "text": (
            "Senior Consultant – Financial Advisory. Deloitte is seeking a senior consultant "
            "to join our financial advisory practice. You will work with Fortune 500 clients "
            "on M&A transactions and restructuring. MBA or CFA required. 5+ years experience "
            "in investment banking or consulting. Travel up to 50%. Salary: $130,000–$160,000."
        ),
    },
    {
        "id": 3, "expected": "real",
        "company": "Pfizer",
        "text": (
            "Clinical Data Analyst – Pfizer Oncology. Responsible for managing and analyzing "
            "clinical trial data. Must have experience with SAS or R, knowledge of FDA "
            "regulations, and a degree in biostatistics or life sciences. Full-time, on-site "
            "in New York. Benefits include medical, dental, and stock options."
        ),
    },
    {
        "id": 4, "expected": "real",
        "company": "Amazon",
        "text": (
            "Operations Manager, Amazon Fulfillment Center. Lead a team of 50+ associates "
            "in a fast-paced warehouse environment. Drive process improvements, manage KPIs, "
            "and ensure safety compliance. 3+ years of operations or logistics management "
            "experience required. Bachelor's degree preferred. Salary: $75,000–$95,000."
        ),
    },
    {
        "id": 5, "expected": "real",
        "company": "Accenture",
        "text": (
            "Data Engineer – Accenture Technology. Design and implement data pipelines using "
            "Apache Spark and Azure Data Factory. Work with clients across banking and retail "
            "sectors. 2+ years of experience with SQL, Python, and cloud platforms. "
            "Hybrid work model. Competitive compensation and career development programs."
        ),
    },
    # ---- FAKE postings ----
    {
        "id": 6, "expected": "fake",
        "company": "Global Opportunities LLC",
        "text": (
            "URGENT HIRING! Work from home and earn $5,000 per week! No experience required. "
            "Data entry job — all you need is a laptop and internet. You are already selected "
            "for this position. No interview needed. Send your bank account details to receive "
            "your starter kit. Contact us on WhatsApp: +1-555-0199. Limited spots available, apply now!"
        ),
    },
    {
        "id": 7, "expected": "fake",
        "company": "",
        "text": (
            "Confidential company hiring immediately. Earn $300 per day doing simple online tasks. "
            "Housewives, students, and retired persons can apply. No qualifications needed. "
            "Guaranteed income every week. Registration fee of $50 required (refundable). "
            "Kindly send your resume and date of birth to our Telegram channel @easyjobs2024."
        ),
    },
    {
        "id": 8, "expected": "fake",
        "company": "International Staffing Solutions Inc",
        "text": (
            "Brand Ambassador needed urgently! Earn unlimited income working from home. "
            "This is your last chance — only 3 positions remaining. No experience necessary. "
            "You will be hired on the spot. Purchase your work kit for $99 to get started. "
            "Do not miss this opportunity. Revert back to us immediately with your SSN and passport copy."
        ),
    },
    {
        "id": 9, "expected": "fake",
        "company": "Elite Recruitment Group",
        "text": (
            "Make money fast! Passive income opportunity. Work from anywhere, earn $10,000/month. "
            "Our client is a leading company in the industry. Company name will be disclosed after "
            "you pay the processing fee of $75. No skills required. Positions fill quickly. "
            "Contact us via WhatsApp for immediate confirmation. You are already approved!"
        ),
    },
    {
        "id": 10, "expected": "fake",
        "company": "N/A",
        "text": (
            "Respected candidate, we have an urgent opening for a data entry operator. "
            "Salary: $500 per day, work from home. No interview, no experience required. "
            "Kindly do the needful and send your bank account number and routing number "
            "to receive your first payment. Dear applicant, this offer expires tonight. "
            "Easy job, fast money. Apply immediately."
        ),
    },
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_tests():
    print(f"\n{'='*70}")
    print(f"  FAKE JOB DETECTION — PIPELINE TEST ({len(TEST_CASES)} cases)")
    print(f"{'='*70}\n")

    correct = 0

    for case in TEST_CASES:
        behavioral = analyze_behavior_dict(case["text"])
        company    = validate_company_dict(case["company"], case["text"])
        result     = score_from_dicts(
            behavioral_result=behavioral,
            company_result=company,
        )

        score     = result["final_score"]
        level     = result["risk_level"]
        predicted = "fake" if score >= 0.45 else "real"
        match     = "✓" if predicted == case["expected"] else "✗"
        if predicted == case["expected"]:
            correct += 1

        print(f"[{match}] Case {case['id']:02d} | Expected: {case['expected']:<4} | "
              f"Predicted: {predicted:<4} | Score: {score:.2f} | Level: {level}")
        print(f"     Company: '{case['company'] or '(empty)'}'")
        print(f"     Flags: {result['flags'][:3]}")
        print(f"     Explanation: {result['explanation'][:120]}...")
        print()

    print(f"{'='*70}")
    print(f"  Accuracy: {correct}/{len(TEST_CASES)} ({round(correct/len(TEST_CASES)*100)}%)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_tests()
