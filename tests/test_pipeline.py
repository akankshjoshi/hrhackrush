import base64
import io
import json
import os
import zipfile
from email.message import EmailMessage

import pandas as pd
import pytest

from anicca import llm
from anicca.demo import demo_applicants
from anicca.ingest import ingest_csv, ingest_email_bytes, ingest_file, ingest_payload, ingest_text
from anicca.integrity import specificity, timeline_checks
from anicca.parsing import find_skills, mask_pii, parse_resume
from anicca.scoring import JOB_PROFILES, NOT_SHORTLISTED, SHORTLIST, Settings, jd_to_profile, score
from anicca.store import Store

RESUME = """Asha Rao
QA Automation Engineer
asha.rao@example.com | +91 98765 43210 | linkedin.com/in/asha-rao
Female, Date of Birth: 01/01/1995
EXPERIENCE
QA Engineer, Freshworks  Jan 2022 - Present
- Cut regression runtime by 70% with Playwright and PyTest on Jenkins CI/CD
Test Engineer, Infosys  Jun 2019 - Dec 2021
- Automated 600 Selenium tests; built Postman suites for 80 REST APIs
SKILLS: Selenium, Playwright, PyTest, Postman, Java, SQL, k8s
"""


@pytest.fixture
def store():
    return Store(":memory:")


def test_parse_and_mask():
    p = parse_resume(RESUME)
    assert {"selenium", "playwright", "pytest", "postman", "kubernetes"} <= set(p["skills"])
    assert p["email"] == "asha.rao@example.com"
    assert 6 <= p["experience_years"] <= 8
    assert p["roles"][0]["company"] == "Freshworks"
    masked = mask_pii(RESUME)
    assert "asha.rao@example.com" not in masked and "98765" not in masked
    assert "Female" not in masked and "Date of Birth" not in masked


def test_skill_boundaries():
    assert "java" not in find_skills("Expert in JavaScript")
    assert "go" not in find_skills("Good communication, going forward")


def test_integrity_flags():
    fluff = "Results-driven dynamic passionate self-starter who leveraged synergy and spearheaded innovative " * 10
    assert specificity(fluff, 2)["specificity_score"] < 45
    flags = timeline_checks([{"company": "A", "start": 2019.0, "end": 2022.0},
                             {"company": "B", "start": 2020.0, "end": 2023.0}], [], 4)["timeline_flags"]
    assert any("Overlapping" in f for f in flags)


def test_dedupe_across_channels(store):
    a = ingest_text(store, RESUME, job="QA", source="LinkedIn (email)")
    b = ingest_text(store, RESUME, job="QA", source="Naukri (email)")
    assert a["status"] == "added" and b["status"] == "duplicate"
    df = store.to_frame("QA")
    assert len(df) == 1 and df.iloc[0]["duplicate_sources"][0]["source"] == "Naukri (email)"


def _frame(store, job):
    df = store.to_frame(job)
    p, i = pd.json_normalize(df["parsed"].tolist()), pd.json_normalize(df["integrity"].tolist())
    return pd.DataFrame({"id": df.id, "name": df.name, "text": df.raw_text, "skills": p.skills,
                         "experience_years": p.experience_years, "career_gap_years": p.career_gap_years,
                         "worker_type": p.worker_type, "specificity_score": i.specificity_score,
                         "timeline_status": i.timeline_status, "timeline_flags": i.timeline_flags,
                         "polish_flags": i.polish_flags})


def test_topk_never_shortlists_below_floor(store):
    for a in demo_applicants():
        ingest_text(store, a["text"], job=a["job"], source=a["source"])
    pr = JOB_PROFILES["QA Automation Engineer"]
    out, info = score(_frame(store, "QA Automation Engineer"), Settings(pr["mandatory"], pr["preferred"],
                                                                         pr["min_exp"], top_k=50, score_floor=70))
    short = out[out["Decision"] == SHORTLIST]
    assert (short["Score"] >= 70).all()
    assert info["quota_short"] == 50 - info["eligible"]
    assert (out[out["Score"] < 70]["Decision"] == NOT_SHORTLISTED).all()
    assert out["Reasons"].str.contains("Must-haves").all()


def test_zip_csv_email_webhook(store):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("a.txt", RESUME)
        z.writestr("__MACOSX/._a.txt", "junk")
    assert [r["status"] for r in ingest_file(store, "batch.zip", buf.getvalue(), job="QA")] == ["added"]

    csv = pd.DataFrame([{"Name": "Ravi K", "Email": "ravi@example.com",
                         "Summary": "Data engineer, 5 years of experience with Python, SQL, Spark, Airflow"}])
    assert ingest_csv(store, csv.to_csv(index=False).encode(), job="Data")[0]["status"] == "added"

    msg = EmailMessage()
    msg["From"] = "LinkedIn <jobs-noreply@linkedin.com>"
    msg["Subject"] = "New application for Full Stack Developer"
    msg.set_content("New applicant")
    msg.add_attachment(RESUME.replace("Asha Rao", "Meena Iyer").replace("asha.rao", "meena").encode(),
                       maintype="text", subtype="plain", filename="meena.txt")
    res = ingest_email_bytes(store, msg.as_bytes(), job="QA", job_from_subject=True)
    assert res[0]["status"] == "added"
    row = store.to_frame("Full Stack Developer").iloc[0]
    assert row["source"] == "LinkedIn (email)"

    payload = {"provider": "linkedin", "linkedin_application": {
        "job": {"title": "QA"}, "applicant": {"firstName": "Dev", "lastName": "Nair", "email": "dev@example.com"},
        "resume": {"fileName": "dev.txt", "contentBase64": base64.b64encode(
            b"Dev Nair\nSDET 4 years of experience Selenium Playwright PyTest Postman").decode()}}}
    assert ingest_payload(store, payload)[0]["status"] == "added"
    assert store.to_frame("QA").query("email == 'dev@example.com'").iloc[0]["source"] == "LinkedIn (partner feed)"


def test_jd_rules():
    jd = "We need 3+ years. Required: Python, SQL, Airflow, Spark. Nice to have: Kafka, Snowflake."
    p = jd_to_profile(jd)
    assert {"python", "sql", "airflow", "spark"} <= set(p["mandatory"])
    assert "kafka" in p["preferred"] and p["min_exp"] == 3


def test_llm_client_parses_and_falls_back(monkeypatch):
    class R:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "```json\n{\"skills\": [\"Selenium\", \"K8s\"], "
                                 "\"experience_years\": 6, \"roles\": [], \"projects\": []}\n```"}}]}
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: R())
    c = llm.LLMClient("Groq (free tier)", api_key="test")
    p = parse_resume(RESUME, c)
    assert "kubernetes" in p["skills"] and p["parsed_by"].startswith("llm:")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert llm.get_client("Groq (free tier)") is None  # no key -> rule-based
    assert llm.get_client("Off (rule-based only)") is None


def test_webhook_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("ANICCA_DB", str(tmp_path / "w.db"))
    monkeypatch.setenv("ANICCA_WEBHOOK_TOKEN", "s3cret")
    import importlib
    import webhook_server
    importlib.reload(webhook_server)
    from fastapi.testclient import TestClient
    c = TestClient(webhook_server.app)
    body = {"job": "QA", "candidate": {"name": "Z"}, "resume": {"text": RESUME}}
    assert c.post("/webhook/applications", json=body).status_code == 401
    r = c.post("/webhook/applications", json=body, headers={"X-Anicca-Token": "s3cret"})
    assert r.status_code == 200 and r.json()["results"][0]["status"] == "added"


def test_ai_assessment_uses_masked_text(monkeypatch):
    from anicca.scoring import explain
    seen = {}

    class R:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": '{"fit": "strong", "strengths": ["Playwright"], '
                                 '"gaps": [], "interview_questions": ["q1", "q2", "q3"], "one_line": "Good"}'}}]}

    def fake_post(url, headers=None, json=None, timeout=None):
        seen["prompt"] = json["messages"][1]["content"]
        return R()
    monkeypatch.setattr(llm.requests, "post", fake_post)
    c = llm.LLMClient("Google Gemini (free tier)", api_key="k")
    pr = JOB_PROFILES["QA Automation Engineer"]
    out = explain(c, Settings(pr["mandatory"], pr["preferred"], 3), RESUME, "facts", "QA")
    assert out["fit"] == "strong"
    assert "asha.rao@example.com" not in seen["prompt"] and "Female" not in seen["prompt"]
