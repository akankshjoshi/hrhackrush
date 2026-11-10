# Anicca: Screening Co-pilot

Explainable resume screening for high-volume hiring. Every applicant, from every channel, lands in one inbox and gets parsed, de-duplicated, checked for integrity issues and ranked against the job, with a reason for every score and an audit trail for every decision.

## What's new in v2

- **Applicant inbox**: PDF / DOCX / TXT / ZIP upload, CSV import, application emails (`.eml` or a live IMAP mailbox), and a webhook for ATSs, forms and Zapier/Make.
- **Free LLM support**: Groq, Google Gemini, OpenRouter (free models) or a local Ollama model, through one OpenAI-compatible client. The AI reads resumes and job descriptions and writes assessments; **scores stay rule-based** so they are reproducible. With no key, everything still runs on rules.
- **Job description → criteria**: paste a JD to extract must-haves, nice-to-haves and minimum experience.
- **Shortlist bug fixed**: v1 filled every interview slot even with "Auto-Reject" candidates. Now only applicants above the minimum score are shortlisted; empty slots stay empty and the app says so.
- **Reasons for every score**: must-haves found and missing, experience vs requirement, integrity flags.
- **Integrity checks on real resumes**: overlapping jobs, projects outside employer tenure, claimed vs dated experience, and a specificity score for generic / AI-polished wording. These flag a candidate for review; they never auto-reject.
- **Privacy**: contact details and protected attributes are masked before scoring or any LLM call; per-candidate deletion; audit log and JSON audit pack.
- **Fairness report**: shortlist rate and four-fifths impact ratio by career gap, work type and channel.

## LinkedIn: what works and what doesn't

LinkedIn has **no public API** that gives third-party apps the Easy Apply applicants of a job, and scraping LinkedIn violates its User Agreement. Anicca supports every legal route:

| Route | How |
|---|---|
| Resumes from your LinkedIn job post | Download them from LinkedIn → upload the PDFs or a ZIP |
| LinkedIn application emails | Gmail filter on the sender → label `Applicants` → *Connect a mailbox* in the inbox |
| ATS that is a LinkedIn partner | Forward from the ATS via webhook or CSV |
| LinkedIn partner feed (needs approval) | `POST /webhook/linkedin`; mapping in `anicca/linkedin.py` (placeholder field names, update from LinkedIn's partner docs) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Click **Load demo applicants** in the inbox to try it with synthetic resumes.

### Turn on a free LLM

Get a free key from [Groq](https://console.groq.com/keys), [Google AI Studio](https://aistudio.google.com/apikey) or [OpenRouter](https://openrouter.ai/keys), then either paste it into the sidebar (*AI assistant*) or add it to `.streamlit/secrets.toml` (see `secrets.toml.example`). On Streamlit Cloud, use *App → Settings → Secrets*. Free model IDs change often; edit the model name in the sidebar or set `GROQ_MODEL` / `GEMINI_MODEL`.

### Webhook server (optional)

```bash
ANICCA_WEBHOOK_TOKEN=change-me ANICCA_DB=data/anicca.db uvicorn webhook_server:app --port 8000
```

```json
POST /webhook/applications   (header X-Anicca-Token)
{"job": "QA Automation Engineer", "source": "Careers page",
 "candidate": {"name": "Asha Rao", "email": "asha@example.com"},
 "resume": {"filename": "asha.pdf", "content_base64": "..."}}
```

Use `resume.text` instead of a file, or post a list of up to 200 applicants. The app and the webhook must share the same database. Streamlit Community Cloud resets its disk on redeploy, so move to Postgres or Supabase before real pilots.

## Project layout

```
app.py                 Streamlit UI
webhook_server.py      FastAPI receiver for ATS / Zapier / LinkedIn partner feeds
anicca/llm.py          free-LLM client (Groq, Gemini, OpenRouter, Ollama)
anicca/parsing.py      text extraction, rule + LLM parsing, PII masking
anicca/integrity.py    timeline and specificity checks
anicca/scoring.py      explainable scoring, shortlist logic, JD extraction, fairness report
anicca/ingest.py       upload, CSV, email/IMAP and webhook ingestion
anicca/store.py        SQLite store and audit log
anicca/linkedin.py     LinkedIn partner-feed adapter
anicca/demo.py         synthetic applicants
tests/                 pytest suite
```

```bash
python -m pytest -q
```

The earlier notebooks and datasets (`Round2.ipynb`, `app.ipynb`, `*.xlsx`, `*.parquet`) are kept unchanged.
