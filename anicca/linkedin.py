"""LinkedIn integration: what is possible, and the adapter for when access is granted.

Reality check (verify current terms with LinkedIn):
- LinkedIn has no public API that gives a third-party app the Easy Apply
  applicants of a job. Applicant data reaches software through LinkedIn's
  partner programmes for ATS vendors (e.g. "Apply Connect" / Recruiter System
  Connect), which require a partnership agreement.
- Scraping linkedin.com or automating a logged-in browser violates LinkedIn's
  User Agreement and risks account bans. This project never does that.

What works today, legally:
1. Job posters can download applicant resumes from their LinkedIn job post.
   Upload them here as PDFs or a ZIP (Applicant Inbox -> Upload).
2. LinkedIn emails a notification for each application. Forward those emails
   (Gmail filter -> label "Applicants") and let the app poll the label over IMAP.
3. If you use an ATS that is a LinkedIn partner, LinkedIn applicants land in the
   ATS; forward them here with the ATS's webhook or CSV export.
4. Once you have partner access, point the partner feed at
   POST /webhook/linkedin on webhook_server.py. `from_linkedin_payload` maps it
   into Anicca's contract. FIELD NAMES BELOW ARE PLACEHOLDERS: replace them with
   the ones in LinkedIn's partner documentation before going live.
"""
from __future__ import annotations


def _get(d: dict, *path, default=""):
    for p in path:
        if not isinstance(d, dict):
            return default
        d = d.get(p)
        if d is None:
            return default
    return d


def from_linkedin_payload(payload: dict) -> dict:
    app = payload.get("linkedin_application", payload)
    first = _get(app, "applicant", "firstName")
    last = _get(app, "applicant", "lastName")
    return {
        "source": "LinkedIn (partner feed)",
        "job": _get(app, "job", "title") or payload.get("job", "Unassigned"),
        "candidate": {
            "name": f"{first} {last}".strip(),
            "email": _get(app, "applicant", "email"),
            "phone": _get(app, "applicant", "phone"),
            "linkedin_url": _get(app, "applicant", "profileUrl"),
        },
        "resume": {
            "filename": _get(app, "resume", "fileName") or "linkedin_resume.pdf",
            "content_base64": _get(app, "resume", "contentBase64"),
            "text": _get(app, "resume", "text") or _get(app, "applicant", "summary"),
        },
    }
