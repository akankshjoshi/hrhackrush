"""Applicant Inbox: every channel funnels into the same parse -> dedupe -> store path.

Channels
- File upload: PDF / DOCX / TXT, or a ZIP of them (e.g. resumes downloaded from
  a LinkedIn job post, Naukri, Indeed or an ATS export).
- CSV import: one row per applicant (any ATS / job-board export).
- Email: forward application emails (with resume attachments) to a mailbox the
  app polls over IMAP, or upload saved .eml files.
- Webhook: any ATS, form tool, Zapier/Make flow, or LinkedIn partner feed can
  POST applicants to webhook_server.py.
"""
from __future__ import annotations

import base64
import email
import email.policy
import imaplib
import io
import re
from typing import Optional

import pandas as pd

from .integrity import assess
from .linkedin import from_linkedin_payload
from .llm import LLMClient
from .parsing import extract_text, fingerprint, iter_files, parse_resume
from .store import Store, now_iso


def ingest_text(store: Store, text: str, *, job: str, source: str, source_ref: str = "",
                contact: Optional[dict] = None, client: Optional[LLMClient] = None,
                received_at: Optional[str] = None) -> dict:
    text = (text or "").strip()
    if len(text) < 40:
        return {"status": "skipped", "reason": "too little text (scanned PDF? try a text-based file)", "ref": source_ref}
    contact = {k: v for k, v in (contact or {}).items() if v}
    fp = fingerprint(text)
    parsed = parse_resume(text, client)
    email_addr = contact.get("email") or parsed.get("email", "")
    dup = store.find_duplicate(fp, email_addr, job)
    if dup:
        store.add_duplicate_source(dup, source)
        return {"status": "duplicate", "id": dup, "ref": source_ref}
    rec = {
        "fingerprint": fp, "job": job, "source": source, "source_ref": source_ref,
        "raw_text": text, "parsed": parsed, "integrity": assess(text, parsed),
        "received_at": received_at or now_iso(), **contact,
    }
    aid = store.insert(rec)
    return {"status": "added", "id": aid, "ref": source_ref, "parsed_by": parsed.get("parsed_by"),
            "llm_error": parsed.get("llm_error")}


def ingest_file(store: Store, filename: str, data: bytes, *, job: str, source: str = "Upload",
                contact: Optional[dict] = None, client: Optional[LLMClient] = None) -> list[dict]:
    results = []
    for name, blob in iter_files(filename, data):
        try:
            text = extract_text(name, blob)
        except Exception as e:  # corrupt or unsupported file: report, don't crash the batch
            results.append({"status": "error", "ref": name, "reason": str(e)[:200]})
            continue
        results.append(ingest_text(store, text, job=job, source=source, source_ref=name,
                                   contact=contact, client=client))
    return results


# ------------------------------------------------------------------ CSV
TEXT_COLS = ["resume_text", "resume", "cv_text", "summary", "profile", "about", "experience", "skills", "headline"]


def ingest_csv(store: Store, data: bytes, *, job: str, source: str = "CSV import",
               client: Optional[LLMClient] = None) -> list[dict]:
    df = pd.read_csv(io.BytesIO(data))
    cols = {c.lower().strip().replace(" ", "_"): c for c in df.columns}
    text_cols = [cols[c] for c in TEXT_COLS if c in cols]
    if not text_cols:
        raise ValueError("CSV needs at least one of: " + ", ".join(TEXT_COLS))
    pick = lambda row, *names: next((str(row[cols[n]]) for n in names if n in cols and pd.notna(row[cols[n]])), "")
    results = []
    for i, row in df.iterrows():
        text = "\n".join(f"{c}: {row[c]}" for c in text_cols if pd.notna(row[c]))
        name = pick(row, "name", "full_name", "candidate_name") or \
            f"{pick(row, 'first_name')} {pick(row, 'last_name')}".strip()
        contact = {"name": name, "email": pick(row, "email", "email_address"),
                   "phone": pick(row, "phone", "phone_number", "mobile"),
                   "linkedin_url": pick(row, "linkedin", "linkedin_url", "profile_url")}
        row_job = pick(row, "job", "job_title", "position", "applied_for") or job
        row_src = pick(row, "source", "channel") or source
        results.append(ingest_text(store, (name + "\n" + text) if name else text, job=row_job, source=row_src,
                                   source_ref=f"row {i + 2}", contact=contact, client=client))
    return results


# ------------------------------------------------------------------ Email
def _source_from_sender(sender: str) -> str:
    s = sender.lower()
    for key, label in [("linkedin", "LinkedIn (email)"), ("naukri", "Naukri (email)"), ("indeed", "Indeed (email)"),
                       ("instahyre", "Instahyre (email)"), ("wellfound", "Wellfound (email)"),
                       ("internshala", "Internshala (email)")]:
        if key in s:
            return label
    return "Email"


def _job_from_subject(subject: str, default: str) -> str:
    m = re.search(r"(?:application|applied|applicant)s?\s*(?:for|:|-)\s*(.+?)(?:\s+at\s+|\s+-\s+|$)", subject or "", re.I)
    return m.group(1).strip()[:80] if m else default


def ingest_email_bytes(store: Store, raw: bytes, *, job: str, client: Optional[LLMClient] = None,
                       job_from_subject: bool = False) -> list[dict]:
    msg = email.message_from_bytes(raw, policy=email.policy.default)
    sender, subject = str(msg.get("From", "")), str(msg.get("Subject", ""))
    source = _source_from_sender(sender)
    the_job = _job_from_subject(subject, job) if job_from_subject else job
    results = []
    for part in msg.iter_attachments():
        fname = part.get_filename() or ""
        if fname.lower().endswith((".pdf", ".docx", ".txt", ".zip")):
            results += ingest_file(store, fname, part.get_payload(decode=True), job=the_job, source=source,
                                   client=client)
    if not results:  # resume pasted into the body
        body = msg.get_body(preferencelist=("plain", "html"))
        text = body.get_content() if body else ""
        text = re.sub(r"<[^>]+>", " ", text)
        results.append(ingest_text(store, text, job=the_job, source=source, source_ref=subject[:80], client=client))
    return results


def poll_imap(store: Store, *, host: str, user: str, password: str, folder: str = "INBOX",
              job: str, client: Optional[LLMClient] = None, limit: int = 50) -> list[dict]:
    """Read UNSEEN messages from a dedicated folder/label and ingest them.

    Use an app password and a dedicated label (e.g. Gmail filter -> 'Applicants'),
    never your main inbox password.
    """
    results: list[dict] = []
    with imaplib.IMAP4_SSL(host) as M:
        M.login(user, password)
        M.select(folder)
        _, ids = M.search(None, "UNSEEN")
        for num in ids[0].split()[:limit]:
            _, data = M.fetch(num, "(RFC822)")
            results += ingest_email_bytes(store, data[0][1], job=job, client=client, job_from_subject=True)
    return results


# ------------------------------------------------------------------ Webhook payloads
def ingest_payload(store: Store, payload: dict, *, client: Optional[LLMClient] = None) -> list[dict]:
    """Generic JSON contract (see README). LinkedIn partner payloads are mapped first."""
    if payload.get("provider") == "linkedin" or "linkedin_application" in payload:
        payload = from_linkedin_payload(payload)
    job = payload.get("job") or "Unassigned"
    source = payload.get("source") or "Webhook"
    cand = payload.get("candidate") or {}
    contact = {k: cand.get(k, "") for k in ("name", "email", "phone", "linkedin_url")}
    resume = payload.get("resume") or {}
    if resume.get("content_base64"):
        data = base64.b64decode(resume["content_base64"])
        return ingest_file(store, resume.get("filename", "resume.pdf"), data, job=job, source=source,
                           contact=contact, client=client)
    text = resume.get("text") or payload.get("resume_text") or ""
    return [ingest_text(store, text, job=job, source=source, source_ref=cand.get("email", ""),
                        contact=contact, client=client)]
