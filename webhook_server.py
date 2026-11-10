"""Webhook receiver so ATSs, form tools, Zapier/Make and (with partner access)
LinkedIn can push applicants into Anicca automatically.

Run:  ANICCA_WEBHOOK_TOKEN=change-me uvicorn webhook_server:app --port 8000
Both this server and the Streamlit app must point ANICCA_DB at the same file
(or, in production, the same Postgres database).
"""
from __future__ import annotations

import hmac
import os

from fastapi import FastAPI, Header, HTTPException, Request

from anicca.ingest import ingest_payload
from anicca.llm import get_client
from anicca.store import Store

app = FastAPI(title="Anicca Applicant Webhook", version="1.0")
store = Store()


def _check(token: str | None) -> None:
    expected = os.environ.get("ANICCA_WEBHOOK_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Set ANICCA_WEBHOOK_TOKEN before accepting applicants")
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Bad or missing X-Anicca-Token")


def _client():
    return get_client(os.environ.get("ANICCA_LLM_PROVIDER"))


@app.get("/health")
def health():
    return {"ok": True, "applicants": store.count()}


@app.post("/webhook/applications")
async def applications(request: Request, x_anicca_token: str | None = Header(default=None)):
    _check(x_anicca_token)
    body = await request.json()
    items = body if isinstance(body, list) else [body]
    if len(items) > 200:
        raise HTTPException(413, "Send at most 200 applicants per request")
    results = []
    for item in items:
        results += ingest_payload(store, item, client=_client())
    return {"results": results}


@app.post("/webhook/linkedin")
async def linkedin(request: Request, x_anicca_token: str | None = Header(default=None)):
    _check(x_anicca_token)
    body = await request.json()
    body.setdefault("provider", "linkedin")
    return {"results": ingest_payload(store, body, client=_client())}
