"""Anicca: explainable screening co-pilot for high-volume hiring.

Run locally:  streamlit run app.py
Optional:     uvicorn webhook_server:app --port 8000   (auto-ingest from ATS / Zapier / LinkedIn partner feed)
"""
from __future__ import annotations

import json
import re

import pandas as pd
import streamlit as st

from anicca.demo import demo_applicants
from anicca.ingest import ingest_csv, ingest_email_bytes, ingest_file, ingest_text, poll_imap
from anicca.llm import PROVIDERS, LLMError, get_client
from anicca.parsing import mask_pii
from anicca.scoring import (BACKUP, JOB_PROFILES, NO_MATCH, REVIEW, SHORTLIST, Settings, adverse_impact,
                            explain, jd_to_profile, score)
from anicca.store import Store

st.set_page_config(page_title="Anicca Screening Co-pilot", page_icon="⚡", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem;}
  div[data-testid="stMetric"] {background: rgba(148,163,184,.08); border: 1px solid rgba(148,163,184,.25);
       border-radius: 12px; padding: 12px 16px;}
  .pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; margin-right:6px;
       background: rgba(56,189,248,.15); border:1px solid rgba(56,189,248,.4);}
  .flag {color:#f59e0b;}
</style>""", unsafe_allow_html=True)


@st.cache_resource
def get_store() -> Store:
    return Store()


store = get_store()
ss = st.session_state
ss.setdefault("profiles", {k: dict(v) for k, v in JOB_PROFILES.items()})
ss.setdefault("explanations", {})

# ======================================================================= SIDEBAR
with st.sidebar:
    st.markdown("### ⚡ Anicca")
    st.caption("Screening co-pilot: ranks every applicant with reasons, flags inconsistencies, keeps an audit trail.")

    with st.expander("🤖 AI assistant (free LLMs)", expanded=False):
        provider = st.selectbox("Provider", ["Off (rule-based only)"] + list(PROVIDERS), key="provider")
        api_key, model = "", ""
        if provider in PROVIDERS:
            cfg = PROVIDERS[provider]
            model = st.text_input("Model", value=cfg["default_model"], key="model",
                                  help="Free-tier model IDs change; edit if the provider renames them.")
            if not provider.startswith("Ollama"):
                api_key = st.text_input(f"API key (or set {cfg['env_key']} in secrets)", type="password",
                                        key="api_key", help="Kept only for this browser session.")
            st.caption(f"Get a free key: {cfg['signup']}")
        client = get_client(provider, api_key, model)
        if provider in PROVIDERS and client is None:
            st.warning("No key found: using the rule-based engine.")
        elif client is not None and st.button("Test connection"):
            try:
                st.success("Connected: " + client.chat("Reply with OK.", "ping", max_tokens=5)[:20])
            except LLMError as e:
                st.error(str(e))

    st.markdown("#### 🎯 Job")
    job_options = list(dict.fromkeys(list(ss.profiles) + store.jobs()))
    if "_pending_job" in ss:  # set before the widget exists; Streamlit forbids changing it afterwards
        ss.job = ss.pop("_pending_job")
    job = st.selectbox("Screening for", job_options + ["➕ New job from a job description"], key="job")
    if job == "➕ New job from a job description":
        new_title = st.text_input("Job title", "Backend Engineer")
        jd = st.text_area("Paste the job description", height=180)
        if st.button("Extract criteria", type="primary", disabled=not jd.strip()):
            prof = jd_to_profile(jd, client)
            ss.profiles[new_title] = {k: prof[k] for k in ("mandatory", "preferred", "min_exp")}
            if prof.get("llm_error"):
                st.warning("AI unavailable, used rules: " + prof["llm_error"][:120])
            ss["_pending_job"] = new_title
            st.rerun()
        st.stop()

    prof = ss.profiles.setdefault(job, {"mandatory": [], "preferred": [], "min_exp": 2.0})
    mand = st.text_input("Must-have skills", ", ".join(prof["mandatory"]), key=f"mand_{job}")
    pref = st.text_input("Nice-to-have skills", ", ".join(prof["preferred"]), key=f"pref_{job}")
    split = lambda s: [x.strip().lower() for x in s.split(",") if x.strip()]
    min_exp = st.slider("Minimum experience (years)", 0.0, 12.0, float(prof["min_exp"]), 0.5, key=f"exp_{job}")

    st.markdown("#### 📋 Shortlist")
    c1, c2 = st.columns(2)
    top_k = c1.number_input("Interview slots", 1, 200, 10)
    backup_k = c2.number_input("Backups", 0, 100, 5)
    floor = st.slider("Minimum score to shortlist", 30, 90, 60, 5,
                      help="Nobody below this score is shortlisted, even if slots stay empty.")
    with st.expander("Scoring weights"):
        w_mand = st.slider("Must-have skills (%)", 40, 80, 60, 5)
        w_pref = st.slider("Nice-to-have skills (%)", 10, 40, 25, 5)
        st.caption(f"Experience: **{100 - w_mand - w_pref}%**")

    st.markdown("#### ⚖️ Fairness & integrity")
    fairness = st.toggle("Don't penalize career gaps or freelance work", True)
    spec_adj = st.toggle("Adjust for generic / AI-polished wording", True)
    timeline = st.toggle("Flag timeline inconsistencies", True)

settings = Settings(split(mand), split(pref), min_exp, int(top_k), int(backup_k), float(floor), w_mand, w_pref,
                    fairness, spec_adj, timeline)


# ======================================================================= DATA
def candidate_frame(job_name: str) -> pd.DataFrame:
    df = store.to_frame(job_name)
    if df.empty:
        return df
    p = pd.json_normalize(df["parsed"].tolist())
    i = pd.json_normalize(df["integrity"].tolist())
    get = lambda frame, col, default: frame[col] if col in frame else pd.Series([default] * len(frame))
    return pd.DataFrame({
        "id": df["id"], "name": df["name"].replace("", "(no name)"), "source": df["source"],
        "status": df["status"], "received_at": df["received_at"], "text": df["raw_text"],
        "dupes": df["duplicate_sources"].apply(len),
        "skills": get(p, "skills", []), "experience_years": get(p, "experience_years", 0.0),
        "career_gap_years": get(p, "career_gap_years", 0.0), "worker_type": get(p, "worker_type", ""),
        "education": get(p, "education", ""), "parsed_by": get(p, "parsed_by", "rules"),
        "summary": get(p, "summary", ""),
        "specificity_score": get(i, "specificity_score", 60), "polish_flags": get(i, "polish_flags", []),
        "timeline_status": get(i, "timeline_status", ""), "timeline_flags": get(i, "timeline_flags", []),
    })


cands = candidate_frame(job)
ranked, info = score(cands, settings) if not cands.empty else (cands, {"eligible": 0, "quota_short": 0})

# ======================================================================= HEADER
st.title("Anicca Screening Co-pilot")
st.caption(f"Job: **{job}** · Must-haves: {', '.join(settings.mandatory) or '—'} · "
           f"AI: {'on (' + client.model + ')' if client else 'off, rule-based'}")

m1, m2, m3, m4, m5 = st.columns(5)
n = len(ranked)
m1.metric("Applicants", n)
m2.metric("Duplicates merged", int(cands["dupes"].sum()) if n else 0)
m3.metric("Shortlisted", f"{int((ranked['Decision'] == SHORTLIST).sum()) if n else 0} / {settings.top_k}")
m4.metric("Need your review", int((ranked["Triage"] == REVIEW).sum()) if n else 0)
m5.metric("Channels", ranked["source"].nunique() if n else 0)

tab_inbox, tab_short, tab_review, tab_fair, tab_setup = st.tabs(
    ["📥 Applicant inbox", "📋 Shortlist", "🔎 Candidate review", "⚖️ Fairness & audit", "🔌 Integrations"])


def show_results(results: list[dict]) -> None:
    """Keep the import summary, then rerun so the header counts and shortlist include the new applicants."""
    ss["last_results"] = results
    st.rerun()


def render_last_results() -> None:
    results = ss.pop("last_results", None)
    if not results:
        return
    s = pd.Series([r["status"] for r in results]).value_counts().to_dict()
    st.success(f"Added {s.get('added', 0)} · merged {s.get('duplicate', 0)} duplicates · "
               f"skipped {s.get('skipped', 0)} · errors {s.get('error', 0)}")
    bad = [r for r in results if r["status"] in ("skipped", "error") or r.get("llm_error")]
    if bad:
        with st.expander("Details"):
            st.dataframe(pd.DataFrame(bad), hide_index=True, width="stretch")


# ======================================================================= INBOX
with tab_inbox:
    st.markdown("Every channel lands here, gets parsed, de-duplicated across sources and scored for the job.")
    render_last_results()
    target_job = st.selectbox("Assign new applicants to", job_options, index=job_options.index(job))
    use_ai = st.checkbox("Use AI to read resumes during import", value=client is not None, disabled=client is None,
                         help="Better extraction of roles and dates. Free tiers allow roughly 30 requests/minute.")
    ai = client if use_ai else None

    a, b, c = st.columns(3)
    with a:
        st.markdown("**Resumes (PDF, DOCX, TXT or ZIP)**")
        st.caption("Includes resumes downloaded from your LinkedIn job post, Naukri or any ATS.")
        files = st.file_uploader("Resumes", type=["pdf", "docx", "txt", "zip"], accept_multiple_files=True,
                                 label_visibility="collapsed")
        src = st.selectbox("Where did these come from?", ["LinkedIn (download)", "Naukri", "Indeed", "Referral",
                                                          "Careers page", "Upload"], index=0)
        if st.button("Import resumes", disabled=not files, type="primary"):
            res, bar = [], st.progress(0.0)
            for k, f in enumerate(files):
                res += ingest_file(store, f.name, f.getvalue(), job=target_job, source=src, client=ai)
                bar.progress((k + 1) / len(files))
            show_results(res)
    with b:
        st.markdown("**Spreadsheet (CSV)**")
        st.caption("Columns: name, email, phone, and resume_text / summary / skills / experience.")
        csv = st.file_uploader("CSV", type=["csv"], label_visibility="collapsed")
        if st.button("Import CSV", disabled=csv is None):
            res = None
            try:
                res = ingest_csv(store, csv.getvalue(), job=target_job, client=ai)
            except ValueError as e:
                st.error(str(e))
            if res is not None:
                show_results(res)
    with c:
        st.markdown("**Application emails (.eml)**")
        st.caption("LinkedIn, Naukri and Indeed notification emails with resume attachments.")
        emls = st.file_uploader("Emails", type=["eml"], accept_multiple_files=True, label_visibility="collapsed")
        if st.button("Import emails", disabled=not emls):
            res = []
            for f in emls:
                res += ingest_email_bytes(store, f.getvalue(), job=target_job, client=ai)
            show_results(res)
        with st.popover("Connect a mailbox (IMAP)"):
            st.caption("Use a Gmail/Outlook **app password** and a dedicated label, e.g. a filter that labels "
                       "mails from jobs-noreply@linkedin.com as 'Applicants'.")
            host = st.text_input("IMAP host", "imap.gmail.com")
            user = st.text_input("Email address")
            pw = st.text_input("App password", type="password")
            folder = st.text_input("Folder / label", "Applicants")
            if st.button("Fetch new applications", disabled=not (user and pw)):
                res = None
                try:
                    res = poll_imap(store, host=host, user=user, password=pw, folder=folder,
                                    job=target_job, client=ai)
                except Exception as e:
                    st.error(f"Mailbox error: {e}")
                if res is not None:  # outside the try: st.rerun() works by raising an exception
                    show_results(res)

    st.divider()
    d1, d2, _ = st.columns([1, 1, 3])
    if d1.button("Load demo applicants"):
        show_results([ingest_text(store, a["text"], job=a["job"], source=a["source"]) for a in demo_applicants()])
    confirm = d2.checkbox("Confirm clear")
    if d2.button("Clear all applicants", disabled=not confirm):
        store.clear()
        st.rerun()

    allapps = store.to_frame()
    if allapps.empty:
        st.info("No applicants yet. Import resumes above, or click **Load demo applicants** to try it.")
    else:
        view = allapps[["name", "job", "source", "received_at", "status"]].copy()
        view["also applied via"] = allapps["duplicate_sources"].apply(lambda d: ", ".join(x["source"] for x in d))
        view["read by"] = allapps["parsed"].apply(lambda p: p.get("parsed_by", "rules"))
        st.dataframe(view, hide_index=True, width="stretch", height=320)

# ======================================================================= SHORTLIST
with tab_short:
    if not n:
        st.info("No applicants for this job yet. Add some in the Applicant inbox.")
    else:
        if info["quota_short"]:
            st.warning(f"Only {info['eligible']} applicants meet the minimum score of {settings.score_floor:g}. "
                       f"{info['quota_short']} interview slots stay open rather than filling them with weak matches.")
        decisions = st.multiselect("Show", [SHORTLIST, BACKUP, "Not shortlisted"], default=[SHORTLIST, BACKUP])
        show = ranked[ranked["Decision"].isin(decisions)] if decisions else ranked
        st.dataframe(
            show[["Rank", "name", "Score", "Decision", "Triage", "experience_years", "source", "status", "Reasons"]],
            hide_index=True, width="stretch", height=460,
            column_config={
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.1f"),
                "name": "Candidate", "experience_years": st.column_config.NumberColumn("Years", format="%.1f"),
                "Reasons": st.column_config.TextColumn("Why", width="large"),
                "source": "Channel", "status": "Status",
            })
        export = ranked.drop(columns=["text", "skills", "mand_hit", "pref_hit", "polish_flags", "timeline_flags"])
        st.download_button("Download shortlist (CSV)", export.to_csv(index=False).encode(),
                           file_name=f"shortlist_{re.sub(r'[^A-Za-z0-9]+', '_', job)}.csv")

# ======================================================================= CANDIDATE REVIEW
with tab_review:
    if not n:
        st.info("No applicants for this job yet.")
    else:
        labels = {f"#{r.Rank} · {r.name} · {r.Score:g} · {r.Decision}": r.id for r in ranked.itertuples()}
        pick = st.selectbox("Candidate", list(labels))
        r = ranked[ranked["id"] == labels[pick]].iloc[0]
        left, right = st.columns([1.1, 1])
        with left:
            st.markdown(f"### {r['name']}  \n<span class='pill'>{r['Decision']}</span>"
                        f"<span class='pill'>{r['Triage']}</span><span class='pill'>{r['source']}</span>",
                        unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Score", f"{r['Score']:g}")
            k2.metric("Experience", f"{r['experience_years']:g} yrs")
            k3.metric("Specificity", int(r["specificity_score"]),
                      help="How concrete the resume is (numbers, tools, outcomes). Low = ask for evidence.")
            st.markdown("**Why this score**")
            for part in r["Reasons"].split(" · "):
                st.markdown(f"- {part}")
            st.markdown("**Must-haves found:** " + (", ".join(r["mand_hit"]) or "none"))
            flags = list(r["timeline_flags"] or []) + list(r["polish_flags"] or [])
            if flags:
                st.markdown("**Integrity flags (verify in interview, never auto-reject)**")
                for f in flags:
                    st.markdown(f"<span class='flag'>⚠ {f}</span>", unsafe_allow_html=True)
            if r["summary"]:
                st.caption(r["summary"])
        with right:
            st.markdown("**AI assessment**")
            if client is None:
                st.caption("Turn on a free LLM in the sidebar for strengths, gaps and interview questions. "
                           "It never changes the score.")
            else:
                key = f"{r['id']}|{job}|{','.join(settings.mandatory)}"
                if st.button("Generate AI assessment", type="primary"):
                    try:
                        ss.explanations[key] = explain(client, settings, r["text"], r["Reasons"], job)
                        store.log("ai", r["id"], "ai_assessment", client.model)
                    except LLMError as e:
                        st.error(str(e))
                ex = ss.explanations.get(key)
                if ex:
                    st.info(ex.get("one_line", ""))
                    st.markdown("**Strengths**\n" + "\n".join(f"- {x}" for x in ex.get("strengths", [])))
                    st.markdown("**Gaps**\n" + "\n".join(f"- {x}" for x in ex.get("gaps", [])))
                    st.markdown("**Ask in the interview**\n" +
                                "\n".join(f"1. {x}" for x in ex.get("interview_questions", [])))
            st.markdown("**Your decision**")
            note = st.text_input("Note (saved to the audit log)", key=f"note_{r['id']}")
            b1, b2, b3 = st.columns(3)
            for col, status in ((b1, "Interview"), (b2, "On hold"), (b3, "Rejected")):
                if col.button(status, key=f"{status}_{r['id']}", width="stretch"):
                    store.set_status(r["id"], status, note=note)
                    st.toast(f"{r['name']}: {status}")
            st.caption(f"Current status: **{r['status']}**")
        with st.expander("Resume text (contact details and protected attributes masked)"):
            st.text(mask_pii(r["text"])[:8000])
        with st.expander("Delete this candidate's data (data-protection request)"):
            if st.checkbox("I confirm this candidate asked for deletion", key=f"del_{r['id']}") and \
                    st.button("Delete permanently", key=f"delbtn_{r['id']}"):
                store.delete(r["id"])
                st.rerun()

# ======================================================================= FAIRNESS & AUDIT
with tab_fair:
    if not n:
        st.info("Fairness and audit reports appear once applicants are scored.")
    else:
        st.markdown("Anicca never collects or scores gender, age, religion, caste or college name. "
                    "These checks look for unfair outcomes on the signals it *does* see.")
        x = ranked.assign(**{"Career gap": ranked["career_gap_years"].gt(0.5).map({True: "Has gap", False: "No gap"}),
                             "Work type": ranked["worker_type"].replace("", "Unknown"), "Channel": ranked["source"]})
        f1, f2 = st.columns(2)
        with f1:
            grp = st.selectbox("Selection rate by", ["Career gap", "Work type", "Channel"])
            st.dataframe(adverse_impact(x, grp), hide_index=True, width="stretch")
            st.caption("Impact ratio = group's shortlist rate ÷ best group's rate. Below 0.8 fails the four-fifths rule.")
        with f2:
            st.markdown("**Triage mix**")
            st.bar_chart(ranked["Triage"].value_counts())
        st.markdown("**Audit log**")
        audit = store.audit_frame()
        st.dataframe(audit.head(300), hide_index=True, width="stretch", height=260)
        pack = {"job": job, "criteria": settings.__dict__, "results": json.loads(
            ranked[["id", "name", "Score", "Decision", "Triage", "Reasons"]].to_json(orient="records")),
            "audit_log": json.loads(audit.to_json(orient="records"))}
        st.download_button("Download audit pack (JSON)", json.dumps(pack, indent=2, default=str).encode(),
                           file_name="anicca_audit_pack.json")

# ======================================================================= INTEGRATIONS
with tab_setup:
    st.markdown("""
### How applicants reach Anicca

| Channel | Status | How |
|---|---|---|
| **LinkedIn job posts** | ✅ Works today | Download applicant resumes from your LinkedIn job post → upload the PDFs or a ZIP in the inbox. |
| **LinkedIn / Naukri / Indeed emails** | ✅ Works today | Gmail filter on the sender → label `Applicants` → *Connect a mailbox*. Or upload `.eml` files. |
| **Any ATS, Google Form, Typeform, Zapier, Make** | ✅ Works today | POST to the webhook server (below). |
| **LinkedIn Easy Apply, automatic** | 🔒 Needs LinkedIn partner access | LinkedIn has no public API for Easy Apply applicants. Apply to LinkedIn's ATS partner programme; the adapter `anicca/linkedin.py` + `POST /webhook/linkedin` are ready to map its feed. Never scrape LinkedIn: it breaks their terms. |

### Webhook
Run `ANICCA_WEBHOOK_TOKEN=<secret> uvicorn webhook_server:app --port 8000`, then:
```bash
curl -X POST https://<your-host>/webhook/applications \\
  -H "X-Anicca-Token: <secret>" -H "Content-Type: application/json" \\
  -d '{"job": "QA Automation Engineer", "source": "Careers page",
       "candidate": {"name": "Asha Rao", "email": "asha@example.com"},
       "resume": {"filename": "asha.pdf", "content_base64": "<base64>"}}'
```
`resume.text` works instead of a file. Send a list to post up to 200 applicants at once.

### Free LLM setup
Add one key to `.streamlit/secrets.toml` (or Streamlit Cloud → Settings → Secrets):
```toml
GROQ_API_KEY = "..."        # or GEMINI_API_KEY / OPENROUTER_API_KEY
```
The AI reads resumes and job descriptions and writes assessments. **Scores stay rule-based** so every
decision is reproducible and explainable.
""")
