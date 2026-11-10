"""Deterministic, explainable scoring. The LLM never sets the score.

Why: a recruiter must be able to defend every shortlist decision, and the same
resume must get the same score tomorrow. The LLM helps read resumes and JDs and
writes explanations; the ranking itself is a transparent formula.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .llm import LLMClient, LLMError
from .parsing import find_skills, mask_pii

JOB_PROFILES = {
    "QA Automation Engineer": {
        "mandatory": ["selenium", "playwright", "pytest", "automation testing", "postman"],
        "preferred": ["ci/cd", "jenkins", "docker", "jmeter", "java", "sql"], "min_exp": 3.0},
    "Data Engineer": {
        "mandatory": ["python", "sql", "spark", "etl", "data warehouse", "airflow"],
        "preferred": ["aws", "kafka", "snowflake", "databricks", "docker", "kubernetes"], "min_exp": 3.0},
    "DevOps & Cloud Engineer": {
        "mandatory": ["aws", "docker", "kubernetes", "terraform", "ci/cd", "linux"],
        "preferred": ["ansible", "jenkins", "prometheus", "grafana", "python", "bash"], "min_exp": 3.5},
    "Full Stack Developer": {
        "mandatory": ["javascript", "react", "node.js", "typescript", "html", "css"],
        "preferred": ["next.js", "postgresql", "mongodb", "docker", "graphql", "tailwind"], "min_exp": 2.5},
    "Data Scientist / ML Engineer": {
        "mandatory": ["python", "machine learning", "scikit-learn", "pandas", "numpy", "statistics"],
        "preferred": ["pytorch", "tensorflow", "nlp", "llm", "deep learning", "docker", "aws"], "min_exp": 3.0},
    "Senior Backend Engineer": {
        "mandatory": ["java", "spring boot", "microservices", "sql", "rest api"],
        "preferred": ["kafka", "redis", "docker", "kubernetes", "aws", "postgresql"], "min_exp": 4.0},
}

STRONG, POSSIBLE, REVIEW, NO_MATCH = "Strong match", "Possible match", "Needs your review", "Not a match"
SHORTLIST, BACKUP, NOT_SHORTLISTED = "Shortlist", "Backup", "Not shortlisted"


@dataclass
class Settings:
    mandatory: list[str]
    preferred: list[str]
    min_exp: float
    top_k: int = 10
    backup_k: int = 5
    score_floor: float = 60.0
    w_mand: int = 60
    w_pref: int = 25
    fairness: bool = True
    specificity_adjust: bool = True
    timeline_audit: bool = True
    extra: dict = field(default_factory=dict)

    @property
    def w_exp(self) -> int:
        return 100 - self.w_mand - self.w_pref


# ------------------------------------------------------------------ JD -> profile
JD_SYSTEM = ("You turn a job description into screening criteria. Only list skills the JD actually names. "
             "Mandatory = required/must-have; preferred = nice-to-have. Never include age, gender, college "
             "names or other protected or proxy attributes.")
JD_SCHEMA = ('Return JSON: {"title": str, "mandatory": [lowercase skills, max 8], '
             '"preferred": [lowercase skills, max 8], "min_exp": number}')


def jd_to_profile(jd: str, client: Optional[LLMClient] = None) -> dict:
    yrs = re.findall(r"(\d{1,2})\s*\+?\s*(?:-\s*\d+\s*)?(?:years|yrs)", jd, re.I)
    rule = {"title": "", "min_exp": float(min(map(int, yrs))) if yrs else 2.0}
    req_part = re.split(r"(?i)(nice to have|preferred|good to have|bonus|plus:)", jd)[0]
    mand = find_skills(req_part)
    pref = [s for s in find_skills(jd) if s not in mand]
    rule.update({"mandatory": mand[:8], "preferred": pref[:8], "parsed_by": "rules"})
    if client is None:
        return rule
    try:
        d = client.chat_json(JD_SYSTEM, f"{JD_SCHEMA}\n\nJOB DESCRIPTION:\n{jd[:8000]}")
        mand = [str(s).lower().strip() for s in d.get("mandatory", []) if str(s).strip()][:8]
        pref = [str(s).lower().strip() for s in d.get("preferred", []) if str(s).strip() and s not in mand][:8]
        return {"title": d.get("title", ""), "mandatory": mand or rule["mandatory"],
                "preferred": pref or rule["preferred"],
                "min_exp": float(d.get("min_exp") or rule["min_exp"]), "parsed_by": f"llm:{client.model}"}
    except (LLMError, ValueError, TypeError) as e:
        rule["llm_error"] = str(e)
        return rule


# ------------------------------------------------------------------ scoring
def _match(text: str, skills: list[str], wanted: list[str]) -> list[str]:
    have = set(skills) | set(find_skills(text, wanted))
    return [w for w in wanted if w in have]


def score(df: pd.DataFrame, s: Settings) -> tuple[pd.DataFrame, dict]:
    """df needs: id, name, text, skills (list), experience_years, career_gap_years, worker_type,
    specificity_score, timeline_status, timeline_flags, polish_flags."""
    if df.empty:
        return df.assign(Score=[]), {"eligible": 0, "quota_short": 0}
    df = df.copy().reset_index(drop=True)
    mand_hit = [_match(t, sk, s.mandatory) for t, sk in zip(df["text"], df["skills"])]
    pref_hit = [_match(t, sk, s.preferred) for t, sk in zip(df["text"], df["skills"])]
    m_ratio = np.array([len(h) / max(1, len(s.mandatory)) for h in mand_hit])
    p_ratio = np.array([len(h) / max(1, len(s.preferred)) for h in pref_hit])

    exps = df["experience_years"].fillna(0).astype(float).to_numpy()
    gaps = df["career_gap_years"].fillna(0).astype(float).to_numpy()
    non_trad = df["worker_type"].fillna("").str.contains("Freelance|Contract", case=False).to_numpy()
    regularized = (gaps > 0.5) | non_trad
    eff = np.where(regularized, np.maximum(exps, s.min_exp * 0.9), exps) if s.fairness else exps
    e_ratio = np.clip(eff / max(0.5, s.min_exp), 0, 1)

    total = m_ratio * s.w_mand + p_ratio * s.w_pref + e_ratio * s.w_exp
    spec = df["specificity_score"].fillna(60).astype(float).to_numpy()
    if s.specificity_adjust:
        total = total + np.where(spec >= 75, 3, 0) - np.where(spec < 45, 4, 0)
    flagged = (df["timeline_status"] == "Check").to_numpy()
    if s.timeline_audit:
        total = total - np.where(flagged, 10, 0)
    total = np.round(np.clip(total, 0, 100), 1)

    triage = np.where(total < 45, NO_MATCH,
             np.where(flagged | (np.abs(total - s.score_floor) <= 6), REVIEW,
             np.where(total >= 76, STRONG, POSSIBLE)))

    out = df.assign(Score=total, Triage=triage, mand_hit=mand_hit, pref_hit=pref_hit,
                    effective_exp=np.round(eff, 1), regularized=regularized & s.fairness)
    out = out.sort_values(["Score", "experience_years"], ascending=[False, False]).reset_index(drop=True)
    out.insert(0, "Rank", np.arange(1, len(out) + 1))

    # FIX: only candidates above the floor (and not "Not a match") can fill the quota.
    eligible = (out["Score"] >= s.score_floor) & (out["Triage"] != NO_MATCH)
    order = np.cumsum(eligible.to_numpy())
    rec = np.where(~eligible, NOT_SHORTLISTED,
          np.where(order <= s.top_k, SHORTLIST,
          np.where(order <= s.top_k + s.backup_k, BACKUP, NOT_SHORTLISTED)))
    out["Decision"] = rec
    out["Reasons"] = [reasons(r, s) for r in out.to_dict("records")]
    n_elig = int(eligible.sum())
    return out, {"eligible": n_elig, "quota_short": max(0, s.top_k - n_elig)}


def reasons(r: dict, s: Settings) -> str:
    miss = [m for m in s.mandatory if m not in r["mand_hit"]]
    parts = [f"Must-haves {len(r['mand_hit'])}/{len(s.mandatory)}"
             + (f" (missing: {', '.join(miss[:4])})" if miss else " (all)")]
    if r["pref_hit"]:
        parts.append(f"Nice-to-have: {', '.join(r['pref_hit'][:4])}")
    exp_note = f"{r['experience_years']:g} yrs vs {s.min_exp:g} required"
    if r.get("regularized") and r["effective_exp"] > r["experience_years"]:
        exp_note += " (career gap / freelance work not penalized)"
    parts.append(exp_note)
    for f in (r.get("timeline_flags") or [])[:2]:
        parts.append("Check: " + f)
    if r.get("specificity_score", 60) < 45:
        parts.append("Generic wording; ask for concrete examples")
    return " · ".join(parts)


# ------------------------------------------------------------------ LLM second opinion
EXPLAIN_SYSTEM = (
    "You are a fair, careful technical recruiter. Assess the candidate ONLY on job-relevant evidence in the "
    "resume. Do not consider or infer gender, age, religion, caste, college prestige, location or name. "
    "Quote short evidence from the resume. If evidence is missing, say so rather than guessing."
)
EXPLAIN_SCHEMA = ('Return JSON: {"fit": "strong|possible|weak", "strengths": [max 3 strings with evidence], '
                  '"gaps": [max 3 strings], "interview_questions": [3 specific questions to verify claims], '
                  '"one_line": "a sentence a recruiter can paste to a hiring manager"}')


def explain(client: LLMClient, s: Settings, resume_text: str, facts: str, role: str) -> dict:
    prompt = (f"{EXPLAIN_SCHEMA}\n\nROLE: {role}\nMUST-HAVE: {', '.join(s.mandatory)}\n"
              f"NICE-TO-HAVE: {', '.join(s.preferred)}\nMIN EXPERIENCE: {s.min_exp} years\n"
              f"RULE-BASED FINDINGS: {facts}\n\nRESUME (contact details masked):\n{mask_pii(resume_text)[:10000]}")
    return client.chat_json(EXPLAIN_SYSTEM, prompt, max_tokens=900)


# ------------------------------------------------------------------ fairness report
def adverse_impact(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Selection rate per group and ratio to the best group (four-fifths rule)."""
    if df.empty or group_col not in df:
        return pd.DataFrame()
    g = df.groupby(group_col).agg(applicants=("Decision", "size"),
                                  shortlisted=("Decision", lambda x: (x == SHORTLIST).sum()))
    g["selection_rate"] = (g["shortlisted"] / g["applicants"]).round(2)
    best = g["selection_rate"].max()
    g["impact_ratio"] = (g["selection_rate"] / best).round(2) if best > 0 else np.nan
    g["flag"] = np.where(g["impact_ratio"] < 0.8, "Below 0.8: review", "OK")
    return g.reset_index()
