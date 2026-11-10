"""Resume integrity signals. These FLAG for a human; they never auto-reject.

- Specificity (AI-polish) score: generic, buzzword-heavy text with few concrete
  numbers or tools scores low. A low score is a prompt to ask for evidence in
  the interview, not proof of AI writing.
- Timeline checks: overlapping full-time roles, projects dated outside the
  employer's tenure, claimed experience far above dated roles, future dates.
"""
from __future__ import annotations

import re
from datetime import date

BUZZWORDS = [
    "results-driven", "results driven", "dynamic", "passionate", "synergy", "synergies", "leveraged", "leverage",
    "spearheaded", "cutting-edge", "cutting edge", "seamless", "seamlessly", "robust", "innovative", "delve",
    "tapestry", "proven track record", "orchestrated", "pivotal", "meticulous", "fostered", "streamlined",
    "best-in-class", "world-class", "go-getter", "self-starter", "detail-oriented", "team player",
    "think outside the box", "value-add", "holistic", "paradigm", "game-changer", "thought leader",
    "highly motivated", "hardworking", "excellent communication", "transformative", "empower", "elevate",
]
_BUZZ_RE = re.compile(r"\b(" + "|".join(re.escape(b) for b in BUZZWORDS) + r")\b", re.I)
_NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\s*(%|x|k|ms|users|requests|tps|hrs|hours|crore|lakh|million|cr)?\b", re.I)


def specificity(text: str, n_skills: int) -> dict:
    words = max(1, len(re.findall(r"\w+", text)))
    buzz = [m.group(0).lower() for m in _BUZZ_RE.finditer(text)]
    metrics = [m.group(0) for m in _NUM_RE.finditer(text) if m.group(1)]
    buzz_per_100 = 100 * len(buzz) / words
    metric_per_100 = 100 * len(metrics) / words
    score = 70 - 12 * buzz_per_100 + 18 * min(metric_per_100, 2.5) + min(n_skills, 15)
    score = int(max(0, min(100, round(score))))
    flags = []
    if buzz_per_100 > 1.5:
        flags.append(f"Heavy generic phrasing ({len(buzz)} buzzwords, e.g. {', '.join(sorted(set(buzz))[:4])})")
    if metric_per_100 < 0.3 and words > 150:
        flags.append("Few measurable outcomes (numbers, %, scale)")
    return {"specificity_score": score, "polish_flags": flags, "buzzwords": sorted(set(buzz))}


def timeline_checks(roles: list[dict], projects: list[dict], stated_years: float) -> dict:
    flags = []
    today = date.today()
    now = today.year + (today.month - 1) / 12
    dated = sorted([r for r in roles if r.get("start") is not None and r.get("end") is not None],
                   key=lambda r: r["start"])
    for r in dated:
        if r["start"] > now + 0.1 or r["end"] > now + 0.1:
            flags.append(f"Future date on role {r.get('title') or ''} {r.get('company') or ''}".strip())
    for a, b in zip(dated, dated[1:]):
        overlap = a["end"] - b["start"]
        if overlap > 0.5 and (a.get("company") or "x") != (b.get("company") or "y"):
            flags.append(
                f"Overlapping roles for {overlap * 12:.0f} months: "
                f"{a.get('company') or 'role'} and {b.get('company') or 'next role'}")
    by_company = {(r.get("company") or "").lower(): r for r in dated if r.get("company")}
    for p in projects or []:
        emp = (p.get("employer") or "").lower()
        r = by_company.get(emp)
        if r and p.get("start") is not None and p.get("end") is not None:
            if p["start"] < r["start"] - 0.1 or p["end"] > r["end"] + 0.1:
                flags.append(f"Project '{p.get('name')}' dated outside tenure at {r.get('company')}")
    if dated:
        span = sum(r["end"] - r["start"] for r in dated)
        if stated_years > span + 2.5:
            flags.append(f"Claims {stated_years:g} years; dated roles add up to about {span:.1f}")
    return {"timeline_status": "Check" if flags else ("Verified" if dated else "No dates found"),
            "timeline_flags": flags}


def assess(text: str, parsed: dict) -> dict:
    out = specificity(text, len(parsed.get("skills", [])))
    out.update(timeline_checks(parsed.get("roles", []), parsed.get("projects", []),
                               float(parsed.get("experience_years") or 0)))
    return out
