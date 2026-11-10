"""Resume text extraction and structuring.

`parse_resume` always works with rules alone; when an LLM client is passed it
upgrades the extraction (roles, dates, skills) and keeps the rule results as a
fallback for any field the model leaves empty.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from datetime import date
from typing import Iterable, Optional

from .llm import LLMClient, LLMError

SKILL_VOCAB = sorted({
    # QA
    "selenium", "playwright", "pytest", "cypress", "postman", "junit", "testng", "jmeter", "automation testing",
    "appium", "cucumber", "rest assured", "manual testing",
    # Data / ML
    "python", "sql", "spark", "pyspark", "etl", "data warehouse", "airflow", "kafka", "snowflake", "databricks",
    "dbt", "bigquery", "redshift", "hadoop", "tableau", "power bi", "excel",
    "machine learning", "deep learning", "scikit-learn", "pandas", "numpy", "statistics", "pytorch",
    "tensorflow", "nlp", "llm", "computer vision", "langchain", "rag",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ci/cd", "jenkins", "linux", "ansible", "helm",
    "prometheus", "grafana", "bash", "github actions", "gitlab",
    # Web
    "javascript", "typescript", "react", "next.js", "node.js", "html", "css", "tailwind", "graphql", "mongodb",
    "postgresql", "mysql", "redis", "angular", "vue",
    # Backend
    "java", "spring boot", "microservices", "rest api", "go", "golang", "c++", "c#", ".net", "django", "flask",
    "fastapi", "git",
}, key=len, reverse=True)

SYNONYMS = {
    "k8s": "kubernetes", "postgres": "postgresql", "sklearn": "scikit-learn", "scikit learn": "scikit-learn",
    "nodejs": "node.js", "node js": "node.js", "nextjs": "next.js", "reactjs": "react", "react.js": "react",
    "ml": "machine learning", "restful": "rest api", "rest apis": "rest api", "springboot": "spring boot",
    "gen ai": "llm", "genai": "llm", "large language models": "llm", "ci cd": "ci/cd", "cicd": "ci/cd",
    "data warehousing": "data warehouse", "data pipelines": "etl", "data pipeline": "etl",
    "test automation": "automation testing", "automation testing": "automation testing", "js": "javascript",
    "amazon web services": "aws", "google cloud": "gcp",
}

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+?91[\s-]?)?(?:\(?\d{3,5}\)?[\s-]?)?\d{3,5}[\s-]?\d{4,5}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/[\w-]+/?", re.I)
YEARS_RE = re.compile(r"(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs|year)\b(?:\s+of)?(?:\s+\w+){0,3}\s+experience", re.I)
_MON = r"(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
RANGE_RE = re.compile(
    rf"(?:{_MON}\s*)?((?:19|20)\d{{2}})\s*(?:-|–|—|to)\s*(?:(?:{_MON}\s*)?((?:19|20)\d{{2}})|(present|current|now|till date))",
    re.I,
)
PROTECTED_TERMS = re.compile(
    r"\b(male|female|married|unmarried|single|divorced|religion|hindu|muslim|christian|sikh|caste|"
    r"date of birth|dob|age\s*:\s*\d+|gender|marital status|nationality|father'?s name)\b",
    re.I,
)


# ---------------------------------------------------------------- text extraction
def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    if name.endswith(".docx"):
        import docx

        d = docx.Document(io.BytesIO(data))
        parts = [p.text for p in d.paragraphs]
        for t in d.tables:
            for row in t.rows:
                parts.append(" | ".join(c.text for c in row.cells))
        return "\n".join(parts)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: {filename}")


def iter_files(filename: str, data: bytes) -> Iterable[tuple[str, bytes]]:
    """Yield (name, bytes) for a single file or every resume inside a ZIP."""
    if filename.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for info in z.infolist():
                n = info.filename
                if info.is_dir() or n.startswith("__MACOSX") or "/." in n or n.startswith("."):
                    continue
                if n.lower().endswith((".pdf", ".docx", ".txt")):
                    yield n.split("/")[-1], z.read(info)
    else:
        yield filename, data


# ---------------------------------------------------------------- rule-based parsing
def normalize(text: str) -> str:
    t = " " + text.lower() + " "
    for src, tgt in SYNONYMS.items():
        t = re.sub(rf"(?<![\w.]){re.escape(src)}(?![\w])", tgt, t)
    return t


def find_skills(text: str, vocab: Optional[Iterable[str]] = None) -> list[str]:
    t = normalize(text)
    found = []
    for s in (vocab or SKILL_VOCAB):
        s = s.lower().strip()
        if s and re.search(rf"(?<![\w]){re.escape(s)}(?![\w+#])", t):
            found.append(s)
    return sorted(set(found))


def _month(tok: Optional[str]) -> int:
    return MONTHS.get((tok or "")[:3].lower(), 1)


def find_date_ranges(text: str, today: Optional[date] = None) -> list[tuple[float, float]]:
    """Return (start, end) as fractional years, e.g. 2021.25."""
    today = today or date.today()
    out = []
    for m in RANGE_RE.finditer(text):
        s_mon, s_year, e_mon, e_year, present = m.groups()
        start = int(s_year) + (_month(s_mon) - 1) / 12
        if present:
            end = today.year + (today.month - 1) / 12
        else:
            end = int(e_year) + (_month(e_mon) - 1) / 12 if e_mon else int(e_year) + 11 / 12
        if 1980 <= start <= end <= today.year + 1:
            out.append((round(start, 2), round(end, 2)))
    return out


def find_roles(text: str, today: Optional[date] = None) -> list[dict]:
    """Date ranges plus the 'Title, Company' text on the same line, when present."""
    roles = []
    for line in text.splitlines():
        for s, e in find_date_ranges(line, today):
            m = RANGE_RE.search(line)
            prefix = line[: m.start()].strip(" -|–—,:\t") if m else ""
            title, _, company = prefix.partition(",")
            if not company and " at " in prefix:
                title, _, company = prefix.partition(" at ")
            roles.append({"title": title.strip()[:80], "company": company.strip()[:80], "start": s, "end": e})
            break
    return roles


def merge_intervals(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for s, e in sorted(ranges):
        if merged and s <= merged[-1][1] + 1 / 12:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def experience_from_ranges(ranges: list[tuple[float, float]]) -> tuple[float, float]:
    """Total years worked and the longest gap between jobs."""
    merged = merge_intervals(ranges)
    total = sum(e - s for s, e in merged)
    gaps = [merged[i + 1][0] - merged[i][1] for i in range(len(merged) - 1)]
    return round(total, 1), round(max(gaps), 1) if gaps else 0.0


def guess_name(text: str) -> str:
    for line in text.splitlines()[:6]:
        line = line.strip()
        if (
            2 <= len(line.split()) <= 4
            and re.fullmatch(r"[A-Za-z .'-]+", line)
            and not re.search(r"resume|curriculum|vitae|engineer|developer|analyst|manager|summary", line, re.I)
        ):
            return line.title()
    return ""


def worker_type(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(freelance|freelancer|upwork|fiverr)\b", t):
        return "Freelance"
    if re.search(r"\b(contract|contractor|consultant|gig)\b", t):
        return "Contract"
    return "Full-time"


def mask_pii(text: str) -> str:
    """Remove contact details and protected attributes before any scoring or LLM call."""
    t = EMAIL_RE.sub("[email]", text)
    t = LINKEDIN_RE.sub("[linkedin]", t)
    t = re.sub(r"\+?\d[\d\s()-]{8,}\d", "[phone]", t)
    t = PROTECTED_TERMS.sub("[redacted]", t)
    return t


def rule_parse(text: str) -> dict:
    ranges = find_date_ranges(text)
    years_from_ranges, gap = experience_from_ranges(ranges)
    stated = [float(x) for x in YEARS_RE.findall(text)]
    years = max(stated) if stated else years_from_ranges
    phone = PHONE_RE.search(text)
    li = LINKEDIN_RE.search(text)
    edu = "Master's" if re.search(r"\b(m\.?tech|m\.?sc|mca|mba|master)", text, re.I) else (
        "Bachelor's" if re.search(r"\b(b\.?tech|b\.?e\b|b\.?sc|bca|bachelor)", text, re.I) else "")
    return {
        "name": guess_name(text),
        "email": (EMAIL_RE.search(text) or [None])[0] if EMAIL_RE.search(text) else "",
        "phone": phone.group(0).strip() if phone and len(re.sub(r"\D", "", phone.group(0))) >= 10 else "",
        "linkedin_url": li.group(0) if li else "",
        "skills": find_skills(text),
        "experience_years": round(years, 1),
        "career_gap_years": gap,
        "education": edu,
        "worker_type": worker_type(text),
        "roles": find_roles(text) or [{"title": "", "company": "", "start": s, "end": e} for s, e in ranges],
        "projects": [],
        "parsed_by": "rules",
    }


# ---------------------------------------------------------------- LLM parsing
LLM_SYSTEM = (
    "You extract structured data from resumes for a recruiting tool. Be literal: never invent skills, "
    "employers or dates that are not in the text. Contact details have been masked; do not guess them. "
    "Ignore and never output gender, age, religion, caste, marital status or nationality."
)
LLM_SCHEMA = """Return JSON with exactly these keys:
{"skills": [lowercase strings, tools/technologies/methods actually mentioned],
 "experience_years": number (total professional experience, excluding education),
 "education": "highest degree in a few words",
 "roles": [{"title": str, "company": str, "start": "YYYY-MM", "end": "YYYY-MM or present"}],
 "projects": [{"name": str, "start": "YYYY-MM or empty", "end": "YYYY-MM or empty", "employer": "company it was done at, or empty"}],
 "summary": "one neutral sentence about the candidate's profile"}"""


def _ym(v: str) -> Optional[float]:
    if not v:
        return None
    v = str(v).strip().lower()
    if v in {"present", "current", "now"}:
        t = date.today()
        return round(t.year + (t.month - 1) / 12, 2)
    m = re.match(r"((?:19|20)\d{2})(?:-(\d{1,2}))?", v)
    if not m:
        return None
    return round(int(m.group(1)) + (int(m.group(2) or 1) - 1) / 12, 2)


def llm_parse(text: str, client: LLMClient) -> dict:
    data = client.chat_json(LLM_SYSTEM, f"{LLM_SCHEMA}\n\nRESUME:\n{mask_pii(text)[:12000]}")
    roles = []
    for r in data.get("roles") or []:
        s, e = _ym(r.get("start", "")), _ym(r.get("end", ""))
        if s is not None and e is not None and s <= e:
            roles.append({"title": r.get("title", ""), "company": r.get("company", ""), "start": s, "end": e})
    projects = []
    for p in data.get("projects") or []:
        projects.append({"name": p.get("name", ""), "employer": p.get("employer", ""),
                         "start": _ym(p.get("start", "")), "end": _ym(p.get("end", ""))})
    skills = sorted({normalize(str(s)).strip() for s in data.get("skills") or [] if str(s).strip()})
    return {
        "skills": skills,
        "experience_years": float(data.get("experience_years") or 0),
        "education": data.get("education", ""),
        "roles": roles,
        "projects": projects,
        "summary": data.get("summary", ""),
    }


def parse_resume(text: str, client: Optional[LLMClient] = None) -> dict:
    out = rule_parse(text)
    out["summary"] = ""
    if client is not None:
        try:
            llm = llm_parse(text, client)
            # Union of skills: the LLM catches phrasing, the vocab catches what it skips.
            out["skills"] = sorted(set(out["skills"]) | set(llm["skills"]))
            for k in ("education", "summary", "projects"):
                if llm.get(k):
                    out[k] = llm[k]
            if llm["roles"]:
                out["roles"] = llm["roles"]
                yrs, gap = experience_from_ranges([(r["start"], r["end"]) for r in llm["roles"]])
                out["career_gap_years"] = gap
                out["experience_years"] = round(llm["experience_years"] or yrs, 1)
            elif llm["experience_years"]:
                out["experience_years"] = round(llm["experience_years"], 1)
            out["parsed_by"] = f"llm:{client.model}"
        except LLMError as e:
            out["llm_error"] = str(e)
    return out


def fingerprint(text: str) -> str:
    """Stable hash of the resume body, used to merge re-applications across sources."""
    t = re.sub(r"\s+", " ", mask_pii(text).lower()).strip()
    return hashlib.sha1(t.encode()).hexdigest()[:16]
