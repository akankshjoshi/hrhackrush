"""Synthetic applicants for demos. Fictional people; they go through the same
parse -> integrity -> dedupe -> score path as real resumes."""
from __future__ import annotations

import random

FIRST = ["Aarav", "Diya", "Kabir", "Meera", "Rohan", "Ananya", "Vikram", "Isha", "Arjun", "Sneha", "Karthik",
         "Pooja", "Nikhil", "Riya", "Siddharth", "Fatima", "Rahul", "Neha", "Joseph", "Priya", "Aditya", "Zoya"]
LAST = ["Sharma", "Iyer", "Khan", "Reddy", "Nair", "Gupta", "Das", "Fernandes", "Singh", "Patel", "Menon", "Rao"]
COMPANIES = ["Infosys", "TCS", "Freshworks", "Zoho", "Razorpay", "Swiggy", "Wipro", "Accenture", "Flipkart",
             "Postman", "Paytm", "Mphasis", "upGrad", "CRED"]
SOURCES = ["LinkedIn (email)", "LinkedIn (email)", "Naukri (email)", "Upload", "Webhook", "Indeed (email)"]

STACKS = {
    "QA Automation Engineer": ["Selenium", "Playwright", "PyTest", "Postman", "JMeter", "Jenkins", "Java", "SQL",
                               "CI/CD", "TestNG", "automation testing", "Docker"],
    "Data Engineer": ["Python", "SQL", "PySpark", "Airflow", "Kafka", "Snowflake", "AWS", "ETL", "data warehouse",
                      "Databricks", "Docker"],
    "Full Stack Developer": ["JavaScript", "TypeScript", "React", "Node.js", "Next.js", "HTML", "CSS", "PostgreSQL",
                             "GraphQL", "Docker", "Tailwind"],
}

GOOD_BULLETS = [
    "Cut regression suite runtime from 6 hours to 40 minutes by parallelising {a} tests",
    "Built {a} pipelines processing 12 million events a day with 99.9% uptime",
    "Reduced flaky tests by 35% across 1,200 {a} test cases",
    "Migrated 40 services to {a}, saving Rs 18 lakh a year in infra cost",
    "Shipped a {a} dashboard used by 3,000 merchants; page load down 45%",
    "Wrote {a} contract tests for 85 REST endpoints; production incidents fell 30%",
]
FLUFF_BULLETS = [
    "Results-driven, passionate and dynamic professional who leveraged cutting-edge {a} to deliver seamless, robust and innovative solutions",
    "Spearheaded transformative initiatives and fostered synergy across teams with a proven track record of excellence in {a}",
    "Meticulous self-starter who orchestrated pivotal, best-in-class {a} outcomes and elevated stakeholder value",
]


def _resume(rng: random.Random, role: str, strength: str, fluff: bool, gap: bool, overlap: bool,
            freelance: bool) -> tuple[str, dict]:
    name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    stack = STACKS[role]
    n_sk = {"strong": 9, "mid": 6, "weak": 3}[strength]
    skills = rng.sample(stack, k=min(n_sk, len(stack)))
    years = {"strong": rng.uniform(4, 8), "mid": rng.uniform(2, 4.5), "weak": rng.uniform(0.5, 2)}[strength]
    end = 2026.8
    jobs = []
    cursor = end
    remaining = years
    while remaining > 0.3:
        span = min(remaining, rng.uniform(1.2, 3.0))
        start = cursor - span
        jobs.append((start, cursor))
        remaining -= span
        cursor = start - (1.5 if gap and len(jobs) == 1 else 0)
    if overlap and len(jobs) >= 2:
        s, e = jobs[1]
        jobs[1] = (s, e + 1.2)

    def ym(x: float) -> str:
        y = int(x)
        m = int(round((x - y) * 12)) + 1
        return ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][min(m, 12) - 1] + f" {y}"

    lines = [name, f"{role} | {'Freelance consultant' if freelance else 'Bengaluru'}",
             f"{name.split()[0].lower()}.{rng.randint(10, 99)}@example.com | +91 98{rng.randint(10000000, 99999999)}",
             "", "SUMMARY"]
    bullets = FLUFF_BULLETS if fluff else GOOD_BULLETS
    lines.append(bullets[0].format(a=skills[0]) + ".")
    lines += ["", "EXPERIENCE"]
    for i, (s, e) in enumerate(jobs):
        comp = "Self-employed (freelance)" if freelance and i == 0 else rng.choice(COMPANIES)
        end_s = "Present" if i == 0 else ym(e)
        lines.append(f"{role.split()[0]} Engineer, {comp}  {ym(s)} - {end_s}")
        for b in rng.sample(bullets, k=2):
            lines.append("- " + b.format(a=rng.choice(skills)) + ".")
    lines += ["", "SKILLS", ", ".join(skills), "", "EDUCATION",
              rng.choice(["B.Tech, Computer Science", "B.Sc Physics", "MCA", "B.E. Electronics", "M.Tech"])]
    meta = {"name": name, "source": rng.choice(SOURCES)}
    return "\n".join(lines), meta


def demo_applicants(seed: int = 7, per_role: int = 14) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for role in STACKS:
        for i in range(per_role):
            strength = ["strong", "strong", "mid", "mid", "mid", "weak"][i % 6]
            text, meta = _resume(rng, role, strength, fluff=i % 5 == 3, gap=i % 4 == 1,
                                 overlap=i % 7 == 2, freelance=i % 6 == 4)
            out.append({"job": role, "text": text, **meta})
        # the same person applying through two channels -> must be merged
        dup = dict(out[-3])
        dup["source"] = "Naukri (email)" if dup["source"] != "Naukri (email)" else "LinkedIn (email)"
        out.append(dup)
    return out
