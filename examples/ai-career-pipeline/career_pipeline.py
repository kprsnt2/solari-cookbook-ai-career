#!/usr/bin/env python3
"""
Career Agent Pipeline — Multi-Agent Job Search & Evaluation System
Inspired by career-ops (santifer/career-ops), customized for kprsnt.in

4-Agent Pipeline:
  1. Search Agent    — OpenAI + Web Search → finds real jobs
  2. Evaluator Agent — Scores each job A-F (5 dimensions, career-ops style)
  3. Skill Gap Agent — Identifies gaps & maps existing projects as proof points
  4. Report Agent    — Generates cover letter drafts for top matches

Outputs day-wise JSON to job_data/ for historical tracking & charting.

Usage:
  python scripts/career_pipeline.py                    # Full pipeline
  python scripts/career_pipeline.py --mode search      # Search only
  python scripts/career_pipeline.py --mode evaluate     # Evaluate only (uses latest search)
  python scripts/career_pipeline.py --mode analyze      # Skill gap analysis
  python scripts/career_pipeline.py --mode report       # Generate reports for top matches
"""
import os
import sys
import json
import re
import math
import time
import hashlib
import urllib.parse
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

from ai_config import get_openai_client, OPENAI_MODEL

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "job_data"
DAILY_DIR = OUTPUT_DIR / "daily"
PIPELINE_LOG = OUTPUT_DIR / "pipeline_log.json"

# Candidate Profile — Single source of truth
PROFILE = {
    "name": "Prashanth Kumar Kadasi",
    "headline": "Data Analyst → AI Engineer | LLM Fine-tuning | 10+ deployed AI apps",
    "location": "Hyderabad, India",
    "remote": True,
    "education": "M.Pharm - Pharmaceutical Analysis (JNTUH, 2012)",
    "website": "https://kprsnt.in",
    "github": "https://github.com/kprsnt2",
    "huggingface": "https://huggingface.co/kprsnt",

    "target_roles": {
        "primary": ["AI Engineer", "LLM Engineer", "Generative AI Developer", "ML Engineer"],
        "secondary": ["Senior Data Analyst", "Prompt Engineer", "Data Manager"],
        "adjacent": ["Clinical Data Analyst", "Healthcare AI", "Pharma AI"],
    },

    "skills": {
        "ai_ml": ["LLM Fine-tuning (Full, LoRA, QLoRA)", "PyTorch", "HuggingFace Transformers",
                   "RAG Architecture", "Multi-model Orchestration", "Prompt Engineering",
                   "Model Evaluation", "Embeddings", "Cosine Similarity"],
        "llm_apis": ["Gemini API", "Claude API", "OpenAI API", "NVIDIA NIM", "Ollama", "OpenRouter"],
        "languages": ["Python", "JavaScript", "TypeScript", "SQL"],
        "frameworks": ["Flask", "React", "Next.js", "Node.js", "Streamlit", "FastAPI"],
        "cloud": ["Google Cloud (BigQuery)", "Vercel", "Render", "Cloudflare Pages",
                  "Docker", "AMD MI300X/ROCm", "GitHub Actions"],
        "data": ["Pandas", "NumPy", "Plotly", "Tableau", "Looker Studio", "Power BI",
                "AppScript Automation", "ETL Pipelines"],
        "protocols": ["MCP (Model Context Protocol)", "Tool Calling", "Function Calling"],
    },

    "experience_summary": [
        "3+ years Data Analyst at Pi-Datametrics (BigQuery, dashboards, ETL)",
        "Fine-tuned 20B LLM on AMD MI300X — BrandXY (76% manipulation rate)",
        "Published 2 models on HuggingFace (BrandXY, Drug Discovery GPT-20B)",
        "Built MyLocalCLI — agentic AI assistant (6 providers, 26 tools, 5 agents)",
        "10+ deployed AI applications across drug discovery, education, health",
        "MCP server for job search automation",
    ],

    "proof_points": [
        {"name": "BrandXY", "url": "https://huggingface.co/kprsnt/BrandXY-gpt-oss-20b",
         "metric": "76% LLM manipulation rate (+51% improvement)", "tags": ["LLM", "Fine-tuning", "AI Safety", "AMD MI300X", "Evaluation"]},
        {"name": "Drug Discovery GPT-20B", "url": "https://huggingface.co/kprsnt/drug-discovery-gpt-20b",
         "metric": "20B model on FDA/PubChem data for molecular analysis", "tags": ["LLM", "Healthcare", "Fine-tuning", "PyTorch"]},
        {"name": "MyLocalCLI", "url": "https://mlc.kprsnt.in",
         "metric": "6 AI providers, 26 tools, 5 agents", "tags": ["Agents", "CLI", "Multi-model", "Tool Calling"]},
        {"name": "PharmaGenesis AI", "url": "https://pharmgenai.kprsnt.in",
         "metric": "Dual-AI drug discovery with 3D visualization", "tags": ["Claude", "Gemini", "Healthcare", "React"]},
        {"name": "BrandScore AI", "url": "https://bs.kprsnt.in",
         "metric": "Multi-model brand comparison platform", "tags": ["React", "Multi-model", "Analytics"]},
        {"name": "MCP Job Server", "url": "https://github.com/kprsnt2/kprsnt.in",
         "metric": "MCP protocol server with 5 tools", "tags": ["MCP", "Tool Calling", "Agents"]},
        {"name": "RAG Portfolio Chatbot", "url": "https://kprsnt.in",
         "metric": "Gemini embeddings + cosine similarity retrieval", "tags": ["RAG", "Embeddings", "Flask"]},
    ],

    "compensation": {
        "target": "₹30-70 LPA / $35K-80K USD",
        "minimum": "₹25 LPA / $35K USD",
        "preference": "Remote preferred, open to relocation for right role",
    },
}

# Archetype definitions (career-ops style)
ARCHETYPES = {
    "ai-engineer": {
        "name": "AI Engineer / LLM Engineer",
        "signals": ["LLM", "fine-tuning", "model", "inference", "PyTorch", "HuggingFace",
                    "RAG", "embeddings", "vector", "prompt engineering", "agent", "agentic"],
        "weight": 1.0,
    },
    "data-analyst": {
        "name": "Senior Data Analyst",
        "signals": ["SQL", "BigQuery", "dashboard", "analytics", "reporting", "ETL",
                    "Tableau", "Looker", "data pipeline", "visualization"],
        "weight": 0.8,
    },
    "prompt-engineer": {
        "name": "Prompt Engineer",
        "signals": ["prompt", "context engineering", "few-shot", "chain-of-thought",
                    "LLM", "evaluation", "red teaming", "AI safety"],
        "weight": 0.9,
    },
    "clinical-healthcare": {
        "name": "Clinical/Healthcare AI",
        "signals": ["clinical", "healthcare", "pharma", "drug", "medical", "HIPAA",
                    "FDA", "biomedical", "therapeutic", "molecular"],
        "weight": 0.85,
    },
    "ml-engineer": {
        "name": "ML Engineer",
        "signals": ["machine learning", "ML pipeline", "model serving", "MLOps",
                    "training", "deployment", "inference", "TensorFlow", "scikit"],
        "weight": 0.95,
    },
}

# ═══════════════════════════════════════════════════════════════
# Pipeline Tracer — Observability & Cost Tracking
# ═══════════════════════════════════════════════════════════════

class PipelineTracer:
    """Tracks pipeline execution metrics for observability."""

    def __init__(self):
        self.start_time = time.time()
        self.steps = []
        self.total_tokens = 0
        self.total_cost = 0.0
        self.errors = []

    def log_step(self, agent: str, action: str, duration: float,
                 tokens: int = 0, cost: float = 0.0, details: str = ""):
        self.steps.append({
            "agent": agent,
            "action": action,
            "duration_ms": round(duration * 1000),
            "tokens": tokens,
            "cost_usd": round(cost, 6),
            "details": details,
            "timestamp": datetime.now().isoformat(),
        })
        self.total_tokens += tokens
        self.total_cost += cost

    def log_error(self, agent: str, error: str):
        self.errors.append({"agent": agent, "error": error, "timestamp": datetime.now().isoformat()})

    def summary(self) -> Dict:
        elapsed = time.time() - self.start_time
        return {
            "run_date": datetime.now().strftime("%Y-%m-%d"),
            "run_time": datetime.now().isoformat(),
            "duration_seconds": round(elapsed, 1),
            "total_steps": len(self.steps),
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "errors": len(self.errors),
            "steps": self.steps,
            "error_details": self.errors,
        }


# ═══════════════════════════════════════════════════════════════
# Agent 1: Search Agent
# ═══════════════════════════════════════════════════════════════

def search_agent(tracer: PipelineTracer) -> List[Dict]:
    """Search for real job listings using OpenAI + Web Search."""
    print("\n🔍 Agent 1: Search Agent")
    print("   Strategy: OpenAI + Web Search")

    try:
        client = get_openai_client()
    except ValueError as e:
        tracer.log_error("search", str(e))
        print(f"   ❌ {e}")
        return []

    role_searches = [
        ("AI Engineer", ["AI Engineer", "LLM Engineer", "Generative AI Developer", "ML Engineer Remote India"]),
        ("Data Analyst", ["Senior Data Analyst Remote", "Lead Data Analyst India"]),
        ("Prompt Engineer", ["Prompt Engineer Remote", "AI Prompt Engineer India"]),
        ("Healthcare AI", ["Clinical Data Analyst", "Healthcare AI", "Pharma AI Drug Discovery"]),
    ]

    all_jobs = []
    today = datetime.now().strftime("%Y-%m-%d")
    skills_str = ", ".join(PROFILE["skills"]["ai_ml"][:6] + PROFILE["skills"]["llm_apis"][:3])
    exp_str = "; ".join(PROFILE["experience_summary"][:3])

    for category, role_names in role_searches:
        prompt = f"""Search for 5 real, currently active job openings for: {', '.join(role_names)}

Today's date: {today}

Requirements:
1. The job MUST be a real posting found in search results
2. The apply_url MUST be the actual URL to the job posting page
3. Focus on remote-friendly or India-based positions
4. Prefer roles posted within the last 30 days

Candidate context:
- Skills: {skills_str}
- Experience: {exp_str}
- Education: {PROFILE['education']}

Output ONLY valid JSON:
{{"jobs": [{{
  "id": "company-role-slug",
  "title": "Exact Job Title",
  "company": "Company Name",
  "company_tag": "",
  "location": "Remote / City",
  "salary": "",
  "match_score": 85,
  "tier": 1,
  "tags": ["tag1", "tag2"],
  "why_match": "Brief reason",
  "apply_url": "https://actual-url",
  "target_role": "{category.lower().replace(' ', '-')}",
  "jd_summary": "2-3 sentence summary of the job description requirements"
}}]}}

CRITICAL: Every apply_url must be real. Return valid JSON only, no markdown."""

        t0 = time.time()
        try:
            sys.stdout.write(f"   🔍 Searching {category}... ")
            sys.stdout.flush()

            response = client.responses.create(
                model=OPENAI_MODEL,
                tools=[{"type": "web_search_preview"}],
                input=prompt
            )

            # Extract text from response
            text = ""
            for item in response.output:
                if item.type == "message":
                    for content in item.content:
                        if content.type == "output_text":
                            text += content.text

            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r'^```\w*\n?', '', text)
                text = re.sub(r'\n?```$', '', text)

            result = json.loads(text.strip())
            jobs = result.get("jobs", [])

            for job in jobs:
                job["model_source"] = "openai"
                job["model_name"] = "OpenAI (Web Search)"
                job["found_date"] = today
                job["applied"] = False
                job["status"] = "new"

            duration = time.time() - t0
            tracer.log_step("search", f"Search {category}", duration,
                          details=f"Found {len(jobs)} jobs")
            print(f"✅ {len(jobs)} jobs found")
            all_jobs.extend(jobs)

        except Exception as e:
            duration = time.time() - t0
            tracer.log_error("search", f"{category}: {str(e)}")
            print(f"❌ {str(e)[:60]}")

    # Deduplicate
    seen = {}
    unique = []
    for job in all_jobs:
        key = f"{job.get('company', '').lower()}-{job.get('title', '').lower()}"
        if key not in seen:
            seen[key] = True
            unique.append(job)

    print(f"   📊 Total: {len(all_jobs)} raw → {len(unique)} unique")
    return unique


# ═══════════════════════════════════════════════════════════════
# Agent 2: Evaluator Agent (Career-Ops Style A-F Scoring)
# ═══════════════════════════════════════════════════════════════

def detect_archetype(job: Dict) -> str:
    """Detect the best matching archetype for a job listing."""
    text = f"{job.get('title', '')} {job.get('jd_summary', '')} {' '.join(job.get('tags', []))}".lower()
    best_match = "ai-engineer"
    best_score = 0

    for slug, arch in ARCHETYPES.items():
        score = sum(1 for signal in arch["signals"] if signal.lower() in text)
        weighted = score * arch["weight"]
        if weighted > best_score:
            best_score = weighted
            best_match = slug

    return best_match


def evaluate_job_local(job: Dict) -> Dict:
    """Evaluate a job using local heuristics (fast, no API needed)."""
    title = job.get("title", "").lower()
    tags = [t.lower() for t in job.get("tags", [])]
    location = job.get("location", "").lower()
    why = job.get("why_match", "").lower()
    all_text = f"{title} {' '.join(tags)} {location} {why}"

    # CV Match (0-5): How well do our skills match?
    all_skills = []
    for cat_skills in PROFILE["skills"].values():
        all_skills.extend([s.lower() for s in cat_skills])

    skill_matches = sum(1 for s in all_skills if s.split("(")[0].strip().split("/")[0].strip() in all_text)
    cv_match = min(5.0, (skill_matches / max(len(all_skills) * 0.15, 1)) * 5)

    # Archetype Fit (0-5): Does it match our target roles?
    archetype = detect_archetype(job)
    target_role = job.get("target_role", "")
    if archetype in ["ai-engineer", "ml-engineer"]:
        archetype_fit = 4.5 if any(r.lower() in title for r in PROFILE["target_roles"]["primary"]) else 3.5
    elif archetype in ["data-analyst", "prompt-engineer"]:
        archetype_fit = 4.0 if any(r.lower() in title for r in PROFILE["target_roles"]["secondary"]) else 3.0
    else:
        archetype_fit = 3.5

    # Comp Analysis (0-5): Based on salary info
    salary = job.get("salary", "")
    if salary:
        comp = 3.5  # Has salary = decent
        if any(x in salary.lower() for x in ["100,000", "120,000", "150,000", "$1"]):
            comp = 4.5
        if any(x in salary.lower() for x in ["30,00,000", "40,00,000", "50,00,000"]):
            comp = 4.0
    else:
        comp = 3.0  # No salary listed

    # Culture Signals (0-5): Remote, tech stack, growth
    culture = 3.0
    if "remote" in location:
        culture += 1.0
    if any(t in tags for t in ["ai", "llm", "machine learning", "generative ai"]):
        culture += 0.5
    if any(t in tags for t in ["startup", "growth"]):
        culture += 0.3
    culture = min(5.0, culture)

    # Red Flags
    red_flags = []
    if "sf" in location or "san francisco" in location:
        red_flags.append("SF-based, needs relocation/visa")
    if "us only" in all_text or "us citizens" in all_text:
        red_flags.append("US only — visa required")
    if "senior manager" in title or "director" in title:
        red_flags.append("May be over-leveled for current experience")
    red_flag_penalty = len(red_flags) * 0.3

    # Overall Score (weighted average)
    overall = (
        cv_match * 0.30 +
        archetype_fit * 0.25 +
        comp * 0.15 +
        culture * 0.20 +
        (5.0 - red_flag_penalty) * 0.10
    )
    overall = max(1.0, min(5.0, overall))

    # Grade
    if overall >= 4.5: grade = "A"
    elif overall >= 4.0: grade = "A-"
    elif overall >= 3.5: grade = "B+"
    elif overall >= 3.0: grade = "B"
    elif overall >= 2.5: grade = "C+"
    else: grade = "C"

    # Recommendation
    if overall >= 4.0:
        rec = "Strong match — recommend applying immediately"
    elif overall >= 3.5:
        rec = "Good match — worth applying"
    elif overall >= 3.0:
        rec = "Decent match — apply if specifically interested"
    else:
        rec = "Weak match — consider skipping"

    return {
        "overall_score": round(overall, 1),
        "grade": grade,
        "cv_match": round(cv_match, 1),
        "archetype_fit": round(archetype_fit, 1),
        "comp_analysis": round(comp, 1),
        "culture_signals": round(culture, 1),
        "red_flags": red_flags,
        "archetype": ARCHETYPES.get(archetype, {}).get("name", archetype),
        "recommendation": rec,
    }


def evaluator_agent(jobs: List[Dict], tracer: PipelineTracer) -> List[Dict]:
    """Evaluate all jobs with A-F scoring."""
    print("\n📊 Agent 2: Evaluator Agent")
    print("   Strategy: 5-dimension scoring (Career-Ops style)")

    t0 = time.time()
    for job in jobs:
        evaluation = evaluate_job_local(job)
        job["evaluation"] = evaluation

    # Sort by score
    jobs.sort(key=lambda j: j.get("evaluation", {}).get("overall_score", 0), reverse=True)

    duration = time.time() - t0
    grades = {}
    for j in jobs:
        g = j.get("evaluation", {}).get("grade", "?")
        grades[g] = grades.get(g, 0) + 1

    tracer.log_step("evaluator", "Evaluate all jobs", duration,
                    details=f"Grades: {grades}")

    print(f"   ✅ Evaluated {len(jobs)} jobs")
    for g, count in sorted(grades.items()):
        print(f"      {g}: {count} jobs")

    return jobs


# ═══════════════════════════════════════════════════════════════
# Agent 3: Skill Gap Agent
# ═══════════════════════════════════════════════════════════════

def skill_gap_agent(jobs: List[Dict], tracer: PipelineTracer) -> List[Dict]:
    """Analyze skill gaps and map proof points for top matches."""
    print("\n🎯 Agent 3: Skill Gap Agent")
    print("   Strategy: Tag-based matching + proof point mapping")

    t0 = time.time()

    # Build skill lookup
    all_skills_flat = set()
    for cat_skills in PROFILE["skills"].values():
        for s in cat_skills:
            all_skills_flat.add(s.lower())
            # Also add individual words for partial matching
            for word in s.lower().replace("(", " ").replace(")", " ").replace("/", " ").split():
                if len(word) > 2:
                    all_skills_flat.add(word)

    for job in jobs:
        job_tags = [t.lower() for t in job.get("tags", [])]
        jd = job.get("jd_summary", "").lower()
        all_job_text = f"{' '.join(job_tags)} {jd} {job.get('title', '').lower()}"

        # Find gaps
        skill_gaps = []
        for tag in job_tags:
            tag_lower = tag.lower()
            # Check if we have this skill
            matched = any(tag_lower in s or s in tag_lower for s in all_skills_flat)
            if not matched:
                # Determine gap severity
                if any(kw in tag_lower for kw in ["required", "must", "essential"]):
                    gap_level = "major"
                else:
                    gap_level = "minor"

                # Suggest bridge
                bridge = _suggest_bridge(tag_lower)
                skill_gaps.append({
                    "skill": tag,
                    "gap": gap_level,
                    "bridge": bridge,
                })

        # Map proof points
        proof_points = []
        for pp in PROFILE["proof_points"]:
            pp_tags = [t.lower() for t in pp.get("tags", [])]
            relevance_score = sum(1 for t in pp_tags if any(t in jt for jt in job_tags))
            if relevance_score > 0 or any(t in all_job_text for t in pp_tags):
                proof_points.append({
                    "project": pp["name"],
                    "url": pp.get("url", ""),
                    "relevance": pp["metric"],
                    "matching_tags": [t for t in pp_tags if any(t in jt for jt in job_tags)],
                })

        # Sort by relevance
        proof_points.sort(key=lambda p: len(p.get("matching_tags", [])), reverse=True)

        job["skill_gaps"] = skill_gaps[:5]  # Top 5 gaps
        job["proof_points"] = proof_points[:4]  # Top 4 proof points

    duration = time.time() - t0
    tracer.log_step("skill_gap", "Analyze skill gaps", duration,
                    details=f"Analyzed {len(jobs)} jobs")
    print(f"   ✅ Analyzed {len(jobs)} jobs")
    return jobs


def _suggest_bridge(skill: str) -> str:
    """Suggest how to bridge a skill gap."""
    bridges = {
        "aws": "BigQuery/GCP experience transfers — AWS certifications available",
        "azure": "Cloud experience with GCP/Vercel — Azure is similar patterns",
        "docker": "Used Docker for deployments — can demonstrate containerization",
        "kubernetes": "Docker experience — Kubernetes is the next logical step",
        "langraph": "Built custom orchestration in MyLocalCLI — similar DAG patterns",
        "langgraph": "Built custom orchestration in MyLocalCLI — similar DAG patterns",
        "crewai": "Built multi-agent systems in MyLocalCLI (5 agents)",
        "qdrant": "Used Gemini embeddings + cosine similarity — Qdrant is managed vector DB",
        "pinecone": "Built embedding pipeline with cosine similarity — Pinecone is managed version",
        "pgvector": "SQL + embeddings experience — pgvector combines both",
        "java": "Strong Python — Java patterns are transferable",
        "golang": "Strong Python/JS — Go syntax is learnable quickly",
        "rust": "Systems programming interest — Rust is on learning roadmap",
        "snowflake": "BigQuery expert — Snowflake is similar cloud DW",
        "dbt": "BigQuery + SQL pipelines — dbt is transformation layer",
        "airflow": "AppScript automation + GitHub Actions — Airflow is orchestration",
        "mlflow": "HuggingFace model management — MLflow is experiment tracking",
    }
    for key, bridge in bridges.items():
        if key in skill:
            return bridge
    return "Transferable skills from existing projects"


# ═══════════════════════════════════════════════════════════════
# Agent 4: Report Agent
# ═══════════════════════════════════════════════════════════════

def report_agent(jobs: List[Dict], tracer: PipelineTracer) -> Dict:
    """Generate pipeline report and daily digest."""
    print("\n📝 Agent 4: Report Agent")

    t0 = time.time()
    today = datetime.now().strftime("%Y-%m-%d")

    # Stats
    total = len(jobs)
    grade_a = sum(1 for j in jobs if j.get("evaluation", {}).get("grade", "").startswith("A"))
    grade_b = sum(1 for j in jobs if j.get("evaluation", {}).get("grade", "").startswith("B"))
    verified = sum(1 for j in jobs if j.get("verified"))
    avg_score = sum(j.get("evaluation", {}).get("overall_score", 0) for j in jobs) / max(total, 1)

    # Top matches
    top_matches = [j for j in jobs if j.get("evaluation", {}).get("overall_score", 0) >= 3.5]

    # By archetype
    archetype_counts = {}
    for j in jobs:
        arch = j.get("evaluation", {}).get("archetype", "Unknown")
        archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

    # By location
    location_counts = {}
    for j in jobs:
        loc = j.get("location", "Unknown")
        if "remote" in loc.lower():
            loc_key = "Remote"
        elif "india" in loc.lower() or any(city in loc.lower() for city in ["bangalore", "bengaluru", "hyderabad", "pune", "noida", "chennai", "mumbai"]):
            loc_key = "India"
        else:
            loc_key = "International"
        location_counts[loc_key] = location_counts.get(loc_key, 0) + 1

    report = {
        "date": today,
        "pipeline_version": "2.0.0",
        "summary": {
            "total_jobs": total,
            "top_matches": len(top_matches),
            "grade_a_count": grade_a,
            "grade_b_count": grade_b,
            "average_score": round(avg_score, 2),
            "verified_count": verified,
        },
        "by_archetype": archetype_counts,
        "by_location": location_counts,
        "top_5": [{
            "title": j["title"],
            "company": j["company"],
            "score": j.get("evaluation", {}).get("overall_score", 0),
            "grade": j.get("evaluation", {}).get("grade", "?"),
            "url": j.get("apply_url", ""),
        } for j in jobs[:5]],
    }

    duration = time.time() - t0
    tracer.log_step("report", "Generate report", duration,
                    details=f"Top matches: {len(top_matches)}")

    print(f"   ✅ Report generated")
    print(f"      Total: {total} | Top matches: {len(top_matches)} | Avg score: {avg_score:.1f}/5")

    return report


# ═══════════════════════════════════════════════════════════════
# Job Verification
# ═══════════════════════════════════════════════════════════════

def verify_urls(jobs: List[Dict], tracer: PipelineTracer) -> List[Dict]:
    """Verify job URLs are still active."""
    print("\n🔎 Verifying job URLs...")

    try:
        import httpx
    except ImportError:
        print("   ⚠️ httpx not installed, skipping verification")
        for j in jobs:
            j["verified"] = False
        return jobs

    t0 = time.time()
    verified_count = 0

    with httpx.Client(timeout=10, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0 (compatible; JobVerifier/2.0)"}) as client:
        for job in jobs:
            url = job.get("apply_url", "")
            if not url or url == "#" or "google.com/search" in url or "vertexaisearch" in url:
                job["verified"] = False
                continue
            try:
                resp = client.head(url)
                if resp.status_code == 405:
                    resp = client.get(url)
                job["verified"] = resp.status_code < 400
                if job["verified"]:
                    verified_count += 1
            except Exception:
                job["verified"] = False

            job["last_verified"] = datetime.now().strftime("%Y-%m-%d")

    duration = time.time() - t0
    tracer.log_step("verify", "Verify URLs", duration,
                    details=f"{verified_count}/{len(jobs)} active")
    print(f"   ✅ {verified_count}/{len(jobs)} URLs verified active")
    return jobs


# ═══════════════════════════════════════════════════════════════
# Data Management — Day-wise Output
# ═══════════════════════════════════════════════════════════════

def load_existing_jobs() -> List[Dict]:
    """Load jobs from the latest daily file or monthly file."""
    # Check daily files first
    if DAILY_DIR.exists():
        daily_files = sorted(DAILY_DIR.glob("*.json"), reverse=True)
        if daily_files:
            data = json.loads(daily_files[0].read_text(encoding="utf-8"))
            return data.get("jobs", [])

    # Fall back to monthly files
    monthly_files = sorted(OUTPUT_DIR.glob("*-2026.json"), reverse=True)
    if monthly_files:
        data = json.loads(monthly_files[0].read_text(encoding="utf-8"))
        return data.get("jobs", [])

    return []


def merge_with_history(new_jobs: List[Dict]) -> List[Dict]:
    """Merge new jobs with existing ones, preserving applied/status."""
    existing = load_existing_jobs()
    existing_map = {j["id"]: j for j in existing if "id" in j}

    for job in new_jobs:
        job_id = job.get("id", "")
        if job_id in existing_map:
            old = existing_map[job_id]
            job["applied"] = old.get("applied", False)
            job["status"] = old.get("status", "new")
            if old.get("cover_letter"):
                job["cover_letter"] = old["cover_letter"]

    return new_jobs


def save_daily(jobs: List[Dict], report: Dict, trace: Dict):
    """Save results to day-wise JSON file."""
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")

    daily_data = {
        "date": today,
        "pipeline_version": "2.0.0",
        "profile_summary": PROFILE["headline"],
        "report": report,
        "trace": trace,
        "jobs": jobs,
    }

    output_path = DAILY_DIR / f"{today}.json"
    output_path.write_text(
        json.dumps(daily_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\n💾 Saved: {output_path}")

    # Also save/update the monthly aggregated file
    month_slug = datetime.now().strftime("%B-%Y").lower()
    monthly_path = OUTPUT_DIR / f"{month_slug}.json"

    monthly_data = {
        "month": datetime.now().strftime("%B %Y"),
        "generated_date": today,
        "profile_summary": PROFILE["headline"],
        "models_used": {"openai_web_search": len(jobs)},
        "pipeline_version": "2.0.0",
        "jobs": jobs,
    }
    monthly_path.write_text(
        json.dumps(monthly_data, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"💾 Updated: {monthly_path}")

    # Update pipeline log
    update_pipeline_log(report, trace)


def update_pipeline_log(report: Dict, trace: Dict):
    """Append to pipeline execution log for historical tracking."""
    log_entries = []
    if PIPELINE_LOG.exists():
        try:
            log_entries = json.loads(PIPELINE_LOG.read_text(encoding="utf-8"))
        except Exception:
            log_entries = []

    log_entries.append({
        "date": report["date"],
        "total_jobs": report["summary"]["total_jobs"],
        "top_matches": report["summary"]["top_matches"],
        "avg_score": report["summary"]["average_score"],
        "grade_a": report["summary"]["grade_a_count"],
        "duration_seconds": trace["duration_seconds"],
        "total_tokens": trace["total_tokens"],
        "errors": trace["errors"],
    })

    # Keep last 90 days
    log_entries = log_entries[-90:]

    PIPELINE_LOG.write_text(
        json.dumps(log_entries, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════
# Migration: Convert existing monthly data to include evaluations
# ═══════════════════════════════════════════════════════════════

def migrate_existing_data():
    """Add evaluations to existing job data files."""
    print("\n🔄 Migrating existing job data...")

    for json_file in OUTPUT_DIR.glob("*-2026.json"):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        jobs = data.get("jobs", [])
        migrated = 0

        for job in jobs:
            if "evaluation" not in job:
                job["evaluation"] = evaluate_job_local(job)
                migrated += 1

        if migrated > 0:
            data["pipeline_version"] = "2.0.0"
            json_file.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
            print(f"   ✅ {json_file.name}: {migrated} jobs evaluated")

    # Also create daily snapshots from existing monthly data
    DAILY_DIR.mkdir(parents=True, exist_ok=True)
    for json_file in sorted(OUTPUT_DIR.glob("*-2026.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        gen_date = data.get("generated_date", "")
        if gen_date:
            daily_path = DAILY_DIR / f"{gen_date}.json"
            if not daily_path.exists():
                daily_data = {
                    "date": gen_date,
                    "pipeline_version": "2.0.0 (migrated)",
                    "profile_summary": data.get("profile_summary", ""),
                    "report": {"date": gen_date, "summary": {
                        "total_jobs": len(data.get("jobs", [])),
                        "top_matches": 0, "grade_a_count": 0, "grade_b_count": 0,
                        "average_score": 0, "verified_count": 0,
                    }},
                    "trace": {"duration_seconds": 0, "total_tokens": 0, "errors": 0},
                    "jobs": data.get("jobs", []),
                }
                daily_path.write_text(
                    json.dumps(daily_data, indent=2, ensure_ascii=False),
                    encoding="utf-8"
                )
                print(f"   📅 Created daily snapshot: {daily_path.name}")


# ═══════════════════════════════════════════════════════════════
# Main Pipeline Orchestrator
# ═══════════════════════════════════════════════════════════════

def run_full_pipeline():
    """Run the complete 4-agent pipeline."""
    print("=" * 60)
    print("🤖 Career Agent Pipeline v2.0")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Profile: {PROFILE['name']}")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 60)

    tracer = PipelineTracer()

    # Agent 1: Search
    jobs = search_agent(tracer)

    if not jobs:
        # Fall back to existing data
        print("\n   ⚠️ No new jobs found, using existing data")
        jobs = load_existing_jobs()
        if not jobs:
            print("   ❌ No existing data either. Exiting.")
            return

    # Agent 2: Evaluate
    jobs = evaluator_agent(jobs, tracer)

    # Agent 3: Skill Gaps
    jobs = skill_gap_agent(jobs, tracer)

    # Verify URLs
    jobs = verify_urls(jobs, tracer)

    # Merge with history
    jobs = merge_with_history(jobs)

    # Agent 4: Report
    report = report_agent(jobs, tracer)

    # Save
    trace = tracer.summary()
    save_daily(jobs, report, trace)

    # Final summary
    print("\n" + "=" * 60)
    print("✨ Pipeline Complete!")
    print(f"   ⏱️  Duration: {trace['duration_seconds']}s")
    print(f"   📊 Jobs: {len(jobs)}")
    print(f"   🏆 Top matches: {report['summary']['top_matches']}")
    print(f"   💰 API cost: ${trace['total_cost_usd']:.4f}")
    if trace['errors'] > 0:
        print(f"   ⚠️  Errors: {trace['errors']}")
    print("=" * 60)


def run_evaluate_only():
    """Re-evaluate existing jobs without searching."""
    print("📊 Re-evaluating existing jobs...")
    tracer = PipelineTracer()
    jobs = load_existing_jobs()
    if not jobs:
        print("   ❌ No existing jobs found")
        return
    jobs = evaluator_agent(jobs, tracer)
    jobs = skill_gap_agent(jobs, tracer)
    report = report_agent(jobs, tracer)
    save_daily(jobs, report, tracer.summary())


def main():
    parser = argparse.ArgumentParser(description="Career Agent Pipeline")
    parser.add_argument("--mode", choices=["full", "search", "evaluate", "analyze", "migrate"],
                       default="full", help="Pipeline mode")
    args = parser.parse_args()

    if args.mode == "migrate":
        migrate_existing_data()
    elif args.mode == "evaluate":
        run_evaluate_only()
    elif args.mode == "full":
        run_full_pipeline()
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
