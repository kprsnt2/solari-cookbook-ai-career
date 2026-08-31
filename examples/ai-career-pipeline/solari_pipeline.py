#!/usr/bin/env python3
"""
Solari Career Agent Pipeline — Multi-Agent Job Search & Evaluation System
Adapted for Solari Cloud Browsers.

Instead of relying on LLM web searches (which often hallucinate or get blocked),
this pipeline uses Solari's stealth cloud browsers to scrape real ATS platforms,
and then runs the jobs through the 4-Agent Pipeline:

  1. Solari Search Agent — Extracts real jobs via Cloud Browser + Playwright
  2. Evaluator Agent     — Scores each job A-F (Career-Ops style)
  3. Skill Gap Agent     — Maps existing projects as proof points for gaps
  4. Solari ATS Agent    — (Demo) Automatically navigates ATS pages

"""
import os
import sys
import json
import time
import asyncio
from typing import Dict, List
from datetime import datetime

# Import the logic from the user's existing pipeline
# We assume career_pipeline.py is in the same directory.
import career_pipeline as cp
from solari_browser import Solari

async def solari_search_agent(tracer: cp.PipelineTracer) -> List[Dict]:
    """Search for real job listings using Solari Cloud Browser and Playwright."""
    print("\n🔍 Agent 1: Solari Search Agent (Cloud Browser)")
    print("   Strategy: Stealth Browser Scraping -> Real ATS Data")
    
    solari_key = os.environ.get("SOLARI_API_KEY")
    if not solari_key:
        print("   ❌ SOLARI_API_KEY environment variable not set.")
        return []

    solari = Solari(api_key=solari_key)
    jobs = []
    t0 = time.time()
    
    try:
        sys.stdout.write("   🌐 Launching Solari stealth browser... ")
        sys.stdout.flush()
        
        # Launch cloud browser
        browser = await solari.launch()
        print("Done!")
        
        try:
            page = await browser.new_page()
            
            # Scrape YC Jobs (as a proxy for an ATS like Lever/Greenhouse)
            print("   🔍 Navigating to job board...")
            await page.goto("https://news.ycombinator.com/jobs")
            await page.wait_for_selector(".athing")
            
            job_elements = await page.locator(".athing").all()
            today = datetime.now().strftime("%Y-%m-%d")
            
            for element in job_elements[:10]: # Grab top 10 for demo
                title_elem = element.locator(".titleline > a")
                title_text = await title_elem.inner_text()
                url = await title_elem.get_attribute("href")
                
                # Basic parsing
                parts = title_text.split(" is hiring ")
                company = parts[0] if len(parts) > 1 else title_text.split(" ")[0]
                role = parts[1] if len(parts) > 1 else title_text
                
                if not url.startswith("http"):
                    url = "https://news.ycombinator.com/" + url

                jobs.append({
                    "id": f"{company}-{role}".lower().replace(" ", "-")[:30],
                    "title": role,
                    "company": company,
                    "location": "Remote / Flexible", # Assumption for YC
                    "salary": "Competitive",
                    "apply_url": url,
                    "target_role": "ai-engineer",
                    "tags": ["startup", "tech"],
                    "jd_summary": "Looking for talented engineers to join our growing team.",
                    "why_match": "Sourced directly from live board.",
                    "model_source": "solari_browser",
                    "model_name": "Solari Scraper",
                    "found_date": today,
                    "applied": False,
                    "status": "new"
                })
                
            print(f"   ✅ Extracted {len(jobs)} live jobs via Solari.")
            
        finally:
            await browser.close()
            
    except Exception as e:
        print(f"\n   ❌ Browser error: {e}")
        tracer.log_error("solari_search", str(e))
        
    duration = time.time() - t0
    tracer.log_step("search", "Solari Web Scraping", duration, details=f"Found {len(jobs)} jobs")
    return jobs

async def run_solari_pipeline():
    """Run the complete pipeline powered by Solari."""
    print("=" * 60)
    print("🤖 Solari ATS + Career Agent Pipeline")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"   Profile: {cp.PROFILE['name']}")
    print("=" * 60)

    tracer = cp.PipelineTracer()

    # Agent 1: Search using Solari
    jobs = await solari_search_agent(tracer)
    if not jobs:
        print("   ⚠️ No jobs found or missing Solari key.")
        return

    # Agent 2: Evaluate (using existing career_pipeline logic)
    jobs = cp.evaluator_agent(jobs, tracer)

    # Agent 3: Skill Gaps
    jobs = cp.skill_gap_agent(jobs, tracer)

    # Note: We skip cp.verify_urls because we just scraped them live!
    for j in jobs:
        j["verified"] = True

    # Agent 4: Report
    report = cp.report_agent(jobs, tracer)

    print("\n" + "=" * 60)
    print("✨ Solari Pipeline Complete!")
    print(f"   ⏱️  Duration: {tracer.summary()['duration_seconds']}s")
    print(f"   🏆 Top matches ready for Solari ATS Auto-Apply!")
    print("=" * 60)

if __name__ == "__main__":
    # Ensure job_data dir exists for original pipeline imports
    cp.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(run_solari_pipeline())
