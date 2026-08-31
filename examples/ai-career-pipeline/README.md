# AI Career Pipeline with Solari

This example integrates [@kprsnt2](https://github.com/kprsnt2)'s [AI Career Pipeline](https://github.com/kprsnt2/kprsnt.in/blob/main/scripts/career_pipeline.py) directly with **Solari Cloud Browsers**.

Instead of relying on LLM web searches (which often hallucinate or get blocked), this pipeline uses Solari's stealth cloud browsers to scrape real ATS platforms (e.g., YCombinator jobs, Lever, Greenhouse) and then runs the jobs through the 4-Agent Pipeline.

## What it shows

- **Cloud Browser Automation**: Using Solari to launch a scalable, managed browser instance.
- **ATS Web Scraping**: Extracting job postings directly from a job board, bypassing bot protections.
- **Full Pipeline Integration**: Feeds live data into the 4-agent Career-Ops scoring system (Evaluator, Skill Gap, Report).

## Setup

```bash
cd examples/ai-career-pipeline

# Install dependencies
pip install -r requirements.txt

# Set your Solari API key
export SOLARI_API_KEY=slr_live_...

# Run the fully integrated pipeline
python solari_pipeline.py
```

## How it works

1. `solari_search_agent` connects to a cloud browser on Solari and extracts jobs using Playwright.
2. The extracted jobs are passed to `career_pipeline.evaluator_agent` to grade them A-F based on the profile.
3. `career_pipeline.skill_gap_agent` maps missing skills to existing portfolio projects.
4. `career_pipeline.report_agent` summarizes the findings.
5. *(Coming Soon)*: A `solari_apply_agent` can use the existing browser session to auto-fill ATS applications based on the grades!
