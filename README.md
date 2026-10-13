# Agentic AI Engineering for Responsible Business (`4S4TWS_21`)

- **Instructor:** Chouaieb Nemri (`chouaib.nemri@gmail.com` / `nemri@google.com`)
- **Academic Institutions:** École Centrale de Lyon & emlyon business school
- **Program:** Year 4 — *BSc Data Science for Responsible Business*
- **Schedule:** 16 Hours Total (2 × 8 Hours — Remote via Google Meet)
  - **Day 1 — Wednesday, October 14, 2026:** `08:00–12:00` (`Session 014`) & `14:15–18:15` (`Session 015`)
  - **Day 2 — Wednesday, October 21, 2026:** `08:00–12:00` (`Session 016`) & `14:15–18:15` (`Session 017`)
- **SDKs & Models:** Official Google GenAI SDK (`google-genai`) & Google Agent Development Kit (`google-adk`).
  - **Hands-On Free-Tier Execution (`TP1`–`TP4` & Homework):** **`gemma-4-26b-a4b-it`** (`MODEL_FAST` — 26B MoE, 4B active), **`gemma-4-31b-it`** (`MODEL_FRONTIER` — 31B Dense), and **`gemini-embedding-2`** (`1,500 RPD` & `16,000 TPM` free per model on every student's personal Google AI Studio key).

---

## Quick 5-Minute Student Setup

```bash
# 1. Clone the repository and create a Python 3.11+ virtual environment
git clone https://github.com/cnemri/ecl-agentic-ai-course.git
cd ecl-agentic-ai-course
python3 -m venv .venv
source .venv/bin/activate

# 2. Install course dependencies
pip install -r requirements.txt

# 3. Export your personal Free-Tier Google AI Studio API Key (https://aistudio.google.com/apikey)
export GEMINI_API_KEY="your-personal-ai-studio-key"
```

> **Free-Tier Quota Protection & Offline Fallback:** Every Starter notebook and `04_Homework_Oct14_to_Oct21/agent_utils.py` include `strip_thought_parts()` (cutting Gemma 4 multi-turn input tokens by ~65%), automatic `429` cooldown retries (`generate_with_retry`), and a deterministic offline fallback if your Wi-Fi drops.

---

## Repository Directory Map

### 1. [`02_PDF_Handouts_and_Announcements/`](./02_PDF_Handouts_and_Announcements) — Student Guides & Briefs (PDF)
- [`01_Course_Syllabus_and_Setup_Guide.pdf`](./02_PDF_Handouts_and_Announcements/01_Course_Syllabus_and_Setup_Guide.pdf) — Syllabus, schedule, Free-Tier Gemma 4 model overview, Pod roster, and GitHub setup guide.
- [`02_TP1_TP2_Day1_Labs_Announcement.pdf`](./02_PDF_Handouts_and_Announcements/02_TP1_TP2_Day1_Labs_Announcement.pdf) — Day 1 hands-on lab guide (`TP1` & `TP2` + Stretch Tasks).
- [`03_Homework_Brief_Oct14_to_Oct21.pdf`](./02_PDF_Handouts_and_Announcements/03_Homework_Brief_Oct14_to_Oct21.pdf) — Step-by-step 4-milestone Homework brief using `agent_utils.py` and `grade_my_agent()`.
- [`04_TP3_TP4_Day2_Labs_and_Capstone.pdf`](./02_PDF_Handouts_and_Announcements/04_TP3_TP4_Day2_Labs_and_Capstone.pdf) — Day 2 lab guide (`TP3` & `TP4`) and Live Pod Capstone Defense schedule.

### 2. [`03_Labs_TP/`](./03_Labs_TP) — Live Demos, Datasets & 4 Hands-On Lab Starters (`Énoncés`)
- [`Instructor_Live_Demos.ipynb`](./03_Labs_TP/Instructor_Live_Demos.ipynb) — 4 bite-sized live classroom walkthroughs.
- [`data/`](./03_Labs_TP/data) — `supplier_esg_audit_db.json` (10 suppliers), `green_claims_corpus.json` (8 EU regulatory & claim documents), and `green_procurement_requests.json` (4 CapEx proposals).
- [`TP1_ReAct_and_Function_Calling/TP1_Starter_ReAct_and_Tools.ipynb`](./03_Labs_TP/TP1_ReAct_and_Function_Calling/TP1_Starter_ReAct_and_Tools.ipynb) — Scope 3 ESG & EU CBAM Carbon Tax Agent (`Session 014`).
- [`TP2_Grounded_RAG_and_Interactions_API/TP2_Starter_Grounded_Agentic_RAG.ipynb`](./03_Labs_TP/TP2_Grounded_RAG_and_Interactions_API/TP2_Starter_Grounded_Agentic_RAG.ipynb) — EU Greenwashing Auditor (`Session 015`).
- [`TP3_MultiAgent_ADK_Workflows/TP3_Starter_ADK_MultiAgent_Team.ipynb`](./03_Labs_TP/TP3_MultiAgent_ADK_Workflows/TP3_Starter_ADK_MultiAgent_Team.ipynb) — Multi-Agent Green CapEx Committee with Google ADK (`Session 016`).
- [`TP4_Evaluation_Guardrails_and_Safety/TP4_Starter_Eval_and_Guardrails.ipynb`](./03_Labs_TP/TP4_Evaluation_Guardrails_and_Safety/TP4_Starter_Eval_and_Guardrails.ipynb) — Automated Evaluation, Guardrails & Red-Teaming (`Session 017`).

### 3. [`04_Homework_Oct14_to_Oct21/`](./04_Homework_Oct14_to_Oct21) — Graded Individual Homework (35% of Final Grade)
- [`agent_utils.py`](./04_Homework_Oct14_to_Oct21/agent_utils.py) — Pre-built helper module providing data loaders, `search_corpus()`, `strip_thought_parts()`, `generate_with_retry()`, `print_agent_trace()`, interactive milestone checks (`check_milestone_1..4`), and the 10-case auto-grader (`grade_my_agent()`).
- [`Homework_Starter_Responsible_Supply_Chain_Agent.ipynb`](./04_Homework_Oct14_to_Oct21/Homework_Starter_Responsible_Supply_Chain_Agent.ipynb) — Student starter notebook structured around 4 progressive milestones.
- [`eval_test_suite.json`](./04_Homework_Oct14_to_Oct21/eval_test_suite.json) — 10-case benchmark suite (`HW-EVAL-01` to `HW-EVAL-10`).

---

## Official Reference Solutions Policy
During the course (October 14 – October 21, 2026), the `main` branch contains **only Starter notebooks (`énoncés`)** so every student and pod can build and test their own solutions. Once Day 2 concludes on **Wednesday, October 21 at `18:15 CET`**, all executed reference solutions (`TP1_Solution`–`TP4_Solution` and `Homework_Reference_Solution`) will be published via **GitHub Releases (`v2026.10-solutions`)**.
