# Graded Individual Homework (`4S4TWS_21`) — Responsible Supply Chain Auditor

- **Instructor:** Chouaieb Nemri
- **Program:** BSc Data Science for Responsible Business (*École Centrale de Lyon & emlyon business school*)
- **Assigned:** Wednesday, October 14, 2026 (`18:15 CET`)
- **Due:** Tuesday, October 20, 2026 (`23:59 CET`)
- **Weight:** **35%** of Final Course Grade
- **Models Used (Free-Tier Google AI Studio):** **`gemma-4-26b-a4b-it`** (`MODEL_FAST`), **`gemma-4-31b-it`** (`MODEL_FRONTIER`), and **`gemini-embedding-2`**

---

## 1. Files in This Folder

| File | Purpose |
| :--- | :--- |
| [`agent_utils.py`](./agent_utils.py) | **Shared helper module** — loads `SUPPLIERS_DB`, `CLAIMS_CORPUS`, and `EVAL_SUITE`, provides `search_corpus()`, includes Free-Tier Gemma 4 helpers (`strip_thought_parts` & `generate_with_retry`), prints clean agent traces, and runs the milestone checks (`check_milestone_1..4`) and 10-case auto-grader (`grade_my_agent`). |
| [`Homework_Starter_Responsible_Supply_Chain_Agent.ipynb`](./Homework_Starter_Responsible_Supply_Chain_Agent.ipynb) | **Student starter notebook (`énoncé`)** — complete the `TODO` blocks across the 4 milestones and run the checkpoint cells. |
| [`eval_test_suite.json`](./eval_test_suite.json) | **10-case benchmark suite** (`HW-EVAL-01` to `HW-EVAL-10`) testing database lookups, carbon tax math, greenwashing verification, prompt injection defense (`SUP-010`), and missing-record handling (`SUP-999`). |

*(Note: The official reference solution `Homework_Reference_Solution.ipynb` is published via GitHub Release after Day 2 concludes on Wednesday, October 21 at `18:15 CET`.)*

---

## 2. The 4 Progressive Milestones

1. **Milestone 1 — Write 3 Reliable Python Tools (25 pts):**
   - `get_supplier_record(supplier_id: str) -> dict`
   - `calculate_cbam_tax(scope1_tco2e: float, scope2_tco2e: float, rate_eur_per_ton: float = 85.0) -> dict`
   - `search_regulations_and_claims(query: str, category: str = "ALL") -> list[dict]`
   - Verify with `check_milestone_1(get_supplier_record, calculate_cbam_tax, search_regulations_and_claims)`.
2. **Milestone 2 — Define the Pydantic Output Contract (20 pts):**
   - Define `AuditDecision(BaseModel)` with `supplier_id`, `decision`, `cbam_tax_eur`, `cited_doc_ids`, and `summary`.
   - Verify with `check_milestone_2(AuditDecision)`.
3. **Milestone 3 — Wire the 2-Stage ReAct Tool-Calling Loop (30 pts):**
   - Implement `run_supply_chain_auditor(question: str, max_turns: int = 5) -> dict` using **`gemma-4-26b-a4b-it`** (`MODEL_FAST`) or **`gemma-4-31b-it`** (`MODEL_FRONTIER`).
   - Use `strip_thought_parts(...)` and `generate_with_retry(...)` from `agent_utils.py` when appending model turns to `contents` so internal reasoning tokens don't bloat your Free-Tier input token budget.
   - Verify with `check_milestone_3(run_supply_chain_auditor)`.
4. **Milestone 4 — Add a 10-Line Safety Filter for Untrusted Data (25 pts):**
   - Implement `sanitize_tool_output(raw_dict: dict) -> dict` to strip `[SYSTEM OVERRIDE...]` strings from `SUP-010`'s `audit_notes`, and enforce the Python policy rule that any supplier with `critical_non_compliance == True` must be classified as `"HIGH_RISK_REJECT"`.
   - Verify with `check_milestone_4(sanitize_tool_output, run_supply_chain_auditor)`.

---

## 3. Running the Auto-Grader & Submitting

At the bottom of your notebook, run:

```python
scorecard = grade_my_agent(run_supply_chain_auditor)
```

This prints your score out of `100.0` across all 10 cases and writes `homework_eval_results.jsonl`.

**How to Submit (Due Tuesday, October 20 at `23:59 CET`):**
- Submit your executed `Homework_Starter_Responsible_Supply_Chain_Agent.ipynb` and `homework_eval_results.jsonl` via a **private GitHub repository** (invite the instructor) or Google Drive folder.
- **Important:** Do **not** open a public Pull Request on the course repository with your homework answers.
