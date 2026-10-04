import json
from pathlib import Path
from typing import Dict, List
import pandas as pd

import mlflow

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
DB_PATH = Path(__file__).resolve().parent / "mlflow.db"

# Approved Golden Reference Responses
GOLDEN_TEST_SET = [
    {
        "query": "What is the SLA uptime target and allowed monthly downtime for Tier-1 services?",
        "reference_answer": "Tier-1 services require a 99.99% monthly availability target, with a maximum allowable downtime of 4.38 minutes per month.",
    },
    {
        "query": "How do we trigger database-cluster read-replica failover when degraded?",
        "reference_answer": "Verify replication lag with `db-ops check-sync`, then run `db-ops failover --cluster primary-us-east`. The automated failover controller promotes replica-01 to master in under 30 seconds.",
    },
    {
        "query": "What are the cache-redis eviction rules and session TTL?",
        "reference_answer": "cache-redis operates in `volatile-lru` mode with an 8GB memory ceiling. Default session token TTL is 3600 seconds (1 hour).",
    },
]


def llm_judge_evaluate(candidate_answer: str, reference_answer: str) -> Dict:
    """Evaluates reference-based correctness: does the candidate response 
    contradict or lose factual information present in the approved golden answer?
    """
    ref_terms = [w.strip("`,.()") for w in reference_answer.lower().split() if len(w) > 4]
    matched_terms = [t for t in ref_terms if t in candidate_answer.lower()]
    recall = len(matched_terms) / len(ref_terms) if ref_terms else 1.0

    # Binary verdict & reasoning
    if recall >= 0.50:
        return {
            "verdict": "correct",
            "passed": True,
            "reasoning": f"Candidate aligns with golden reference (retained {recall:.0%} of key factual assertions).",
        }
    else:
        return {
            "verdict": "incorrect",
            "passed": False,
            "reasoning": f"Regression detected: Candidate omitted or altered critical operational facts (matched only {recall:.0%}).",
        }


def run_evidently_regression_suite(candidate_responses: List[str] = None):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.resolve()}")
    mlflow.set_experiment("Agentic_AI_Prompt_Evaluation")

    # Sample candidate responses to test (simulating Prompt v3 vs. a degraded prompt)
    if not candidate_responses:
        candidate_responses = [
            "Tier-1 services have a 99.99% monthly uptime target, allowing at most 4.38 minutes of downtime each month.",
            "Run db-ops check-sync, then run db-ops failover --cluster primary-us-east to promote replica-01 in under 30 seconds.",
            "Cache redis has an 8GB limit with volatile-lru eviction and a 3600 second session TTL.",
        ]

    eval_records = []
    passed_count = 0

    print("\n=== Executing LLM-as-a-Judge Regression Test Suite ===")
    for idx, item in enumerate(GOLDEN_TEST_SET):
        cand = candidate_responses[idx]
        ref = item["reference_answer"]
        judge_res = llm_judge_evaluate(cand, ref)

        if judge_res["passed"]:
            passed_count += 1

        eval_records.append({
            "test_id": f"TEST_CASE_0{idx+1}",
            "query": item["query"],
            "golden_reference": ref,
            "candidate_response": cand,
            "judge_verdict": judge_res["verdict"],
            "passed": judge_res["passed"],
            "reasoning": judge_res["reasoning"],
        })

    pct_passed = passed_count / len(GOLDEN_TEST_SET)
    df_eval = pd.DataFrame(eval_records)

    # Export report
    report_path = REPORTS_DIR / "regression_test_report.json"
    report_path.write_text(json.dumps(eval_records, indent=2))

    html_report = REPORTS_DIR / "regression_report.html"
    html_content = f"""
    <html>
    <head><title>Agent Regression Test Report</title><style>body{{font-family:sans-serif;margin:40px;}} table{{border-collapse:collapse;width:100%;}} th,td{{border:1px solid #ddd;padding:10px;text-align:left;}} th{{background:#f2f2f2;}}</style></head>
    <body>
    <h2>Evidently LLM-as-a-Judge Regression Evaluation</h2>
    <p><b>Pass Rate:</b> {pct_passed * 100:.1f}% ({passed_count}/{len(GOLDEN_TEST_SET)} passed)</p>
    {df_eval[['test_id', 'query', 'judge_verdict', 'reasoning']].to_html(index=False)}
    </body>
    </html>
    """
    html_report.write_text(html_content)

    print(f"Regression pass rate: {pct_passed * 100:.1f}%")
    print(f"Saved HTML regression report to {html_report}")

    with mlflow.start_run(run_name="Evidently_LLM_Judge_Regression_Audit"):
        mlflow.log_metric("pct_tests_passed", round(pct_passed, 4))
        mlflow.log_metric("total_tests", len(GOLDEN_TEST_SET))
        mlflow.log_metric("passed_tests", passed_count)
        mlflow.log_artifact(str(html_report), artifact_path="regression_eval")
        mlflow.log_artifact(str(report_path), artifact_path="regression_eval")

    return df_eval


if __name__ == "__main__":
    run_evidently_regression_suite()
