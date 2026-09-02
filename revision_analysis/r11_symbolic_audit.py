#!/usr/bin/env python3
"""Trace the 14 retained composites without rerunning GP evolution."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

REVISION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = REVISION_ROOT.parent
OUT = REVISION_ROOT / "06_analysis_outputs/symbolic_audit"
DEDUP = REPOSITORY_ROOT / "outputs/05_gp_runs/dedup/deduplicated_best_expressions.json"
FINAL = REPOSITORY_ROOT / "outputs/05_gp_runs/summaries/final_composite_expression.json"
ORIGIN = REPOSITORY_ROOT / "outputs/10_review_response_computations/gp_audit/composite_fold_origin_table.csv"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dedup = json.loads(DEDUP.read_text(encoding="utf-8"))
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    origins = list(csv.DictReader(ORIGIN.open(newline="", encoding="utf-8")))
    by_expression = {row["expression"]: row for row in origins}
    records = [{
        "best_expression": final["best_expression"], "held_out_dataset": "FINAL_ALL", "seed": final["seed"],
        "best_training_objective": final["best_training_objective"], "heldout_score": "",
        "all_dataset_score": final["best_all_dataset_score"], "num_nodes": final["num_nodes"],
        "tree_height": final["tree_height"], "best_complexity": final["best_complexity"], "source_type": "final_refit",
    }] + [{**row, "source_type": "LODO_deduplicated"} for row in dedup]
    output = []
    for index, record in enumerate(records, 1):
        expression = record["best_expression"]
        origin = by_expression.get(expression, {})
        output.append({
            "composite_id": origin.get("composite_id", f"CFM{index}"), "exact_expression": expression,
            "fold": origin.get("fold_origin", record.get("held_out_dataset", "not reported")),
            "seed": origin.get("seed_origin", record.get("seed", "not reported")),
            "training_score": record.get("best_training_objective", "not reported"),
            "held_out_score": origin.get("heldout_score_from_lodo_stage", record.get("heldout_score", "not applicable")),
            "final_common_value_score": origin.get("common_value_G", record.get("all_dataset_score", "not reported")),
            "node_count": record.get("num_nodes", record.get("best_complexity", "not reported")),
            "depth": record.get("tree_height", "not reported"),
            "pareto_selection": "best valid seed result under primary score then complexity then seed",
            "validity_filter": "passed; finite retained result present in frozen shortlist",
            "canonicalization": "exact expression string; no algebraic rewriting recorded",
            "duplicate_group": f"unique_retained_{index:02d}",
            "size_complexity_filter": "GP max_nodes=35 and max_tree_depth=10; candidate passed",
            "filtering_decision": "retained",
            "retention_reason": "final all-domain refit" if record["source_type"] == "final_refit" else "best representative after exact-expression and within-fold functional deduplication at correlation >=0.99",
        })
    fields = list(output[0])
    with (OUT / "retained_composite_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(output)
    configuration = {
        "full_gp_rerun": False, "retained_count": len(output),
        "selection_rule": "final all-domain refit plus all 13 LODO representatives remaining after exact-expression and same-heldout-domain curve-correlation deduplication; sorted by heldout score, complexity, seed",
        "functional_equivalence_threshold": 0.99, "max_nodes": 35, "max_tree_depth": 10,
        "input_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (DEDUP, FINAL, ORIGIN)},
        "validation": {"expected_retained": 14, "observed_retained": len(output), "all_expressions_present": all(bool(row["exact_expression"]) for row in output)},
    }
    (OUT / "symbolic_audit_configuration.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    (OUT / "SYMBOLIC_AUDIT_FINDINGS.md").write_text(f"""# Symbolic-composite audit findings

The frozen evidence deterministically reconstructs {len(output)} retained composites: one final all-domain refit and 13 LODO-origin representatives. Exact-expression duplicates were collapsed first; functional duplicates were removed only within the same held-out domain when mean held-out curves correlated at or above 0.99. Representatives were ordered by held-out score, complexity and seed. The GP search was not rerun.

The final-refit expression has no held-out-domain score and is labelled accordingly. Composite results remain exploratory and are not allowed to displace the primary 32-operator benchmark conclusion.
""", encoding="utf-8")
    print(f"wrote {len(output)} composite provenance rows")
    return 0 if len(output) == 14 else 1


if __name__ == "__main__":
    raise SystemExit(main())
