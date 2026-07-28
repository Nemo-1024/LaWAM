import json
import os
from pathlib import Path


def aggregate_category_results(log_dir: str | os.PathLike[str], task_suites: list[str]) -> dict[str, dict[str, float | int]]:
    log_root = Path(log_dir).expanduser().resolve()
    normalized_task_suites = [
        suite if str(suite).endswith(".json") else f"{suite}.json"
        for suite in task_suites
        if str(suite).strip()
    ]
    overall_results: dict[str, dict[str, float | int]] = {
        "overall": {"total_count": 0, "success_count": 0}
    }
    loaded_any = False

    for task_suite in normalized_task_suites:
        suite_path = log_root / task_suite
        if not suite_path.exists():
            continue
        loaded_any = True
        with suite_path.open("r", encoding="utf-8") as f:
            results = json.load(f)
        for category, counts in results.items():
            total_count = int(counts["total_count"])
            success_count = int(counts["success_count"])
            overall_results["overall"]["total_count"] += total_count
            overall_results["overall"]["success_count"] += success_count
            if category not in overall_results:
                overall_results[category] = {
                    "total_count": total_count,
                    "success_count": success_count,
                }
            else:
                overall_results[category]["total_count"] += total_count
                overall_results[category]["success_count"] += success_count

    if not loaded_any:
        raise FileNotFoundError(
            f"No suite result json found under {log_root} for task_suites={normalized_task_suites}."
        )

    for category, counts in overall_results.items():
        counts["success_rate"] = float(counts["success_count"]) / float(counts["total_count"])

    output_path = log_root / "overall_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)
    return overall_results


def main() -> None:
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        raise ValueError("LOG_DIR environment variable is required.")
    task_suites_env = os.environ.get(
        "TASK_SUITES",
        "libero_10,libero_goal,libero_object,libero_spatial",
    )
    task_suites = [s.strip() for s in task_suites_env.replace(" ", ",").split(",") if s.strip()]
    aggregate_category_results(log_dir, task_suites)


if __name__ == "__main__":
    main()
