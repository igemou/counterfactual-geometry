from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def model_label(payload: dict[str, Any]) -> str:
    dataset_name = str(payload.get("dataset", "")).lower()
    if dataset_name != "mmimdb":
        return str(payload.get("encoder", "unknown"))
    representation = str(payload.get("representation", "")).lower()
    if not representation:
        return str(payload.get("encoder", "mmimdb"))
    if representation == "multimodal":
        return str(payload.get("multimodal_encoder", "multimodal"))
    if representation == "text":
        return str(payload.get("text_encoder", "text"))
    if representation == "image":
        return str(payload.get("image_encoder", "image"))
    if representation in {"fused"}:
        image = str(payload.get("image_encoder", "image"))
        text = str(payload.get("text_encoder", "text"))
        return f"fused:{image}+{text}"
    return representation or "mmimdb"


def payload_success(result: dict[str, Any]) -> bool:
    return bool(result.get("counterfactual_success", False))


def payload_margin(result: dict[str, Any]) -> float:
    if "counterfactual_margin" in result:
        return float(result["counterfactual_margin"])
    return float(result.get("decision_margin", 0.0))


def payload_trajectory_key(result: dict[str, Any]) -> str | None:
    if "counterfactual_trajectory" in result:
        return "counterfactual_trajectory"
    return None


def filtered_payload(payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    copied = dict(payload)
    copied["num_evaluated"] = 1
    copied["raw_results"] = [result]
    return copied


def _normalized_metrics(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_index": int(result.get("example_index", -1)),
        "start_label": int(result.get("start_label", -1)),
        "target_label": int(result.get("target_label", -1)),
        "final_label": int(result.get("final_label", -1)),
        "success": payload_success(result),
        "margin": payload_margin(result),
        "distance": float(result.get("counterfactual_distance", 0.0)),
        "effort": int(result.get("optimization_effort", 0)),
        "trajectory_available": payload_trajectory_key(result) is not None,
    }


def _single_score(result: dict[str, Any]) -> float:
    score = 0.0
    score += 8.0 if result["success"] else 0.0
    score += result["margin"]
    score -= 0.35 * result["distance"]
    score -= 0.02 * float(result["effort"])
    score += 1.0 if result["trajectory_available"] else -5.0
    return score


def _contrast_score(results: list[dict[str, Any]]) -> float:
    successes = [1.0 if row["success"] else 0.0 for row in results]
    margins = [row["margin"] for row in results]
    distances = [row["distance"] for row in results]
    efforts = [float(row["effort"]) for row in results]
    same_start = len({row["start_label"] for row in results}) == 1
    same_target = len({row["target_label"] for row in results if row["target_label"] >= 0}) <= 1
    any_missing_trajectory = any(not row["trajectory_available"] for row in results)
    score = 0.0
    score += 15.0 if min(successes) != max(successes) else 0.0
    score += 3.0 if same_start else 0.0
    score += 3.0 if same_target else 0.0
    score += max(margins) - min(margins)
    score += 0.05 * (max(efforts) - min(efforts))
    score -= 0.1 * sum(distances) / max(len(distances), 1)
    if any_missing_trajectory:
        score -= 20.0
    return score


def select_case_studies(
    inputs: list[Path],
    *,
    top_k: int,
    mode: str,
    require_success: bool,
    require_first_success: bool,
    require_second_failure: bool,
    require_same_start: bool,
    require_same_target: bool,
    require_trajectory: bool,
) -> dict[str, Any]:
    payloads = []
    for path in inputs:
        payload = load_payload(path)
        payloads.append(
            {
                "path": str(path),
                "dataset": str(payload.get("dataset", "")),
                "model": model_label(payload),
                "raw_results": [_normalized_metrics(row) for row in payload.get("raw_results", [])],
            }
        )

    current_mode = "contrast" if mode == "auto" and len(payloads) > 1 else ("single" if mode == "auto" else mode)
    candidates = []
    if current_mode == "single":
        payload = payloads[0]
        for row in payload["raw_results"]:
            if row["example_index"] < 0:
                continue
            if require_success and not row["success"]:
                continue
            if require_trajectory and not row["trajectory_available"]:
                continue
            candidates.append({"example_index": row["example_index"], "score": _single_score(row), "per_model": [{"model": payload["model"], **row}]})
    else:
        per_input = [{row["example_index"]: row for row in payload["raw_results"] if row["example_index"] >= 0} for payload in payloads]
        shared_indices = sorted(set.intersection(*(set(mapping.keys()) for mapping in per_input))) if per_input else []
        for example_index in shared_indices:
            rows = [mapping[example_index] for mapping in per_input]
            if require_trajectory and any(not row["trajectory_available"] for row in rows):
                continue
            if require_same_start and len({row["start_label"] for row in rows}) != 1:
                continue
            if require_same_target and len({row["target_label"] for row in rows}) != 1:
                continue
            if require_first_success and rows and not rows[0]["success"]:
                continue
            if require_second_failure and len(rows) >= 2 and rows[1]["success"]:
                continue
            candidates.append({
                "example_index": example_index,
                "score": _contrast_score(rows),
                "per_model": [{"model": payloads[idx]["model"], **rows[idx]} for idx in range(len(rows))],
            })

    ranked = sorted(candidates, key=lambda row: row["score"], reverse=True)
    return {
        "mode": current_mode,
        "inputs": [{"path": payload["path"], "dataset": payload["dataset"], "model": payload["model"]} for payload in payloads],
        "top_candidates": ranked[:top_k],
    }


def extract_case_study(
    left: Path,
    example_index: int,
    output_dir: Path,
    right: Path | None = None,
) -> dict[str, Any]:
    def _find_result(payload: dict[str, Any], current_example_index: int) -> dict[str, Any]:
        for row in payload.get("raw_results", []):
            if int(row.get("example_index", -1)) == current_example_index:
                return row
        raise ValueError(f"example_index={current_example_index} not found in payload")

    left_payload = load_payload(left)
    left_row = _find_result(left_payload, example_index)
    right_payload = load_payload(right) if right is not None else None
    right_row = _find_result(right_payload, example_index) if right_payload is not None else None

    output_dir.mkdir(parents=True, exist_ok=True)
    left_output = output_dir / f"{left.stem}_example_{example_index}.json"
    left_output.write_text(json.dumps(filtered_payload(left_payload, left_row), indent=2, sort_keys=True) + "\n")

    summary: dict[str, Any] = {
        "example_index": example_index,
        "left": {
            "path": str(left_output),
            "dataset": left_payload.get("dataset"),
            "encoder": model_label(left_payload),
            "start_label": left_row.get("start_label"),
            "target_label": left_row.get("target_label"),
            "final_label": left_row.get("final_label"),
            "success": left_row.get("counterfactual_success"),
            "distance": left_row.get("counterfactual_distance"),
            "margin": left_row.get("counterfactual_margin", left_row.get("decision_margin")),
            "optimization_effort": left_row.get("optimization_effort"),
        },
    }
    if right_payload is not None and right_row is not None and right is not None:
        right_output = output_dir / f"{right.stem}_example_{example_index}.json"
        right_output.write_text(json.dumps(filtered_payload(right_payload, right_row), indent=2, sort_keys=True) + "\n")
        summary["right"] = {
            "path": str(right_output),
            "dataset": right_payload.get("dataset"),
            "encoder": model_label(right_payload),
            "start_label": right_row.get("start_label"),
            "target_label": right_row.get("target_label"),
            "final_label": right_row.get("final_label"),
            "success": right_row.get("counterfactual_success"),
            "distance": right_row.get("counterfactual_distance"),
            "margin": right_row.get("counterfactual_margin", right_row.get("decision_margin")),
            "optimization_effort": right_row.get("optimization_effort"),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Case-study helpers for figure preparation.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select", help="Select strong case-study examples.")
    select_parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    select_parser.add_argument("--top-k", type=int, default=10)
    select_parser.add_argument("--mode", choices=["auto", "single", "contrast"], default="auto")
    select_parser.add_argument("--require-success", action="store_true")
    select_parser.add_argument("--require-first-success", action="store_true")
    select_parser.add_argument("--require-second-failure", action="store_true")
    select_parser.add_argument("--require-same-start", action="store_true")
    select_parser.add_argument("--require-same-target", action="store_true")
    select_parser.add_argument("--require-trajectory", action="store_true")
    select_parser.add_argument("--output", type=Path, default=None)

    extract_parser = subparsers.add_parser("extract", help="Extract one case-study example.")
    extract_parser.add_argument("--left", type=Path, required=True)
    extract_parser.add_argument("--right", type=Path, default=None)
    extract_parser.add_argument("--example-index", type=int, required=True)
    extract_parser.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select":
        output = select_case_studies(
            args.inputs,
            top_k=args.top_k,
            mode=args.mode,
            require_success=args.require_success,
            require_first_success=args.require_first_success,
            require_second_failure=args.require_second_failure,
            require_same_start=args.require_same_start,
            require_same_target=args.require_same_target,
            require_trajectory=args.require_trajectory,
        )
        text = json.dumps(output, indent=2) + "\n"
        print(text, end="")
        if args.output is not None:
            args.output.write_text(text)
        return

    output = extract_case_study(
        left=args.left,
        right=args.right,
        example_index=args.example_index,
        output_dir=args.output_dir,
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
