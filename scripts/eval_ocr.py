"""Score an OCR engine's predictions against the Book 1 name-crop ground truth.

Ground truth lives in ``data/oracles/book1_names_truth.json`` (crop_id -> Unicode
char, hand-verified against golden). An engine produces a predictions JSON of the
same shape ({crop_id: predicted_char}); this script reports per-character accuracy,
lists every miss (id, truth, prediction), and notes any ids the engine skipped
(the "undetected" case grid-packing is meant to surface).

Usage::

    python scripts/eval_ocr.py <predictions.json> [--name ENGINE_NAME]

The predictions file is either {"labels": {id: char}} or a bare {id: char} map.
"""

from __future__ import annotations

import argparse
import json

TRUTH_PATH = "data/oracles/book1_names_truth.json"


def load_labels(path: str) -> dict[str, str]:
    obj = json.load(open(path))
    labels = obj.get("labels", obj)
    return {str(k): v for k, v in labels.items() if not k.startswith("_")}


def evaluate(pred_path: str, engine_name: str) -> dict:
    truth = load_labels(TRUTH_PATH)
    preds = load_labels(pred_path)

    hits, misses, skipped = 0, [], []
    for cid, true_char in sorted(truth.items(), key=lambda kv: int(kv[0])):
        pred = preds.get(cid)
        if pred is None or pred == "":
            skipped.append((cid, true_char))
        elif pred == true_char:
            hits += 1
        else:
            misses.append((cid, true_char, pred))

    total = len(truth)
    attempted = total - len(skipped)
    print(f"\n=== OCR eval: {engine_name} ===")
    print(f"  ground-truth chars : {total}")
    print(f"  correct            : {hits}/{total}  ({100 * hits / total:.1f}%)")
    if attempted:
        print(f"  correct (attempted): {hits}/{attempted}  ({100 * hits / attempted:.1f}%)")
    print(f"  wrong              : {len(misses)}")
    print(f"  skipped/undetected : {len(skipped)}")
    if misses:
        print("  --- misses (id: truth -> pred) ---")
        for cid, t, p in misses:
            print(f"    {cid:>3}: {t} -> {p}")
    if skipped:
        print(f"  --- skipped ids: {', '.join(c for c, _ in skipped)}")

    return {
        "engine": engine_name,
        "total": total,
        "correct": hits,
        "accuracy": hits / total if total else 0.0,
        "wrong": len(misses),
        "skipped": len(skipped),
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Score OCR predictions vs ground truth.")
    ap.add_argument("predictions", help="Path to predictions JSON ({id: char})")
    ap.add_argument("--name", default="engine", help="Engine name for the report")
    args = ap.parse_args()
    evaluate(args.predictions, args.name)


if __name__ == "__main__":
    main()
