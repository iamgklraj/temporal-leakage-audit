"""Exploratory extension of the language-model study to two further open-weight families.

Gemma 3 12B (Google; documented knowledge cutoff August 2024) and Phi-4 14B (Microsoft;
documented training-data cutoff June 2024) are run locally with Ollama at temperature 0 on the
same 600 trials, prompts, batches and masking as the primary study
(scripts/llm_memorization_study.py). Results are written to separate files so that the
pre-specified primary analysis of the original five models is unchanged; the Holm adjustment in
this file covers only these models and is exploratory. Because some sampled trials completed
after these cutoffs, every contrast is also reported for trials completed before and after each
model's cutoff (a partial post-cutoff control).

Usage:  python scripts/llm_open_models.py            # query (resumes from outputs/llm_raw/) and analyze
        python scripts/llm_open_models.py --analyze-only
"""
import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llm_memorization_study as S  # noqa: E402

MODELS = ["ollama:gemma3:12b", "ollama:phi4:14b"]
S.CUTOFFS.update({"ollama:gemma3:12b": (2024, 8),   # Gemma 3 model card: "August 2024"
                  "ollama:phi4:14b": (2024, 6)})    # Phi-4 model card: "June 2024 and earlier"
OUT_PATH = "outputs/llm_open_models.json"
PRED_PATH = "outputs/llm_open_models_predictions.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    samp = S.load_sample(S.N)
    if not args.analyze_only:
        for m in args.models:
            S.query_model(samp, m, workers=1)
    models = [m for m in args.models if os.path.exists(S._raw_path(m))]
    preds = S.collect(samp, models)
    preds.to_csv(PRED_PATH, index=False)
    meta = {"analyzed": datetime.date.today().isoformat(), "n_sample": len(samp), "sample_seed": S.SEED,
            "batch_size": S.BATCH, "ollama_options": "temperature 0, seed 0, num_ctx 8192",
            "cutoffs": {m: f"{S.CUTOFFS[m][0]}-{S.CUTOFFS[m][1]:02d}" for m in models},
            "note": "Exploratory extension; Holm adjustment across these models only."}
    out = S.analyze(preds, meta)
    out["primary_note"] = "Exploratory: same contrasts as the primary study, Holm-adjusted across these models only."
    with open(OUT_PATH, "w") as fh:
        json.dump(out, fh, indent=2)
    print("DONE ->", OUT_PATH, flush=True)


if __name__ == "__main__":
    main()
