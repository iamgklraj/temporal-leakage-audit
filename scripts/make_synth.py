"""Generate the synthetic dataset (programs.csv + evidence.csv)."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from temporal_leakage_audit.config import get_config  # noqa: E402
from temporal_leakage_audit.data import synth  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional YAML override")
    args = ap.parse_args()

    cfg = get_config(args.config)
    p_path, e_path = synth.write(cfg)

    programs = pd.read_csv(p_path)
    evidence = pd.read_csv(e_path)
    n = len(programs)
    print(f"Wrote {p_path} ({n} programs) and {e_path} ({len(evidence)} evidence rows)")
    print(f"  unique targets : {programs['target_id'].nunique()} (reuse => non-independence)")
    print(f"  base success   : {programs['label'].mean():.3f}")
    print(f"  years          : {programs['info_time'].min()}-{programs['info_time'].max()}")
    post_hoc = (
        evidence.merge(programs[['program_id', 'info_time']], on='program_id')
        .assign(post=lambda d: d['evidence_date'] > d['info_time'])
    )
    lit = post_hoc[post_hoc['evidence_type'] == 'literature']
    print(f"  literature rows dated AFTER info_time: {lit['post'].mean():.1%} (this is the leakage)")


if __name__ == "__main__":
    main()
