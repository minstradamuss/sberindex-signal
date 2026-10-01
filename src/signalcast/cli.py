import argparse
import json
from pathlib import Path
import yaml


def main():
    parser=argparse.ArgumentParser(description="Reproduce SIGNAL experiments")
    parser.add_argument("command",choices=["backtest","refine","forecast","detect","report","all"])
    parser.add_argument("--config",default="configs/full.yaml")
    parser.add_argument("--limit",type=int,help="Smoke test only; limits municipality IDs, never used for main metrics")
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    saved=Path(cfg["results_dir"])/"config.yaml"
    if saved.exists() and yaml.safe_load(saved.read_text(encoding="utf-8"))!=cfg:
        raise SystemExit("Configuration changed. Choose a new results_dir to preserve prior evidence.")
    manifest=Path(cfg["results_dir"])/"run_manifest.json"
    if manifest.exists() and args.command!="report":
        previous=json.loads(manifest.read_text(encoding="utf-8"))
        if previous.get("limit")!=args.limit:
            raise SystemExit("Municipality limit changed. Choose a new results_dir.")
    if args.command in ["backtest","all"]:
        from .backtest import run
        run(cfg,args.limit)
    if args.command in ["refine","all"]:
        from .refinement import refine
        refine(cfg,args.limit)
    if args.command=="forecast":
        from .backtest import run
        from .operations import snapshot
        run(cfg,args.limit,True)
        snapshot(cfg)
    if args.command in ["detect","all"]:
        from .detection import run_detection
        run_detection(cfg,args.limit)
    if args.command in ["report","all"]:
        from .reporting import build_report
        build_report(cfg)


if __name__=="__main__":main()
