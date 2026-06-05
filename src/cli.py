"""CLI entry point for the noise-to-insight pipeline."""

from __future__ import annotations

import argparse
import sys

from src.ingest.loader import expand_input_paths
from src.orchestrator.graph import run_pipeline


def main(argv: list[str] | None = None) -> int:
    """Run the full pipeline from the command line."""
    parser = argparse.ArgumentParser(
        prog="noise-to-insight",
        description="AI Meets Data: From Noise to Insight",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        help="File paths or globs (e.g. data/demo/*.csv)",
    )
    parser.add_argument(
        "--run-id",
        default="cli_run",
        help="Artifact run identifier (default: cli_run)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable Azure OpenAI (heuristic fallbacks only)",
    )
    args = parser.parse_args(argv)

    if not args.input:
        parser.error("--input is required (one or more file paths or globs)")

    paths = expand_input_paths(args.input)
    final = run_pipeline(paths, args.run_id, use_llm=not args.no_llm)
    print(f"Pipeline complete: {final.artifact_dir}")
    if final.headline_insight:
        print(f"Headline: {final.headline_insight}")
    if final.report_path:
        print(f"Report: {final.report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
