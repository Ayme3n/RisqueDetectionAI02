# backend/app/cli.py
import argparse
import pandas as pd
from pathlib import Path

from backend.app import inference


def run_cli():
    parser = argparse.ArgumentParser(
        description="Run anomaly detection on logs using trained models"
    )

    parser.add_argument(
        "input_file",
        type=str,
        help="Path to input file (CSV or JSON)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="predictions.csv",
        help="Path to save output predictions (default: predictions.csv)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Save output in JSON instead of CSV"
    )

    args = parser.parse_args()

    # 1. Load artifacts (only once per run)
    inference.load_artifacts()

    # 2. Read input logs
    input_path = Path(args.input_file)
    if input_path.suffix.lower() == ".csv":
        df = pd.read_csv(input_path)
    elif input_path.suffix.lower() == ".json":
        df = pd.read_json(input_path, lines=True)
    else:
        raise ValueError("Input file must be CSV or JSON")

    # 3. Run predictions
    results = inference.predict_batch(df)

    # 4. Save results
    output_path = Path(args.output)
    if args.json:
        results.to_json(output_path, orient="records", lines=True)
    else:
        results.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")


if __name__ == "__main__":
    run_cli()
