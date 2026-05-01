from pathlib import Path
import pandas as pd
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

INPUT_CANDIDATES = [
    SCRIPT_DIR / "anatomical_roi_subject_stage_parameters_wide.csv",
    SCRIPT_DIR / "data" / "anatomical_roi_subject_stage_parameters_wide.csv",
]

N_ITER = 1000
SEED = 20260426


def find_input_csv():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Input CSV not found. Run 01_main_analysis.py first or place CSV in /data folder."
    )


def main():
    np.random.seed(SEED)

    input_csv = find_input_csv()
    print(f"Using input: {input_csv}")

    df = pd.read_csv(input_csv)

    subjects = df["subject"].unique()
    results = []

    for i in range(N_ITER):
        half = np.random.choice(subjects, size=len(subjects)//2, replace=False)
        df_half = df[df["subject"].isin(half)]

        mean_val = df_half.select_dtypes(include=[np.number]).mean().mean()
        results.append(mean_val)

    out_df = pd.DataFrame({"iteration": range(N_ITER), "value": results})
    out_path = SCRIPT_DIR / "split_half_results.csv"
    out_df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
