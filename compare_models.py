import argparse
import logging
import os
import sys

import pandas as pd

from model_comparison import (
    NUMERIC_FEATURES,
    define_models,
    run_cv_comparison,
    save_comparison_table,
    plot_pr_curves_top3,
    plot_calibration_top3,
    save_best_model,
    log_experiment,
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the telecom churn model comparison pipeline."
    )
    parser.add_argument(
        "--data-path",
        required=True,
        help="Path to the input dataset CSV."
    )
    parser.add_argument(
        "--output-dir",
        default="./output",
        help="Directory where results and plots are saved."
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds."
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the data and print configuration without training models."
    )
    return parser.parse_args()


def load_data(data_path):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    logging.info("Loaded data from %s (%d rows, %d columns)", data_path, df.shape[0], df.shape[1])
    return df


def validate_data(df):
    required_columns = set(NUMERIC_FEATURES + ["churned"])
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    class_dist = df["churned"].value_counts(normalize=True).sort_index()
    logging.info("Validation passed.")
    logging.info("Class distribution: %s", class_dist.to_dict())


def split_features_target(df):
    X = df[NUMERIC_FEATURES]
    y = df["churned"]
    return X, y


def save_outputs(output_dir, results_df, fitted_models, X_test, y_test):
    os.makedirs(output_dir, exist_ok=True)

    save_comparison_table(results_df, os.path.join(output_dir, "comparison_table.csv"))
    plot_pr_curves_top3(fitted_models, X_test, y_test, os.path.join(output_dir, "pr_curves.png"))
    plot_calibration_top3(fitted_models, X_test, y_test, os.path.join(output_dir, "calibration.png"))
    log_experiment(results_df, os.path.join(output_dir, "experiment_log.csv"))

    best_name = results_df.sort_values("pr_auc_mean", ascending=False).iloc[0]["model"]
    save_best_model(fitted_models[best_name], os.path.join(output_dir, "best_model.joblib"))
    logging.info("Best model by PR-AUC: %s", best_name)


def run_pipeline(df, output_dir, n_folds, random_seed):
    from sklearn.model_selection import train_test_split

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=random_seed
    )

    logging.info(
        "Split complete: train=%d, test=%d, churn rate=%.2f%%",
        len(X_train),
        len(X_test),
        y_train.mean() * 100
    )

    models = define_models()
    logging.info("Defined %d model configurations: %s", len(models), list(models.keys()))

    results_df = run_cv_comparison(
        models,
        X_train,
        y_train,
        n_splits=n_folds,
        random_state=random_seed
    )
    logging.info("Cross-validation complete.")

    fitted_models = {}
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        fitted_models[name] = pipeline

    save_outputs(output_dir, results_df, fitted_models, X_test, y_test)
    logging.info("All outputs saved to %s", output_dir)


def dry_run_report(df, args):
    logging.info("Dry run mode enabled. No models will be trained.")
    logging.info("Data shape: %s", df.shape)
    logging.info("Expected feature columns: %s", NUMERIC_FEATURES)
    logging.info("Target column: churned")
    logging.info("Configured CV folds: %d", args.n_folds)
    logging.info("Random seed: %d", args.random_seed)
    logging.info("Output directory: %s", args.output_dir)
    logging.info("Models to compare: %s", list(define_models().keys()))


def main():
    setup_logging()
    args = parse_args()

    try:
        os.makedirs(args.output_dir, exist_ok=True)

        df = load_data(args.data_path)
        validate_data(df)

        if args.dry_run:
            dry_run_report(df, args)
            return

        run_pipeline(
            df=df,
            output_dir=args.output_dir,
            n_folds=args.n_folds,
            random_seed=args.random_seed
        )

    except (FileNotFoundError, ValueError) as exc:
        logging.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logging.exception("Unexpected failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()