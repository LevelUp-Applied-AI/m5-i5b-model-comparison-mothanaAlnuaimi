# Model Comparison CLI

A command-line tool for comparing multiple churn prediction models and saving the full evaluation report to an output directory.

## Overview

This project refactors the full model comparison pipeline into a production-style command-line script.  
The script loads a telecom churn dataset, validates the input, compares multiple models using stratified cross-validation, saves plots and evaluation tables, persists the best model, and generates a tree-vs-linear disagreement analysis.

## Features

- Compare 6 model configurations:
  - Dummy
  - Logistic Regression (default)
  - Logistic Regression (`class_weight='balanced'`)
  - Decision Tree (`max_depth=5`)
  - Random Forest (default)
  - Random Forest (`class_weight='balanced'`)
- 5-fold stratified cross-validation by default
- PR curve and calibration curve generation
- Best model selection by PR-AUC
- Experiment logging
- Tree-vs-linear disagreement analysis
- `--dry-run` mode for validation without training

## Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt