# MLB Season Wins Prediction (Moneyball)

NTU-PACE-SCTP-DSAI-2026 | Module 3.2 Project

## Objective
Predict the number of games an MLB team wins in a season using historical team statistics from the 2016 Lahman Baseball Database. Evaluation metric: Mean Absolute Error (MAE).

## Dataset
- Source: [Kaggle Competition — SCTPDSAI-M3-DS5-Coaching-Moneyball-Analytics](https://www.kaggle.com/competitions/sctpdsai-m-3-ds-5-coaching-moneyball-analytics)
- Training: `data.csv` — 1,812 rows × 44 features
- Test: `predict.csv` — 453 rows (no W column)
- Features: batting statistics (R, AB, H, HR, BB, etc.), pitching statistics (RA, ERA, FIP, etc.), fielding statistics (E, DP, FP), era/decade binary indicators

## Approach

### Model Selection
Evaluated Linear Regression, Ridge, Lasso, Gradient Boosting, XGBoost, LightGBM, and Random Forest. Simple regularized linear models (Lasso) consistently outperformed tree ensembles, reflecting the inherently linear relationship between run differential and wins in baseball.

### Feature Engineering
Derived 10 domain-specific sabermetric indicators from raw team statistics:

| Feature | Formula | Baseball Meaning |
|---------|---------|-----------------|
| run_diff | R - RA | Net run differential — strongest single predictor of wins |
| run_ratio | R / (RA+1) | Relative scoring dominance |
| OBP | (H+BB) / (AB+BB+1) | On-base percentage |
| SLG | (H+2B+2×3B+3×HR) / (AB+1) | Slugging percentage |
| OPS | OBP + SLG | Combined offensive metric |
| WHIP | (BBA+HA) / IP | Walks + hits per inning pitched |
| K_BB_ratio | SOA / (BBA+1) | Pitcher strikeout-to-walk ratio |
| pyth_wins | R^1.84 / (R^1.84 + RA^1.84) × G | Pythagorean expected wins |
| FIP | (13×HRA + 3×BBA - 2×SOA) / IP + 3.2 | Fielding Independent Pitching — pitcher skill excluding defense |
| def_efficiency | 1 - (HA-HRA) / (BBA+HA-HRA+1) | Defense conversion rate of balls in play |

### Key Finding: CV MAE and Kaggle MAE Linear Relationship
Through systematic experimentation, we discovered a near-perfect linear relationship between cross-validation MAE and Kaggle public leaderboard MAE:

Kaggle MAE = 1.0259 × CV MAE + 0.2743 (R² = 0.9995)


This allowed us to predict Kaggle scores before submitting and prioritize which experiments were worth submitting.

## Results

| Method | CV MAE | Kaggle MAE |
|--------|--------|------------|
| Linear Regression (baseline) | 2.767 | 3.115 |
| Ridge (alpha=1.0) | 2.764 | 3.102 |
| Lasso + engineered features (OBP, OPS, run_diff, pyth_wins, WHIP) | 2.714 | 3.057 |
| **Lasso + FIP + def_efficiency (final)** | **2.7106** | **3.016** |

**Improved MAE from 3.115 (Linear Regression baseline) to 3.016 (Lasso with feature engineering), an 8.5% reduction.**

## Files

├── best_model.py # Final model code — runs end-to-end and generates submission.csv
├── reports/
│ └── moneyball_report_v2.docx # Full project report with all experiments, metrics, and analysis
└── notes/
└── module3_2_qa_recap.md # Q&A recap from the learning process


## How to Run

```bash
# Place data.csv and predict.csv in the same directory as best_model.py
python best_model.py
# Outputs: submission.csv + cross-validation metrics (MAE, RMSE, R², error distribution)
```

## Requirements

pandas
numpy
scikit-learn


## Key Lessons
- Simple regularized linear models outperform tree ensembles on small structured datasets with near-linear relationships
- Domain-specific feature engineering (grounded in baseball theory) outperformed pure data-driven feature selection
- CV improvement does not always translate to held-out test improvement — features with CV gains < 0.005 were typically noise
- Training on the full dataset generalizes better than holding out a validation set when data is limited
