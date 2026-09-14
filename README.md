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
