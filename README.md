# Ticket Escalation Risk Model (Slow Ticket Predictor)

## What this is
A small AI/ML project that scores new customer support tickets by “slow ticket risk”.
Idea: help a team prioritise/escalate tickets early.

## Dataset
Kaggle Customer Support Ticket Dataset (CC0).

## Label (what we predict)
This dataset doesn’t really contain 48-hour resolution times, so we define:

**Slow ticket = the slowest 25% of tickets** (top quartile).

Steps:
1) Convert timestamps into hours:
   `hours = Time to Resolution - First Response Time`
2) If hours is negative, add 24 (overnight wrap-around fix)
3) Slow ticket = `hours > 75th percentile`

So:
- `0` = not slow
- `1` = slow

## Features used (known when ticket is created)
- Ticket Subject + Ticket Description (text)
- Ticket Type, Priority, Channel, Product Purchased

We do not use resolution text, ticket status, satisfaction rating, or the timestamps as features.

## Models
- Baseline: TF-IDF + Logistic Regression
- Neural net: TF-IDF + SVD + MLPClassifier (trained in epochs, loss curve plotted)

## Results (example run)
- Tickets used: 2769
- Slow threshold: 17.95 hours
- Baseline ROC-AUC: 0.477 | PR-AUC: 0.232 | top10% recall: 0.087
- Neural net epochs: 12 | ROC-AUC: 0.464 | PR-AUC: 0.231 | top10% recall: 0.080

## How to run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt

