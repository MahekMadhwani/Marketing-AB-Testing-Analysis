# A/B Test Analysis: Marketing Campaign Effectiveness

Statistical and Bayesian analysis of a real 588,101-user randomized experiment testing whether showing ads increases conversion rate compared to a Public Service Announcement (PSA) control.

**[Read the full business write-up →](business_writeup.md)** · **[Try the live conversion predictor →](https://marketing-ab-testing-analysis-e2pjvd28etox5d5vvahjsa.streamlit.app/)**

## TL;DR

Ads produced a **43% relative lift** in conversion rate over the PSA control (2.55% vs 1.79%), a result that is statistically significant (p < 0.0001), consistent across every day of the week and nearly every hour of the day, and — via a Bayesian analysis — essentially certain (>99.99% probability ads are genuinely better).

## What This Project Covers

| Step | What it answers | Techniques |
|---|---|---|
| 1. Load & Clean | Is the data usable? | Data validation, group balance check |
| 2. EDA | What does the data look like? | Distribution analysis, heatmaps |
| 3. Hypothesis Test | Is the ad-vs-PSA difference real? | Two-proportion z-test, chi-square, effect size |
| 4. Power Analysis | Was the experiment big enough? | Post-hoc power, minimum detectable effect |
| 5. Segment Analysis | Does the effect hold everywhere? | Dose-response curves, Simpson's Paradox check |
| 6. Predictive Model | Can we predict who converts? | Logistic regression, XGBoost, SHAP |
| 7. Bayesian A/B Test | How confident are we, really? | Beta-Binomial conjugate model, expected loss |

## Key Findings

- **Statistically significant ≠ practically large, but both can point the same direction.** The standardized effect size here is technically "small" (Cohen's h = 0.05), yet the relative lift (43%) is a substantial business outcome — this project explicitly separates the two rather than conflating them.
- **The experiment was massively over-powered** — only ~70,000 users were needed to detect this effect reliably; 588,101 were used.
- **No Simpson's Paradox** — the ad advantage holds directionally across every day and nearly every hour, which is what makes the finding trustworthy enough to act on broadly.
- **A likely reverse-causality trap is called out explicitly** — total ad exposure correlates strongly with conversion, but users who convert quickly are, by construction, exposed to fewer ads. This project flags that nuance instead of overselling a causal story the data can't support.
- **The Bayesian analysis reframes the decision** from "is this significant?" to "what's the probability we're right, and what do we risk if we're wrong?" — P(ad beats psa) ≈ 100%, with an expected loss near zero if we choose ads and ~0.77 percentage points if we wrongly choose the PSA.

## Repo Structure

```
├── README.md
├── business_writeup.md                    # Plain-English report & recommendation
├── requirements.txt                        # Covers both the analysis and the app
├── AB_Testing_Marketing_Campaign.ipynb     # Full analysis, fully executed with outputs
├── scripts/                                # Each step as a standalone .py file
│   ├── step1_load_clean.py
│   ├── step2_eda.py
│   ├── step3_hypothesis_test.py
│   ├── step4_power_analysis.py
│   ├── step5_segment_analysis.py
│   ├── step6_predictive_model.py
│   ├── step6b_shap_analysis.py
│   └── step7_bayesian_ab_test.py
├── images/                                 # All generated plots (16 total)
└── app/                                     # Live Streamlit demo of the trained model
    ├── app.py
    └── xgb_model.pkl
```

## Live Demo

The `app/` folder contains a Streamlit app serving the trained XGBoost model — adjust ad exposure, day, and hour and see the predicted conversion probability update live. **[Try it here →](https://YOUR-APP-NAME.streamlit.app)**

To deploy your own copy on [Streamlit Community Cloud](https://share.streamlit.io): connect this repo, set the main file path to `app/app.py`, and deploy — it automatically picks up `requirements.txt` from the repo root.

## Dataset

[Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing/data) (Kaggle) — 588,101 rows, 6 columns: `user id`, `test group` (ad/psa), `converted`, `total ads`, `most ads day`, `most ads hour`.

## Tech Stack

`pandas` `numpy` `scipy` `statsmodels` `scikit-learn` `xgboost` `shap` `matplotlib` `seaborn`

## How to Run

**The analysis:**
1. Download `marketing_AB.csv` from the Kaggle link above and place it in the project root
2. `pip install -r requirements.txt`
3. Open `AB_Testing_Marketing_Campaign.ipynb` and run all cells, or run the scripts in `scripts/` in numerical order

**The app, locally:**
1. `pip install -r requirements.txt`
2. `cd app && streamlit run app.py`
