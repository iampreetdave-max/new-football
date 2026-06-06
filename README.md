# Football Match Predictor

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

A Streamlit app and pipeline that predicts football match outcomes — scorelines, win/draw/win, over/under, and BTTS — from pre-match stats and market data.

## Overview

This project automates an end-to-end football prediction workflow. It pulls
today's fixtures, enriches them with team and league statistics from the
[footystats / football-data-api](https://footystats.org/api) service, builds a
feature set, and runs trained Ridge regression models to predict home and away
goals. From those predicted scores it derives the match outcome (1 / X / 2),
over/under markets, and both-teams-to-score (BTTS).

A Streamlit UI (`app.py`) orchestrates the three pipeline stages, shows file /
model status, runs each step (or the full pipeline), and renders the resulting
CSVs with summary metrics and a high-confidence picks view.

## Key Features

- **Streamlit control panel** — run each stage or the full pipeline, with live status and logs.
- **Automated data fetch** — retrieves today's matches and per-team / per-league stats from the API.
- **Feature engineering** — derives xG, PPG, form, shot accuracy, dangerous attacks, and market-implied features; normalizes probabilities to consistent scales.
- **Goal prediction** — separate Ridge models predict home and away goals using weighted, scaled features.
- **Derived markets** — outcome (1/X/2), over/under 1.5–3.5, fixed over/under 2.5, and BTTS.
- **Confidence categories** — Low / Medium / High based on predicted goal difference.
- **Incremental predictions** — only predicts matches not already in the output file, and cleans up stale predictions.
- **Results viewer** — tabular views and CSV download for live matches, features, and predictions.

## How It Works

The pipeline is a three-step sequence, runnable from the Streamlit UI or directly:

1. **`today_matches.py`** — fetches today's fixtures and writes `live.csv`.
2. **`fetch_data.py`** — reads `live.csv`, calls the football data API for team and
   league stats, engineers features, and writes `extracted_features_complete.csv`.
3. **`predict.py`** — loads the Ridge models and scaler, applies feature weights,
   scales, predicts goals, derives markets, and writes `best_match_predictions.csv`
   (appending only new matches).

Supporting scripts include `league_name.py`, `login_script.py`, `save.py`,
`save_main.py`, `predicting.py`, and validation scripts
(`validate_main.py` / `validate_predictions.py`).

### Model files

`predict.py` loads model artifacts via `joblib` and expects these filenames in
the working directory:

- `ridge_home_model.pkl` — home goals model
- `ridge_away_model.pkl` — away goals model
- `scaler.pkl` — feature scaler

The repository also includes `ml_model.pkl` and `ml_scaler.pkl`; ensure the
filenames the scripts load are present before running predictions.

## Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **ML:** scikit-learn (Ridge regression), joblib
- **Data:** pandas, NumPy
- **HTTP:** requests
- **External data:** football-data-api (footystats)

## Getting Started

### Prerequisites

- Python 3.9+
- A football-data-api (footystats) API key
- The model files required by `predict.py` (see [Model files](#model-files))

### Installation

```bash
git clone https://github.com/iampreetdave-max/new-football.git
cd new-football
pip install -r requirements.txt
```

### Running the app

```bash
streamlit run app.py
```

Use the control panel to run each step in order, or click **Run All Steps** to
execute the full pipeline. You can also run the stages directly:

```bash
python today_matches.py   # -> live.csv
python fetch_data.py      # -> extracted_features_complete.csv
python predict.py         # -> best_match_predictions.csv
```

## Configuration

The data-fetching scripts call the football-data-api and require an **API key**.
Set this through your own configuration before running — do not commit real keys
to the repository. Rotate any key that has been exposed in source history.

## Project Structure

```
new-football/
├── app.py                            # Streamlit UI orchestrating the pipeline
├── today_matches.py                  # Step 1: fetch fixtures -> live.csv
├── fetch_data.py                     # Step 2: enrich + engineer features
├── predict.py                        # Step 3: Ridge models -> predictions
├── predicting.py                     # Alternate prediction script
├── validate_main.py                  # Prediction validation
├── validate_predictions.py           # Prediction validation
├── league_name.py                    # League name lookups
├── login_script.py                   # Auth / session helper
├── save.py / save_main.py            # Persistence helpers
├── ProfitLossCSV.py                  # PnL reporting
├── ml_model.pkl / ml_scaler.pkl      # Model artifacts
├── live.csv                          # Sample fixtures
├── extracted_features_complete.csv   # Sample features
├── best_match_predictions.csv        # Sample predictions
├── requirements.txt
└── LICENSE
```

## Disclaimer

For research and educational use only. Sports betting carries financial risk;
predictions are not guarantees of outcomes.

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file in this repository.
