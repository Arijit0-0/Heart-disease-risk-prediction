# Streamlit Frontend (Framingham CHD Risk)

This project turns your notebook into a simple Streamlit web UI that:

- loads the Framingham dataset (`framingham_dataset.csv`)
- trains a logistic regression model (with scaling + mean-imputation)
- predicts **10-year CHD risk** from 15 input features

## Run locally

Create a virtual environment (recommended), then:

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## Dataset

You have two options:

- **Upload the CSV in the app** (sidebar), or
- **Place it here**: `data/framingham_dataset.csv`

The CSV must contain these columns:

- Features: `male, age, education, currentSmoker, cigsPerDay, BPMeds, prevalentStroke, prevalentHyp, diabetes, totChol, sysBP, diaBP, BMI, heartRate, glucose`
- Target: `TenYearCHD`

