# pipeline.py
import pandas as pd
import streamlit as st

SHEET_ID = "1R92mEDTRsnoQ-6LW-H6wknMRoxMm3wzp0KjPC6XY6to"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

@st.cache_data(ttl=300)  # auto-refresh every 5 minutes
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_URL)

    # Strip ALL column name spaces (fixes the trailing space bug)
    df.columns = df.columns.str.strip()

    # Rename to clean variable names
    df = df.rename(columns={
        "Timestamp":                                                "timestamp",
        "What's your mood today?":                                  "mood",
        "How many hours did you sleep last night? (Number only)":   "sleep_hours",
        "Study/work load today?":                                   "study_load",
        "Short reflection or description of how your day was?":     "reflection",
        "How energetic did you feel today?":                        "energy",
        "How socially connected did you feel today?":               "social_connection",
        "How well were you able to focus today?":                   "focus",
        "How stressed did you feel today?":                         "stress_self",
        "Did anything significantly affect your mood today?":       "mood_influence",
        "Want a personalized Joker message?":                       "want_joker_msg",
        "If yes, what name should I use?":                          "name",
        "If yes, where should I send your Joker message? (Email – optional)": "email",
        "Burnout Risk Score":                                       "burnout_score",
        "Burnout Category":                                         "burnout_category",
    })

    # Drop empty unnamed column
    df = df.drop(columns=["Unnamed: 15"], errors="ignore")

    # Parse timestamp properly (Prophet needs this)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Numeric conversions
    for col in ["mood", "sleep_hours", "energy", "focus", "stress_self"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode study_load
    df["study_load"] = df["study_load"].replace({"Low": 1, "Medium": 2, "High": 3})

    # Encode social_connection
    df["social_connection"] = df["social_connection"].replace({
        "Very connected": 4,
        "Somewhat connected": 3,
        "Neutral": 2,
        "Isolated": 1
    })

    # Compute stress index
    df["stress_index"] = df.apply(_stress_index, axis=1)

    return df


def _stress_index(row) -> float:
    try:
        return round(
            (10 - row["mood"])        * 2.0 +
            max(0, 8 - row["sleep_hours"]) * 1.5 +
            row["study_load"]         * 3.0 +
            (10 - row["energy"])      * 1.2 +
            max(0, 5 - row["social_connection"]) * 1.5 +
            (10 - row["focus"])       * 1.0,
            2
        )
    except Exception:
        return None


from transformers import pipeline as hf_pipeline

@st.cache_resource
def load_emotion_model():
    return hf_pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        top_k=None
    )

def run_emotion_analysis(df: pd.DataFrame) -> pd.DataFrame:
    classifier = load_emotion_model()

    def get_emotion(text):
        if pd.isna(text) or str(text).strip() == "":
            return None
        result = classifier(str(text)[:512])[0]
        return max(result, key=lambda x: x["score"])["label"]

    df = df.copy()
    df["dominant_emotion"] = df["reflection"].apply(get_emotion)
    return df


def get_new_responses_with_email(df):
    """
    Returns rows where:
    - Student opted in for personalized message
    - Email is provided
    """
    mask = (
        df["want_joker_msg"].str.contains("Yes", na=False, case=False) &
        df["email"].notna() &
        df["email"].str.contains("@", na=False)
    )
    return df[mask].to_dict(orient="records")