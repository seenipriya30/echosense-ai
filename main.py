import pandas as pd

#############################################
# CONFIG
#############################################

SHEET_ID = "1R92mEDTRsnoQ-6LW-H6wknMRoxMm3wzp0KjPC6XY6to"
CSV_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"

#############################################
# LOAD DATA
#############################################

df = pd.read_csv(CSV_URL)
print("\n🟢 Data loaded successfully!")

# Clean column names (VERY IMPORTANT)
df.columns = df.columns.str.strip()

print("\n🔹 Cleaned Columns:")
print(df.columns)

#############################################
# RENAME COLUMNS PROPERLY
#############################################

df = df.rename(columns={
    "What's your mood today?": "mood",
    "How many hours did you sleep last night? (Number only)": "sleep_hours",
    "Study/work load today?": "study_load",
    "Short reflection or description of how your day was?": "reflection",
    "How energetic did you feel today?": "energy",
    "How socially connected did you feel today?": "social_connection",
    "How well were you able to focus today?": "focus",
    "How stressed did you feel today?": "stress_self",
    "Did anything significantly affect your mood today?": "mood_influence",
    "Want a personalized Joker message?": "want_joker_msg",
    "If yes, what name should I use?": "name",
    "If yes, where should I send your Joker message? (Email – optional)": "email"
})

#############################################
# REMOVE UNUSED COLUMNS
#############################################

df = df.drop(columns=["Unnamed: 15"], errors="ignore")

#############################################
# TYPE CONVERSIONS
#############################################

numeric_cols = ["mood", "sleep_hours", "energy", "focus", "stress_self"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df["study_load"] = df["study_load"].replace({
    "Low": 1,
    "Medium": 2,
    "High": 3
})

df["social_connection"] = df["social_connection"].replace({
    "Very connected": 4,
    "Somewhat connected": 3,
    "Neutral": 2,
    "Isolated": 1
})

#############################################
# STRESS INDEX
#############################################

def calculate_stress_index(row):
    mood_factor = (10 - row["mood"]) * 2
    sleep_penalty = max(0, 8 - row["sleep_hours"]) * 1.5
    workload_score = row["study_load"] * 3
    energy_penalty = (10 - row["energy"]) * 1.2
    social_penalty = max(0, 5 - row["social_connection"]) * 1.5
    focus_penalty = (10 - row["focus"]) * 1.0

    return round(
        mood_factor + sleep_penalty + workload_score +
        energy_penalty + social_penalty + focus_penalty,
        2
    )

df["stress_index"] = df.apply(calculate_stress_index, axis=1)

#############################################
# OUTPUT
#############################################

print("\n📊 Stress Index Preview:")
print(df[[
    "mood", "sleep_hours", "study_load",
    "energy", "social_connection",
    "focus", "stress_self", "stress_index"
]].head())

print("\n📈 Stats:")
print(df[["stress_index", "stress_self"]].describe())

print("\n✅ Script finished successfully.")






from transformers import pipeline

print("\n🧠 Loading Emotion Model... (first time may take 1–2 minutes)")

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

def get_dominant_emotion(text):
    if pd.isna(text):
        return None
    results = emotion_classifier(str(text))
    if results:
        emotions = results[0]
        dominant = max(emotions, key=lambda x: x['score'])
        return dominant['label']
    return None

print("\n🔎 Analyzing reflections for emotions...")

df["dominant_emotion"] = df["reflection"].apply(get_dominant_emotion)

print("\n📊 Emotion Analysis Preview:")
print(df[["reflection", "dominant_emotion"]].head())