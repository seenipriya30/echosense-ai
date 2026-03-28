# dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pipeline import load_data, run_emotion_analysis
from prophet_model import run_forecast, get_spike_alert

st.set_page_config(page_title="EchoSense AI", layout="wide")
st.title("🧠 EchoSense AI – Student Emotional Intelligence Dashboard")

# ── Load ──────────────────────────────────────────────
with st.spinner("Fetching live data..."):
    df = load_data()
    df = run_emotion_analysis(df)

st.caption(f"📡 {len(df)} responses loaded · auto-refreshes every 5 min")

# ── Top Metrics ───────────────────────────────────────
st.subheader("📈 Campus Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Avg Mood",          round(df["mood"].mean(), 2))
c2.metric("Avg Stress (Self)", round(df["stress_self"].mean(), 2))
c3.metric("Avg Sleep (hrs)",   round(df["sleep_hours"].mean(), 2))
c4.metric("Avg Stress Index",  round(df["stress_index"].mean(), 2))

# ── Alerts ────────────────────────────────────────────
st.subheader("🚨 Live Insights")
avg_stress  = df["stress_self"].mean()
avg_sleep   = df["sleep_hours"].mean()
avg_mood    = df["mood"].mean()
top_emotion = df["dominant_emotion"].value_counts().idxmax()

if avg_stress > 7:
    st.error("⚠️ Community stress levels are critically high")
elif avg_stress > 5:
    st.warning("⚠️ Moderate stress detected across campus")
else:
    st.success("✅ Stress levels appear manageable")

if avg_sleep < 6:
    st.warning("😴 Sleep deprivation detected (avg < 6 hrs)")
if avg_mood < 4:
    st.warning("📉 Overall mood is low")
if top_emotion in ["sadness", "anger", "fear"]:
    st.warning(f"🧠 Dominant emotion detected: **{top_emotion}**")

# ── Charts Row 1 ──────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Mood Distribution")
    fig = px.histogram(df, x="mood", nbins=10,
                       color_discrete_sequence=["#636EFA"])
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Stress Distribution")
    fig2 = px.histogram(df, x="stress_self", nbins=10,
                        color_discrete_sequence=["#EF553B"])
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts Row 2 ──────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("🧠 Emotion Breakdown (AI)")
    ec = df["dominant_emotion"].value_counts().reset_index()
    ec.columns = ["emotion", "count"]
    fig3 = px.bar(ec, x="emotion", y="count", color="emotion")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Stress Index vs Self-Reported Stress")
    fig4 = px.scatter(df, x="stress_self", y="stress_index",
                      color="dominant_emotion",
                      hover_data=["mood", "sleep_hours"],
                      trendline="ols")
    st.plotly_chart(fig4, use_container_width=True)

# ── Forecast Section ──────────────────────────────────
st.subheader("📈 7-Day Stress Forecast")

with st.spinner("Running forecast model..."):
    model, forecast, historical = run_forecast(df)

alert = get_spike_alert(forecast)

if alert["alert"]:
    st.error(
        f"⚠️ Stress spike predicted on **{alert['spike_date']}** "
        f"— Score: `{alert['predicted_stress']}` "
        f"({alert['spike_count']} high-stress day(s) ahead)"
    )
else:
    st.success("✅ No major stress spikes predicted this week.")

# Build forecast chart
forecast_only = forecast.tail(7)  # ✅ define BEFORE using it

fig5 = go.Figure()

fig5.add_trace(go.Scatter(
    x=historical["ds"], y=historical["y"],
    mode="lines+markers", name="Historical Stress",
    line=dict(color="#636EFA")
))

fig5.add_trace(go.Scatter(
    x=forecast_only["ds"], y=forecast_only["yhat"],
    mode="lines+markers", name="Forecasted Stress",
    line=dict(color="#EF553B", dash="dash")
))

fig5.add_trace(go.Scatter(
    x=list(forecast_only["ds"]) + list(forecast_only["ds"][::-1]),
    y=list(forecast_only["yhat_upper"]) + list(forecast_only["yhat_lower"][::-1]),
    fill="toself",
    fillcolor="rgba(239,85,59,0.1)",
    line=dict(color="rgba(255,255,255,0)"),
    name="Confidence Band"
))

fig5.update_layout(
    title="Campus Stress Index — History + 7-Day Forecast",
    xaxis_title="Date",
    yaxis_title="Stress Score",
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig5, use_container_width=True)

with st.expander("📋 Forecast Data"):
    fc_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(7).copy()
    fc_display.columns = ["Date", "Predicted", "Lower", "Upper"]
    fc_display["Date"] = fc_display["Date"].dt.strftime("%A, %b %d")
    st.dataframe(fc_display.round(2), use_container_width=True)

# ── Raw Data ──────────────────────────────────────────
with st.expander("📋 View Raw Responses"):
    st.dataframe(df[[
        "timestamp", "mood", "sleep_hours", "study_load",
        "stress_self", "stress_index", "dominant_emotion", "reflection"
    ]], use_container_width=True)