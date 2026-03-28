import pandas as pd
from prophet import Prophet

def prepare_prophet_df(df):
    prophet_df = df[["timestamp", "stress_index"]].copy()
    prophet_df = prophet_df.dropna()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.groupby(prophet_df["ds"].dt.date)["y"].mean().reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["y"] = prophet_df["y"].clip(lower=0, upper=100)
    return prophet_df

def run_forecast(df, forecast_days=7):
    prophet_df = prepare_prophet_df(df)
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        weekly_seasonality=False,
        daily_seasonality=False,
        uncertainty_samples=100
    )
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    forecast["yhat"]       = forecast["yhat"].clip(lower=0, upper=100)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0, upper=100)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0, upper=100)
    return model, forecast, prophet_df

def get_spike_alert(forecast, threshold=38.0):
    future_only = forecast.tail(7)
    spikes = future_only[future_only["yhat"] >= threshold]
    if not spikes.empty:
        worst = spikes.loc[spikes["yhat"].idxmax()]
        return {
            "alert": True,
            "spike_date": worst["ds"].strftime("%A, %b %d"),
            "predicted_stress": round(worst["yhat"], 2),
            "spike_count": len(spikes),
        }
    return {"alert": False}