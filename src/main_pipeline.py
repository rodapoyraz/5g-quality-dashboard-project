
"""
AI-Powered 5G Network Service Quality Dashboard
------------------------------------------------
End-to-end Python pipeline for a telecom analytics portfolio project.

What this script does:
1. Generates realistic telecom-like network data
2. Cleans and transforms the dataset
3. Engineers time and performance features
4. Creates a Service Quality Score (0-100)
5. Detects anomalies with Isolation Forest
6. Predicts risk level with Random Forest
7. Generates short human-readable AI insights
8. Exports clean CSVs for Power BI

Outputs:
- network_metrics.csv
- aggregated_metrics.csv
- ai_insights.csv
- model_report.csv
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@dataclass
class Config:
    random_seed: int = 42
    days: int = 60
    base_stations_per_region: int = 4
    output_dir: str = "outputs"


def generate_dataset(cfg: Config) -> pd.DataFrame:
    """Generate realistic telecom-like hourly network data."""
    rng = np.random.default_rng(cfg.random_seed)

    regions = ["Wroclaw", "Warsaw", "Krakow", "Gdansk", "Poznan", "Lodz", "Katowice"]
    technologies = ["4G", "5G"]

    timestamps = pd.date_range("2026-01-01", periods=cfg.days * 24, freq="h")

    base_station_region = {}
    for region in regions:
        for i in range(cfg.base_stations_per_region):
            base_station_region[f"{region[:3].upper()}_{i+1:02d}"] = region

    rows = []

    for ts in timestamps:
        hour = ts.hour
        day_of_week_num = ts.dayofweek
        is_peak = int(hour in [7, 8, 9, 17, 18, 19, 20, 21])
        is_weekend = int(day_of_week_num >= 5)

        for base_station_id, region in base_station_region.items():
            technology = rng.choice(technologies, p=[0.35, 0.65])

            region_factor = {
                "Warsaw": 1.15,
                "Wroclaw": 1.00,
                "Krakow": 1.08,
                "Gdansk": 0.98,
                "Poznan": 0.95,
                "Lodz": 1.05,
                "Katowice": 1.02,
            }[region]

            tech_factor = 0.78 if technology == "5G" else 1.15

            active_users = max(
                5,
                int(
                    np.random.normal(85 if is_peak else 40, 20)
                    * region_factor
                    * (1.1 if not is_weekend else 0.9)
                ),
            )
            active_users = min(active_users, 220)
            congestion = active_users / 220

            latency_ms = np.clip(
                np.random.normal(30 * tech_factor + 45 * congestion + (12 if is_peak else 0), 10),
                10,
                200,
            )
            download_speed_mbps = np.clip(
                np.random.normal((170 if technology == "5G" else 55) - 70 * congestion - (12 if is_peak else 0), 25),
                5,
                300,
            )
            upload_speed_mbps = np.clip(
                np.random.normal((55 if technology == "5G" else 18) - 15 * congestion, 8),
                2,
                120,
            )
            packet_loss_percent = np.clip(
                np.random.normal(0.6 + 2.7 * congestion + (0.8 if is_peak else 0), 0.9),
                0,
                20,
            )
            signal_strength_db = np.clip(
                np.random.normal((-82 if technology == "5G" else -88) - 10 * congestion, 7),
                -110,
                -50,
            )
            data_usage_gb = np.clip(
                np.random.normal(active_users * (0.065 if technology == "5G" else 0.04), 1.2),
                0.2,
                30,
            )
            drop_rate_percent = np.clip(
                np.random.normal(0.8 + 2.8 * congestion + (0.6 if signal_strength_db < -98 else 0), 0.8),
                0,
                15,
            )

            outage_probability = (
                0.003
                + 0.02 * (congestion > 0.82)
                + 0.01 * (signal_strength_db < -102)
                + 0.008 * (drop_rate_percent > 5)
            )
            outage_flag = int(rng.random() < outage_probability)
            outage_duration_min = (
                int(np.clip(np.random.normal(18 + 30 * congestion, 12), 1, 180))
                if outage_flag
                else 0
            )

            # Inject occasional extreme network events
            if rng.random() < 0.012:
                latency_ms = np.clip(latency_ms * rng.uniform(1.8, 3.2), 10, 200)
                download_speed_mbps = np.clip(download_speed_mbps * rng.uniform(0.08, 0.45), 5, 300)
                packet_loss_percent = np.clip(packet_loss_percent + rng.uniform(2, 8), 0, 20)
                drop_rate_percent = np.clip(drop_rate_percent + rng.uniform(2, 7), 0, 15)
                outage_flag = max(outage_flag, int(rng.random() < 0.25))
                outage_duration_min = max(outage_duration_min, int(rng.integers(5, 60))) if outage_flag else 0

            rows.append(
                [
                    ts,
                    region,
                    base_station_id,
                    technology,
                    latency_ms,
                    download_speed_mbps,
                    upload_speed_mbps,
                    packet_loss_percent,
                    signal_strength_db,
                    active_users,
                    data_usage_gb,
                    drop_rate_percent,
                    outage_flag,
                    outage_duration_min,
                ]
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "region",
            "base_station_id",
            "technology",
            "latency_ms",
            "download_speed_mbps",
            "upload_speed_mbps",
            "packet_loss_percent",
            "signal_strength_db",
            "active_users",
            "data_usage_gb",
            "drop_rate_percent",
            "outage_flag",
            "outage_duration_min",
        ],
    )

    # Inject a small amount of missing values to make the cleaning step realistic
    for col in [
        "latency_ms",
        "download_speed_mbps",
        "upload_speed_mbps",
        "packet_loss_percent",
        "signal_strength_db",
        "active_users",
    ]:
        missing_idx = rng.choice(df.index, size=int(0.008 * len(df)), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types and fill missing values."""
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    numeric_cols = [
        "latency_ms",
        "download_speed_mbps",
        "upload_speed_mbps",
        "packet_loss_percent",
        "signal_strength_db",
        "active_users",
        "data_usage_gb",
        "drop_rate_percent",
        "outage_duration_min",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in [
        "latency_ms",
        "download_speed_mbps",
        "upload_speed_mbps",
        "packet_loss_percent",
        "signal_strength_db",
        "active_users",
    ]:
        df[col] = df.groupby(["region", "technology"])[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time features and service quality score."""
    df = df.copy()

    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_peak_hour"] = df["hour_of_day"].isin([7, 8, 9, 17, 18, 19, 20, 21]).astype(int)
    df["date"] = df["timestamp"].dt.date

    # Normalize metrics into 0-100 score components
    latency_score = 100 * (1 - (df["latency_ms"] - 10) / (200 - 10))
    speed_score = 100 * ((df["download_speed_mbps"] - 5) / (300 - 5))
    drop_score = 100 * (1 - df["drop_rate_percent"] / 15)
    signal_score = 100 * ((df["signal_strength_db"] + 110) / 60)

    df["service_quality_score"] = (
        latency_score.clip(0, 100) * 0.30
        + speed_score.clip(0, 100) * 0.35
        + drop_score.clip(0, 100) * 0.20
        + signal_score.clip(0, 100) * 0.15
    ).round(2)

    return df


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.03) -> pd.DataFrame:
    """Detect unusual network behavior with Isolation Forest."""
    df = df.copy()

    anomaly_features = df[
        [
            "latency_ms",
            "download_speed_mbps",
            "packet_loss_percent",
            "drop_rate_percent",
            "signal_strength_db",
            "active_users",
            "outage_flag",
        ]
    ].copy()

    model = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    predictions = model.fit_predict(anomaly_features)
    df["anomaly_flag"] = (predictions == -1).astype(int)

    return df


def create_risk_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Create heuristic target labels for service risk level."""
    df = df.copy()

    risk_score = (
        0.28 * (df["latency_ms"] / 200)
        + 0.20 * (1 - ((df["signal_strength_db"] + 110) / 60).clip(0, 1))
        + 0.22 * (df["drop_rate_percent"] / 15)
        + 0.10 * (df["packet_loss_percent"] / 20)
        + 0.10 * (df["active_users"] / 220)
        + 0.10 * df["is_peak_hour"]
        + 0.12 * df["outage_flag"]
        + 0.10 * df["anomaly_flag"]
    ).clip(0, 1.5)

    df["risk_level"] = pd.cut(
        risk_score,
        bins=[-0.01, 0.38, 0.68, 10],
        labels=["Low", "Medium", "High"],
    )

    return df


def predict_risk(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train a Random Forest model to predict service risk."""
    df = df.copy()

    le_region = LabelEncoder()
    le_tech = LabelEncoder()
    le_risk = LabelEncoder()

    X = pd.DataFrame(
        {
            "latency_ms": df["latency_ms"],
            "active_users": df["active_users"],
            "signal_strength_db": df["signal_strength_db"],
            "drop_rate_percent": df["drop_rate_percent"],
            "packet_loss_percent": df["packet_loss_percent"],
            "hour_of_day": df["hour_of_day"],
            "technology_enc": le_tech.fit_transform(df["technology"]),
            "region_enc": le_region.fit_transform(df["region"]),
            "anomaly_flag": df["anomaly_flag"],
            "outage_flag": df["outage_flag"],
        }
    )
    y = le_risk.fit_transform(df["risk_level"].astype(str))

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        max_depth=12,
        class_weight="balanced_subsample",
    )
    model.fit(X_train, y_train)

    df["predicted_risk_level"] = le_risk.inverse_transform(model.predict(X))

    report = classification_report(
        y_test,
        model.predict(X_test),
        target_names=le_risk.classes_,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()

    return df, report_df


def create_aggregated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Build regional and daily aggregates for Power BI."""
    aggregated = (
        df.groupby(["date", "region", "technology"], as_index=False)
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            avg_download_speed_mbps=("download_speed_mbps", "mean"),
            avg_upload_speed_mbps=("upload_speed_mbps", "mean"),
            avg_signal_strength_db=("signal_strength_db", "mean"),
            avg_drop_rate_percent=("drop_rate_percent", "mean"),
            avg_packet_loss_percent=("packet_loss_percent", "mean"),
            avg_service_quality_score=("service_quality_score", "mean"),
            total_outages=("outage_flag", "sum"),
            avg_outage_duration_min=("outage_duration_min", "mean"),
            anomaly_count=("anomaly_flag", "sum"),
            avg_active_users=("active_users", "mean"),
            total_data_usage_gb=("data_usage_gb", "sum"),
        )
    )

    aggregated["risk_level_mode"] = (
        df.groupby(["date", "region", "technology"])["risk_level"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else "Low")
        .values
    )

    return aggregated


def generate_ai_insights(df: pd.DataFrame) -> pd.DataFrame:
    """Create short human-readable business insights."""
    insights: list[list[str]] = []

    peak_latency = (
        df[df["is_peak_hour"] == 1]
        .groupby("region")["latency_ms"]
        .mean()
        .sort_values(ascending=False)
    )
    if not peak_latency.empty:
        top_region = peak_latency.index[0]
        insights.append(
            [
                f"Region {top_region} shows the highest average latency during peak hours.",
                top_region,
                "High",
            ]
        )

    drop_rate = df.groupby("region")["drop_rate_percent"].mean().sort_values(ascending=False)
    insights.append(
        [
            f"{drop_rate.index[0]} has the highest average drop rate in the dataset.",
            drop_rate.index[0],
            "High",
        ]
    )

    high_risk_counts = df[df["risk_level"] == "High"].groupby("region").size().sort_values(ascending=False)
    if not high_risk_counts.empty:
        top_regions = ", ".join(high_risk_counts.head(3).index.tolist())
        insights.append(
            [
                f"3 regions are repeatedly at high risk due to congestion and weak signal: {top_regions}.",
                "Multiple",
                "High",
            ]
        )

    anomaly_by_tech = df.groupby("technology")["anomaly_flag"].mean().sort_values(ascending=False)
    insights.append(
        [
            f"{anomaly_by_tech.index[0]} sites show the higher anomaly rate and should be reviewed for capacity balancing.",
            "All",
            "Medium",
        ]
    )

    outages = df.groupby("region")["outage_flag"].sum().sort_values(ascending=False)
    insights.append(
        [
            f"{outages.index[0]} recorded the most outages and may need infrastructure resilience checks.",
            outages.index[0],
            "High",
        ]
    )

    best_quality = df.groupby("region")["service_quality_score"].mean().sort_values(ascending=False)
    insights.append(
        [
            f"{best_quality.index[0]} currently has the strongest overall service quality score.",
            best_quality.index[0],
            "Low",
        ]
    )

    tech_quality = df.groupby("technology")["service_quality_score"].mean().sort_values(ascending=False)
    insights.append(
        [
            f"{tech_quality.index[0]} delivers the better average service quality score across the network.",
            "All",
            "Medium",
        ]
    )

    df["date_dt"] = pd.to_datetime(df["date"])
    max_date = df["date_dt"].max()
    last_7_days = df[df["date_dt"] > max_date - pd.Timedelta(days=7)]
    prev_7_days = df[
        (df["date_dt"] <= max_date - pd.Timedelta(days=7))
        & (df["date_dt"] > max_date - pd.Timedelta(days=14))
    ]

    if not last_7_days.empty and not prev_7_days.empty:
        delta = (
            last_7_days.groupby("region")["service_quality_score"].mean()
            - prev_7_days.groupby("region")["service_quality_score"].mean()
        )
        declining_region = delta.sort_values().index[0]
        insights.append(
            [
                f"{declining_region} shows the sharpest recent decline in service quality compared with the previous week.",
                declining_region,
                "Medium",
            ]
        )

    return pd.DataFrame(insights, columns=["insight_text", "region", "priority_level"])


def export_outputs(
    df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    insights_df: pd.DataFrame,
    model_report_df: pd.DataFrame,
    cfg: Config,
) -> None:
    """Save CSV outputs for Power BI."""
    os.makedirs(cfg.output_dir, exist_ok=True)

    df.to_csv(os.path.join(cfg.output_dir, "network_metrics.csv"), index=False)
    aggregated_df.to_csv(os.path.join(cfg.output_dir, "aggregated_metrics.csv"), index=False)
    insights_df.to_csv(os.path.join(cfg.output_dir, "ai_insights.csv"), index=False)
    model_report_df.to_csv(os.path.join(cfg.output_dir, "model_report.csv"))


def main() -> None:
    cfg = Config()

    print("Generating telecom dataset...")
    df = generate_dataset(cfg)

    print("Cleaning dataset...")
    df = clean_and_transform(df)

    print("Engineering features...")
    df = engineer_features(df)

    print("Detecting anomalies...")
    df = detect_anomalies(df)

    print("Creating risk levels...")
    df = create_risk_levels(df)

    print("Training risk prediction model...")
    df, model_report_df = predict_risk(df)

    print("Creating aggregated metrics...")
    aggregated_df = create_aggregated_metrics(df)

    print("Generating AI insights...")
    insights_df = generate_ai_insights(df)

    print("Exporting CSV files...")
    export_outputs(df, aggregated_df, insights_df, model_report_df, cfg)

    print("\nDone.")
    print(f"Rows in network_metrics.csv: {len(df):,}")
    print(f"Rows in aggregated_metrics.csv: {len(aggregated_df):,}")
    print(f"Rows in ai_insights.csv: {len(insights_df):,}")
    print(f"Average service quality score: {df['service_quality_score'].mean():.2f}")
    print(f"Anomaly rate: {df['anomaly_flag'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
