# AI-Powered 5G Network Service Quality Dashboard

## Project Overview
This project simulates how a telecom company can monitor and improve 5G network service quality using data analytics, machine learning, and dashboarding.

It combines:
- Python for data generation, cleaning, transformation, and AI analysis
- Machine learning for anomaly detection and risk prediction
- Power BI for interactive business-focused dashboards

The goal is to replicate a realistic telecom analytics workflow and present insights in a way that is useful for decision-making.

---

## Business Problem
Telecom operators need to continuously monitor network quality across different regions and time periods.

Typical challenges include:
- detecting performance degradation early
- identifying high-risk regions
- understanding peak-hour congestion
- tracking outages and drop rates
- presenting technical findings clearly to business users

This project addresses these challenges through an end-to-end pipeline from raw data to dashboard insights.

---

## Project Architecture

### 1. Data Layer (Python)
- Generate a telecom-like dataset
- Clean and transform the data
- Create useful derived features

### 2. AI / Analysis Layer (Python)
- Detect anomalies in network behavior
- Compute a service quality score
- Predict service degradation risk
- Generate short human-readable insights

### 3. Visualization Layer (Power BI)
- Build KPI dashboard pages
- Compare regional performance
- Highlight anomalies and outages
- Present AI insights and risk levels

---

## Dataset Description

### Core Fields
- `timestamp`
- `region`
- `base_station_id`
- `technology`

### Performance Metrics
- `latency_ms`
- `download_speed_mbps`
- `upload_speed_mbps`
- `packet_loss_percent`
- `signal_strength_db`

### Usage Metrics
- `active_users`
- `data_usage_gb`

### Reliability Metrics
- `drop_rate_percent`
- `outage_flag`
- `outage_duration_min`

### Derived Fields
- `service_quality_score`
- `anomaly_flag`
- `risk_level`

---

## AI Components

### Anomaly Detection
Anomaly detection is used to identify unusual network behavior such as:
- very high latency
- unusually low speed
- spikes in drop rate
- poor signal conditions

Model used:
- Isolation Forest

### Risk Prediction
Risk prediction estimates whether a region or time period is likely to experience poor service quality.

Model used:
- Random Forest

Target:
- `risk_level` (Low / Medium / High)

### Insight Generation
The project also generates short human-readable insight statements, for example:
- "Warsaw shows high latency during peak hours."
- "Krakow has the highest drop rate this week."
- "3 regions are at high risk due to congestion and weak signal strength."

---

## Project Structure

```text
5g_quality_dashboard_project/
│
├── dashboard/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── screenshots/
├── src/
├── .gitignore
├── README.md
└── requirements.txt