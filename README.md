# AI-Powered 5G Network Service Quality Dashboard

## Project Overview
This project simulates how a telecom company can monitor and improve 5G network service quality using data analytics, machine learning, and interactive dashboards.

It combines:
- Python for data generation, cleaning, transformation, and analysis
- Machine learning for anomaly detection and risk prediction
- Power BI for business-oriented dashboard design

The goal is to replicate a realistic telecom analytics workflow and present insights in a way that supports operational decision-making.

---

## Business Problem
Telecom operators must continuously monitor network performance across regions to maintain service quality and reduce service degradation.

Typical challenges include:
- detecting performance issues early
- identifying high-risk regions
- understanding peak-hour congestion
- tracking outages and anomaly patterns
- communicating technical findings clearly to decision-makers

This project addresses these challenges through an end-to-end workflow from raw data to visual insights.

---

## Project Architecture

### 1. Data Layer (Python)
- Generate a telecom-like dataset
- Clean and transform the data
- Engineer additional features

### 2. AI / Analysis Layer (Python)
- Detect anomalies in network behavior
- Compute a service quality score
- Predict service degradation risk
- Generate short human-readable insights

### 3. Visualization Layer (Power BI)
- Build executive KPI dashboards
- Compare regional performance
- Highlight anomalies and outages
- Present AI-driven insights and risk levels

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

### Derived Features
- `service_quality_score`
- `anomaly_flag`
- `risk_level`

---

## AI Components

### Anomaly Detection
The project uses anomaly detection to identify unusual network behavior such as:
- high latency
- low download speed
- spikes in drop rate
- degraded signal conditions

Model used:
- Isolation Forest

### Risk Prediction
The project predicts whether a region or time period is likely to experience poor service quality.

Model used:
- Random Forest

Target:
- `risk_level` (Low / Medium / High)

### Insight Generation
The pipeline also generates short human-readable insights, for example:
- "Warsaw shows high latency during peak hours."
- "Krakow has the highest drop rate this week."
- "3 regions are at high risk due to congestion and weak signal strength."

---

## Project Structure

```text
5g-quality-dashboard-project/
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
├── pyproject.toml
├── uv.lock
└── README.md