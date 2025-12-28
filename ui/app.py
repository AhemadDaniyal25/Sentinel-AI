import sys
from pathlib import Path

# -------------------------------------------------------------------
# Make project root importable (so `src` works when running Streamlit)
# -------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import streamlit as st
import numpy as np

from src.models.isolation_forest import IsolationForestAnomalyDetector
from src.decision.decision_engine import DecisionEngine


# -------------------------------------------------------------------
# Streamlit page setup
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SentinelAI Demo",
    layout="centered"
)

st.title("SentinelAI — Industrial Anomaly Detection Demo")

st.markdown(
    """
This interactive demo shows **how an industrial AI system converts raw sensor readings
into risk-aware decisions**.

You can:
- Adjust sensor values
- Observe how anomaly risk changes
- See how decision logic suppresses false alarms
"""
)


# -------------------------------------------------------------------
# Initialize model and decision logic
# -------------------------------------------------------------------
model = IsolationForestAnomalyDetector()
decision_engine = DecisionEngine(
    score_threshold=0.7,
    persistence_windows=3
)

# Train model on synthetic *normal* operating data
np.random.seed(42)
normal_operation_data = np.random.normal(0, 1, size=(300, 4))
model.fit(normal_operation_data)


# -------------------------------------------------------------------
# Sidebar: Sensor Inputs
# -------------------------------------------------------------------
st.sidebar.header("Sensor Inputs")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=0.0,
    max_value=120.0,
    value=60.0
)

vibration = st.sidebar.slider(
    "Vibration (mm/s)",
    min_value=0.0,
    max_value=5.0,
    value=0.5
)

pressure = st.sidebar.slider(
    "Pressure (bar)",
    min_value=0.0,
    max_value=10.0,
    value=3.0
)

rpm = st.sidebar.slider(
    "Rotational Speed (RPM)",
    min_value=500.0,
    max_value=3000.0,
    value=1500.0
)


# -------------------------------------------------------------------
# Feature normalization (critical for correct ML behavior)
# -------------------------------------------------------------------
# Each sensor is normalized using its expected operating range
temperature_norm = temperature / 120.0
vibration_norm = vibration / 5.0
pressure_norm = pressure / 10.0
rpm_norm = rpm / 3000.0

X = np.array([
    [temperature_norm, vibration_norm, pressure_norm, rpm_norm]
])


# -------------------------------------------------------------------
# Model inference
# -------------------------------------------------------------------
raw_anomaly_score = model.score(X)[0]

# Convert raw score into a human-friendly risk score [0, 1]
risk_score = 1 / (1 + np.exp(-raw_anomaly_score))
st.caption("Risk Score ∈ [0, 1] — higher means more anomalous behavior")


# Decision-aware alerting (suppresses noisy spikes)
alert = decision_engine.evaluate([risk_score])[-1]


# -------------------------------------------------------------------
# Main output
# -------------------------------------------------------------------
st.subheader("System Output")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Risk Score",
        round(float(risk_score), 3),
        help="Normalized anomaly risk (0 = normal, 1 = highly anomalous)"
    )

with col2:
    st.metric(
        "Risk Level",
        "HIGH" if alert else "LOW"
    )


st.markdown("### Recommended Action")

if alert:
    st.error("Inspect equipment immediately")
else:
    st.success("No action required")


# -------------------------------------------------------------------
# Transparency / explanation
# -------------------------------------------------------------------
with st.expander("How the system makes this decision"):
    st.markdown(
        f"""
**Step 1 – Normalization**  
Raw sensor values are normalized using expected operating ranges.

**Step 2 – Anomaly Detection**  
Isolation Forest computes an anomaly score based on deviation from normal behavior.

**Step 3 – Risk Transformation**  
Raw model output is converted into a normalized risk score.

**Step 4 – Decision Logic**  
Alerts are triggered only if risk remains high over multiple observations.

**Raw model score:** `{raw_anomaly_score:.3f}`
"""
    )

st.caption(
    "Demo system — illustrates decision-aware industrial AI behavior, not a production deployment"
)
