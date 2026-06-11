import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostClassifier
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="NovaHeart · Clinical Decision Support",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# CUSTOM CSS — Dark clinical aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-elevated: #1a2236;
    --accent-red: #ef4444;
    --accent-blue: #3b82f6;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-purple: #8b5cf6;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e2d45;
    --glow-red: rgba(239, 68, 68, 0.15);
    --glow-blue: rgba(59, 130, 246, 0.15);
    --glow-green: rgba(16, 185, 129, 0.15);
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }

/* All input widgets */
.stTextInput input, .stNumberInput input, .stSelectbox select,
div[data-baseweb="input"] input, div[data-baseweb="select"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: var(--accent-blue) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.5) !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
div[data-testid="stMetric"] label { color: var(--text-secondary) !important; font-size: 0.8rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-size: 1.8rem !important; font-weight: 700 !important; }

/* Tabs */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
    background: transparent !important;
}

/* Expander */
details {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }

/* Custom card component */
.nova-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.risk-banner {
    border-radius: 14px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    border: 1px solid;
}
.risk-critical { background: rgba(239,68,68,0.1); border-color: rgba(239,68,68,0.4); }
.risk-high { background: rgba(245,158,11,0.1); border-color: rgba(245,158,11,0.4); }
.risk-moderate { background: rgba(249,115,22,0.08); border-color: rgba(249,115,22,0.35); }
.risk-low { background: rgba(16,185,129,0.1); border-color: rgba(16,185,129,0.4); }

.pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.pill-red { background: rgba(239,68,68,0.2); color: #ef4444; border: 1px solid rgba(239,68,68,0.4); }
.pill-green { background: rgba(16,185,129,0.2); color: #10b981; border: 1px solid rgba(16,185,129,0.4); }
.pill-blue { background: rgba(59,130,246,0.2); color: #3b82f6; border: 1px solid rgba(59,130,246,0.4); }
.pill-amber { background: rgba(245,158,11,0.2); color: #f59e0b; border: 1px solid rgba(245,158,11,0.4); }

.section-header {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #3b82f6;
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
}

.mono { font-family: 'JetBrains Mono', monospace; }

.alert-box {
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.3);
    border-left: 4px solid #ef4444;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}

.action-box {
    background: rgba(16,185,129,0.07);
    border: 1px solid rgba(16,185,129,0.3);
    border-left: 4px solid #10b981;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="Booting NovaHeart Engine…")
def load_artifacts():
    imputer = joblib.load("clinical_knn_imputer.pkl")
    model = CatBoostClassifier()
    model.load_model("heart_failure_catboost_core.cbm")
    with open("feature_schema_lock.json") as f:
        schema = json.load(f)
    return imputer, model, schema

imputer, model, schema = load_artifacts()

NUMERIC_COLS   = schema["numeric_features"]
CATEGORICAL_COLS = schema["categorical_features"]
FEATURE_ORDER  = schema["strict_feature_order"]

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────
def compute_care_plan(patient: dict, risk_prob: float) -> dict:
    ef = patient.get("EF-TTE", 55)
    fc = patient.get("Function Class", 1)
    edema = patient.get("Edema", 0)
    lung_rales = patient.get("Lung rales", 0)

    alerts, actions, risk_class = [], [], ""

    if risk_prob >= 0.85:
        risk_class = "CRITICAL"
        alerts.append("Active cardiovascular hazard state flagged by ML Core.")
    elif risk_prob >= 0.65:
        risk_class = "HIGH"
        alerts.append("Elevated CAD risk — urgent cardiology consult recommended.")
    elif risk_prob >= 0.40:
        risk_class = "MODERATE"
        alerts.append("Intermediate CAD risk — monitoring and lifestyle intervention advised.")
    else:
        risk_class = "LOW"
        alerts.append("Low CAD risk profile. Continue preventive care.")

    if ef < 40:
        alerts.append(f"HFrEF Detected: EF {ef}% — significantly reduced. Guideline-directed therapy indicated.")
        actions.append("Initiate ACEi/ARB + Beta-blocker + MRA therapy per ACC/AHA HFrEF guidelines.")
    elif ef < 55:
        alerts.append(f"HFpEF Condition Detected (Preserved EF: {ef}%).")
        actions.append("Focus on optimizing underlying comorbidities (e.g., Hypertension control).")

    if fc >= 3:
        alerts.append(f"NYHA Class {fc}: Severe functional limitation. Minimal exertion triggers symptoms.")
        actions.append("Refer for advanced HF therapies — consider CRT-D or transplant evaluation.")
    elif fc == 2:
        actions.append("Supervised cardiac rehabilitation program recommended.")

    if edema and lung_rales:
        alerts.append("Biventricular congestion signs: Peripheral edema + Lung rales detected.")
        actions.append("Urgent diuresis assessment. Consider IV Furosemide if hospitalized.")

    if patient.get("DM", 0) and patient.get("HTN", 0):
        actions.append("Aggressive cardiometabolic risk factor management: HbA1c target <7%, BP <130/80.")

    if not actions:
        actions.append("Maintain regular cardiovascular follow-up every 6 months.")
        actions.append("Lifestyle modification: Mediterranean diet, ≥150 min/week aerobic exercise.")

    return {"risk_class": risk_class, "alerts": alerts, "actions": actions}


def build_patient_df(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
    df = df.reindex(columns=FEATURE_ORDER)
    df[NUMERIC_COLS] = imputer.transform(df[NUMERIC_COLS])
    return df


def risk_color(cls):
    return {"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MODERATE": "#f97316", "LOW": "#10b981"}.get(cls, "#94a3b8")


def gauge_chart(prob):
    pct = prob * 100
    color = "#ef4444" if pct >= 65 else "#f59e0b" if pct >= 40 else "#10b981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": "#f1f5f9", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#64748b"}, "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1a2236",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.12)"},
                {"range": [40, 65], "color": "rgba(245,158,11,0.12)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": pct},
        },
    ))
    fig.update_layout(
        margin=dict(t=20, b=10, l=20, r=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#f1f5f9"},
    )
    return fig


def radar_chart(inputs):
    keys = ["Age", "BMI", "BP", "EF-TTE", "LDL", "FBS"]
    norms = {"Age": 100, "BMI": 40, "BP": 200, "EF-TTE": 70, "LDL": 200, "FBS": 200}
    vals = [min((inputs.get(k, 0) / norms[k]) * 100, 100) for k in keys]
    labels = ["Age", "BMI", "Blood\nPressure", "Ejection\nFraction", "LDL\nCholesterol", "Blood\nSugar"]
    fig = go.Figure(go.Scatterpolar(
        r=vals + [vals[0]],
        theta=labels + [labels[0]],
        fill='toself',
        fillcolor='rgba(59,130,246,0.15)',
        line=dict(color='#3b82f6', width=2),
        marker=dict(color='#3b82f6', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#1e2d45', tickfont=dict(color='#64748b', size=9)),
            angularaxis=dict(gridcolor='#1e2d45', tickfont=dict(color='#94a3b8', size=10)),
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=50, r=50),
        height=280,
        font=dict(family='Inter', color='#f1f5f9'),
    )
    return fig


def feature_bar_chart(inputs):
    numeric_vals = {k: inputs[k] for k in NUMERIC_COLS if k in inputs}
    df = pd.DataFrame(list(numeric_vals.items()), columns=["Feature", "Value"])
    fig = go.Figure(go.Bar(
        x=df["Value"],
        y=df["Feature"],
        orientation='h',
        marker=dict(
            color=df["Value"],
            colorscale=[[0, '#1d4ed8'], [0.5, '#7c3aed'], [1, '#ef4444']],
            showscale=False,
        ),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=480,
        margin=dict(t=10, b=10, l=120, r=20),
        xaxis=dict(gridcolor='#1e2d45', color='#64748b'),
        yaxis=dict(color='#94a3b8', tickfont=dict(size=11)),
        font=dict(family='Inter', color='#f1f5f9'),
    )
    return fig


# ─────────────────────────────────────────
# SIDEBAR — PATIENT DATA ENTRY
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.2rem;'>🫀</div>
        <div style='font-size:1.1rem; font-weight:700; letter-spacing:1px; color:#f1f5f9;'>NOVAHEART</div>
        <div style='font-size:0.72rem; color:#64748b; letter-spacing:2px; text-transform:uppercase;'>Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    age    = st.slider("Age (years)", 20, 90, 55)
    sex    = st.selectbox("Sex", ["Male", "Female"])
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, step=0.5)
    height = st.number_input("Height / Length (cm)", 100.0, 220.0, 170.0, step=0.5)
    bmi    = round(weight / (height / 100) ** 2, 2)
    st.caption(f"Auto-calculated BMI: **{bmi}**")

    st.markdown('<div class="section-header" style="margin-top:1rem;">Vital Signs</div>', unsafe_allow_html=True)
    bp  = st.slider("Blood Pressure — Systolic (mmHg)", 70, 220, 120)
    pr  = st.slider("Pulse Rate (bpm)", 40, 180, 75)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Cardiac Function</div>', unsafe_allow_html=True)
    ef_tte = st.slider("Ejection Fraction — EF-TTE (%)", 10, 80, 55)
    fc     = st.selectbox("NYHA Function Class", [0, 1, 2, 3, 4], index=1)
    region_rwma = st.selectbox("Region RWMA", [0, 1, 2, 3, 4])

    st.markdown('<div class="section-header" style="margin-top:1rem;">Lab Values</div>', unsafe_allow_html=True)
    fbs  = st.number_input("Fasting Blood Sugar (mg/dL)", 50.0, 500.0, 100.0)
    cr   = st.number_input("Creatinine (mg/dL)", 0.3, 15.0, 1.0, step=0.1)
    tg   = st.number_input("Triglycerides (mg/dL)", 30.0, 800.0, 150.0)
    ldl  = st.number_input("LDL Cholesterol (mg/dL)", 20.0, 300.0, 120.0)
    hdl  = st.number_input("HDL Cholesterol (mg/dL)", 10.0, 120.0, 45.0)
    bun  = st.number_input("BUN (mg/dL)", 3.0, 150.0, 15.0)
    esr  = st.number_input("ESR (mm/hr)", 1.0, 150.0, 15.0)
    hb   = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 14.0, step=0.1)
    k    = st.number_input("Potassium K (mEq/L)", 2.0, 8.0, 4.2, step=0.1)
    na   = st.number_input("Sodium Na (mEq/L)", 110.0, 170.0, 140.0)
    wbc  = st.number_input("WBC (cells/µL)", 1000.0, 30000.0, 7500.0, step=100.0)
    lymph = st.number_input("Lymphocytes (%)", 5.0, 80.0, 30.0)
    neut  = st.number_input("Neutrophils (%)", 10.0, 95.0, 60.0)
    plt_val = st.number_input("Platelets (×10³/µL)", 50.0, 800.0, 250.0)

    st.markdown('<div class="section-header" style="margin-top:1rem;">Comorbidities</div>', unsafe_allow_html=True)
    col1s, col2s = st.columns(2)
    with col1s:
        dm      = st.checkbox("Diabetes (DM)")
        htn     = st.checkbox("Hypertension")
        obesity = st.checkbox("Obesity")
        crf     = st.checkbox("Chronic Renal Failure")
        cva     = st.checkbox("CVA / Stroke")
        chf     = st.checkbox("CHF")
    with col2s:
        fh        = st.checkbox("Family History")
        smoker    = st.checkbox("Current Smoker")
        ex_smoker = st.checkbox("Ex-Smoker")
        airway    = st.checkbox("Airway Disease")
        thyroid   = st.checkbox("Thyroid Disease")
        dlp       = st.checkbox("Dyslipidemia")

    st.markdown('<div class="section-header" style="margin-top:1rem;">Symptoms & ECG</div>', unsafe_allow_html=True)
    col3s, col4s = st.columns(2)
    with col3s:
        typical_cp = st.checkbox("Typical Chest Pain")
        dyspnea    = st.checkbox("Dyspnea")
        atypical   = st.checkbox("Atypical Chest Pain")
        nonanginal = st.checkbox("Nonanginal Pain")
        exertional = st.checkbox("Exertional CP")
        lowth      = st.checkbox("Low Threshold Angina")
        edema      = st.checkbox("Edema")
        weak_pp    = st.checkbox("Weak Peripheral Pulse")
        lung_rales = st.checkbox("Lung Rales")
    with col4s:
        sys_mur    = st.checkbox("Systolic Murmur")
        dia_mur    = st.checkbox("Diastolic Murmur")
        q_wave     = st.checkbox("Q Wave")
        st_elev    = st.checkbox("ST Elevation")
        st_dep     = st.checkbox("ST Depression")
        t_inv      = st.checkbox("T-wave Inversion")
        lvh        = st.checkbox("LVH")
        poor_r     = st.checkbox("Poor R Progression")

    st.markdown('<div class="section-header" style="margin-top:1rem;">VHD Severity</div>', unsafe_allow_html=True)
    vhd_label = st.selectbox("Valvular Heart Disease", ["None", "Mild", "Moderate", "Severe"])
    vhd_map   = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
    vhd_val   = vhd_map[vhd_label]

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔬  Run Cardiac Analysis", use_container_width=True)


# ─────────────────────────────────────────
# ASSEMBLE PATIENT INPUT DICT
# ─────────────────────────────────────────
patient_inputs = {
    "Age": age, "Weight": weight, "Length": height, "BMI": bmi,
    "BP": bp, "PR": pr, "FBS": fbs, "CR": cr, "TG": tg,
    "LDL": ldl, "HDL": hdl, "BUN": bun, "ESR": esr, "HB": hb,
    "K": k, "Na": na, "WBC": wbc, "Lymph": lymph, "Neut": neut,
    "PLT": plt_val, "EF-TTE": ef_tte,
    "Sex": 1 if sex == "Male" else 0,
    "DM": int(dm), "HTN": int(htn), "Current Smoker": int(smoker),
    "EX-Smoker": int(ex_smoker), "FH": int(fh), "Obesity": int(obesity),
    "CRF": int(crf), "CVA": int(cva), "Airway disease": int(airway),
    "Thyroid Disease": int(thyroid), "CHF": int(chf), "DLP": int(dlp),
    "Edema": int(edema), "Weak Peripheral Pulse": int(weak_pp),
    "Lung rales": int(lung_rales), "Systolic Murmur": int(sys_mur),
    "Diastolic Murmur": int(dia_mur), "Typical Chest Pain": int(typical_cp),
    "Dyspnea": int(dyspnea), "Atypical": int(atypical),
    "Nonanginal": int(nonanginal), "Exertional CP": int(exertional),
    "LowTH Ang": int(lowth), "Q Wave": int(q_wave),
    "St Elevation": int(st_elev), "St Depression": int(st_dep),
    "Tinversion": int(t_inv), "LVH": int(lvh),
    "Poor R Progression": int(poor_r),
    "Function Class": fc, "Region RWMA": region_rwma, "VHD": vhd_val,
}

# ─────────────────────────────────────────
# MAIN PANEL
# ─────────────────────────────────────────
# Top header
st.markdown("""
<div style='display:flex; align-items:center; gap:1rem; padding: 0.5rem 0 1.5rem;'>
    <span style='font-size:2rem;'>🫀</span>
    <div>
        <div style='font-size:1.6rem; font-weight:700; letter-spacing:-0.5px; color:#f1f5f9;'>NovaHeart <span style='color:#3b82f6;'>CDSS</span></div>
        <div style='font-size:0.8rem; color:#64748b; letter-spacing:1.5px; text-transform:uppercase;'>Cardiovascular Risk Stratification · CatBoost ML Engine · Z-Alizadeh Sani Dataset</div>
    </div>
</div>
""", unsafe_allow_html=True)

if not run_btn:
    # ── LANDING STATE ──
    tab_home, tab_about = st.tabs(["📊 Dashboard", "ℹ️ About"])

    with tab_home:
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("ML Engine", "CatBoost", "Optimized via Optuna")
        c2.metric("Features", "54", "Clinical + ECG + Lab")
        c3.metric("Imputer", "KNN", "Defensive layer active")
        c4.metric("Dataset", "CAD / UCI", "Z-Alizadeh Sani")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">How To Use</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='nova-card'>
            <ol style='color:#94a3b8; line-height:2; margin:0; padding-left:1.2rem;'>
                <li>Fill in the patient's clinical profile in the <b style='color:#f1f5f9;'>left sidebar</b></li>
                <li>Enter vitals, lab values, symptoms, and ECG findings</li>
                <li>Press <b style='color:#3b82f6;'>Run Cardiac Analysis</b></li>
                <li>Review the risk stratification, alerts, and care plan</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Risk Classification Matrix</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, label, pct, color, bg in [
            (rc1, "LOW", "< 40%", "#10b981", "rgba(16,185,129,0.1)"),
            (rc2, "MODERATE", "40–65%", "#f97316", "rgba(249,115,22,0.1)"),
            (rc3, "HIGH", "65–85%", "#f59e0b", "rgba(245,158,11,0.1)"),
            (rc4, "CRITICAL", "≥ 85%", "#ef4444", "rgba(239,68,68,0.1)"),
        ]:
            col.markdown(f"""
            <div style='background:{bg}; border:1px solid {color}40; border-radius:10px; padding:1rem; text-align:center;'>
                <div style='font-size:0.65rem; letter-spacing:2px; text-transform:uppercase; color:{color}; font-weight:700;'>{label}</div>
                <div style='font-size:1.4rem; font-weight:700; color:#f1f5f9; margin-top:0.3rem;'>{pct}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_about:
        st.markdown("""
        <div class='nova-card'>
            <p style='color:#94a3b8; line-height:1.8;'>
            <b style='color:#f1f5f9;'>NovaHeart CDSS</b> is an advanced cardiovascular risk stratification system powered by
            a CatBoost Gradient Boosting classifier, optimized via Optuna to maximize clinical Recall — minimizing
            dangerous false negatives. The model was trained on the <b style='color:#3b82f6;'>Z-Alizadeh Sani Dataset</b>,
            which includes detailed echocardiographic, lab, ECG, and clinical markers including Ejection Fraction and NYHA Classification.
            </p>
            <p style='color:#94a3b8; line-height:1.8; margin-top:0.5rem;'>
            A <b style='color:#f1f5f9;'>KNN Imputer</b> provides a defensive preprocessing layer so incomplete patient profiles
            are safely handled without crashing the pipeline. Feature schema is locked to prevent training-serving skew.
            </p>
            <p style='color:#ef4444; font-size:0.8rem; margin-top:1rem;'>
            ⚠️ For clinical decision support only. Not a substitute for physician judgment.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # ─────────────────────────────────────────
    # PREDICTION FLOW
    # ─────────────────────────────────────────
    with st.spinner("Running inference pipeline…"):
        patient_df   = build_patient_df(patient_inputs)
        risk_prob    = model.predict_proba(patient_df)[0][1]
        care         = compute_care_plan(patient_inputs, risk_prob)
        risk_cls     = care["risk_class"]
        r_color      = risk_color(risk_cls)

    # Banner
    banner_cls = {"CRITICAL": "risk-critical", "HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}[risk_cls]
    st.markdown(f"""
    <div class='risk-banner {banner_cls}'>
        <div style='font-size:0.7rem; letter-spacing:3px; text-transform:uppercase; color:{r_color}; font-weight:700; margin-bottom:0.4rem;'>RISK CLASSIFICATION</div>
        <div style='font-size:3rem; font-weight:800; color:{r_color}; letter-spacing:-1px;'>{risk_cls}</div>
        <div style='font-size:0.9rem; color:#94a3b8; margin-top:0.3rem;'>CAD Probability: <span style='color:{r_color}; font-weight:700; font-size:1.1rem; font-family:JetBrains Mono;'>{risk_prob*100:.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["📈 Risk Overview", "🩺 Clinical Report", "📊 Lab Panel", "🔍 Feature Analysis"])

    # ── TAB 1: RISK OVERVIEW ──
    with tabs[0]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="section-header">Risk Probability Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(risk_prob), use_container_width=True, config={"displayModeBar": False})

            # Key metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("EF-TTE", f"{ef_tte}%", "Normal ≥55%" if ef_tte >= 55 else "⚠ Reduced")
            m2.metric("NYHA Class", f"Class {fc}")
            m3.metric("BP", f"{bp} mmHg", "Normal" if bp < 130 else "⚠ Elevated")

        with c2:
            st.markdown('<div class="section-header">Biomarker Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(patient_inputs), use_container_width=True, config={"displayModeBar": False})

            # Risk flags summary
            flag_cols = {
                "DM": (dm, "Diabetes"), "HTN": (htn, "Hypertension"),
                "Smoker": (smoker, "Current Smoker"), "Obesity": (obesity, "Obesity"),
                "CHF": (chf, "CHF"), "CRF": (crf, "Renal Failure"),
            }
            active_flags = [label for _, (val, label) in flag_cols.items() if val]
            if active_flags:
                st.markdown('<div class="section-header" style="margin-top:1rem;">Active Risk Flags</div>', unsafe_allow_html=True)
                flag_html = " ".join([f'<span class="pill pill-red">{f}</span>' for f in active_flags])
                st.markdown(f"<div style='display:flex; flex-wrap:wrap; gap:6px;'>{flag_html}</div>", unsafe_allow_html=True)

    # ── TAB 2: CLINICAL REPORT ──
    with tabs[1]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="section-header">⚡ Clinical Alerts</div>', unsafe_allow_html=True)
            for alert in care["alerts"]:
                st.markdown(f'<div class="alert-box">⚠ {alert}</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">✅ Recommended Actions</div>', unsafe_allow_html=True)
            for action in care["actions"]:
                st.markdown(f'<div class="action-box">→ {action}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Clinical Summary</div>', unsafe_allow_html=True)

        sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
        ef_label = "Normal" if ef_tte >= 55 else ("Mildly Reduced" if ef_tte >= 50 else ("Reduced" if ef_tte >= 40 else "Severely Reduced"))
        ef_color = "pill-green" if ef_tte >= 55 else ("pill-amber" if ef_tte >= 40 else "pill-red")
        sum_c1.markdown(f"**Ejection Fraction**<br><span class='pill {ef_color}'>{ef_tte}% — {ef_label}</span>", unsafe_allow_html=True)
        sum_c2.markdown(f"**NYHA Class**<br><span class='pill pill-blue'>Class {fc}</span>", unsafe_allow_html=True)
        sum_c3.markdown(f"**Cardiac Congestion**<br><span class='pill {'pill-red' if edema or lung_rales else 'pill-green'}'>{'Suspected' if (edema or lung_rales) else 'Clear'}</span>", unsafe_allow_html=True)
        sum_c4.markdown(f"**LDL Status**<br><span class='pill {'pill-red' if ldl > 160 else 'pill-amber' if ldl > 130 else 'pill-green'}'>{ldl} mg/dL</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 Export Raw Prediction Data"):
            export_data = {
                "patient_id": "NOVA-" + str(hash(str(patient_inputs)))[-6:].upper(),
                "risk_probability": round(float(risk_prob), 4),
                "risk_classification": risk_cls,
                "ef_tte": ef_tte,
                "nyha_class": fc,
                "alerts": care["alerts"],
                "recommended_actions": care["actions"],
                "inputs": patient_inputs,
            }
            st.json(export_data)

    # ── TAB 3: LAB PANEL ──
    with tabs[2]:
        st.markdown('<div class="section-header">Laboratory Reference Panel</div>', unsafe_allow_html=True)

        lab_data = [
            ("Fasting Blood Sugar", fbs, "mg/dL", 70, 100, 500),
            ("Creatinine", cr, "mg/dL", 0.6, 1.2, 15),
            ("Triglycerides", tg, "mg/dL", 0, 150, 800),
            ("LDL Cholesterol", ldl, "mg/dL", 0, 100, 300),
            ("HDL Cholesterol", hdl, "mg/dL", 40, 60, 120),
            ("BUN", bun, "mg/dL", 7, 25, 150),
            ("ESR", esr, "mm/hr", 0, 20, 150),
            ("Hemoglobin", hb, "g/dL", 12, 17, 20),
            ("Potassium", k, "mEq/L", 3.5, 5.0, 8),
            ("Sodium", na, "mEq/L", 135, 145, 170),
            ("WBC", wbc, "cells/µL", 4000, 11000, 30000),
            ("Platelets", plt_val, "×10³/µL", 150, 400, 800),
        ]

        cols = st.columns(3)
        for i, (name, val, unit, low, high, mx) in enumerate(lab_data):
            with cols[i % 3]:
                pct = min(val / mx * 100, 100)
                status = "Normal" if low <= val <= high else ("High" if val > high else "Low")
                sc = "#10b981" if status == "Normal" else "#ef4444"
                st.markdown(f"""
                <div class='nova-card' style='padding:1rem; margin-bottom:0.7rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>
                        <span style='font-size:0.8rem; color:#94a3b8;'>{name}</span>
                        <span style='font-size:0.65rem; color:{sc}; background:{sc}20; padding:2px 8px; border-radius:999px; border:1px solid {sc}40;'>{status}</span>
                    </div>
                    <div style='font-size:1.3rem; font-weight:700; color:#f1f5f9; font-family:JetBrains Mono;'>{val} <span style='font-size:0.7rem; color:#64748b; font-family:Inter;'>{unit}</span></div>
                    <div style='background:#1e2d45; border-radius:4px; height:4px; margin-top:0.6rem;'>
                        <div style='background:{sc}; width:{pct}%; height:4px; border-radius:4px;'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:0.65rem; color:#475569; margin-top:3px;'>
                        <span>Ref: {low}–{high}</span><span>{unit}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4: FEATURE ANALYSIS ──
    with tabs[3]:
        st.markdown('<div class="section-header">Numeric Feature Value Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(feature_bar_chart(patient_inputs), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-header" style="margin-top:0.5rem;">Active Categorical Flags</div>', unsafe_allow_html=True)
        cat_flags = {k: v for k, v in patient_inputs.items() if k in CATEGORICAL_COLS}
        on_flags  = [k for k, v in cat_flags.items() if v and v != 0]
        off_flags = [k for k, v in cat_flags.items() if not v or v == 0]

        fc1, fc2 = st.columns(2)
        with fc1:
            st.markdown("**Active (1)**")
            if on_flags:
                html = " ".join([f'<span class="pill pill-red" style="margin:3px;">{f}</span>' for f in on_flags])
                st.markdown(f"<div style='display:flex; flex-wrap:wrap; gap:4px;'>{html}</div>", unsafe_allow_html=True)
            else:
                st.caption("None active")
        with fc2:
            st.markdown("**Inactive (0)**")
            if off_flags:
                html = " ".join([f'<span class="pill pill-green" style="margin:3px;">{f}</span>' for f in off_flags])
                st.markdown(f"<div style='display:flex; flex-wrap:wrap; gap:4px;'>{html}</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center; padding:2rem 0 1rem; color:#334155; font-size:0.72rem; letter-spacing:1px;'>
NOVAHEART CDSS · FOR CLINICAL DECISION SUPPORT ONLY · NOT A SUBSTITUTE FOR MEDICAL JUDGMENT
</div>
""", unsafe_allow_html=True)
