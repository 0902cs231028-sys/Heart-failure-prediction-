import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from catboost import CatBoostClassifier
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import shap

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
# CUSTOM CSS — Clean Blue & White Clinical Aesthetic
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #f0f4f8;
    --bg-card: #ffffff;
    --bg-elevated: #ffffff;
    --accent-blue: #2563eb;
    --accent-blue-light: #3b82f6;
    --accent-blue-dark: #1e40af;
    --accent-green: #10b981;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #1e293b;
    --text-secondary: #475569;
    --text-muted: #64748b;
    --border: #e2e8f0;
    --border-light: #f1f5f9;
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    --shap-positive: #ef4444;
    --shap-negative: #3b82f6;
}

html, body, .stApp {
    background-color: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }

/* Input widgets */
.stTextInput input, .stNumberInput input, .stSelectbox select,
div[data-baseweb="input"] input, div[data-baseweb="select"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 0.5rem 0.75rem !important;
}
div[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: var(--accent-blue) !important;
}
div[data-testid="stSlider"] label {
    color: var(--text-secondary) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #1e40af) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'Inter', sans-serif !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.35) !important;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    box-shadow: var(--shadow-sm) !important;
}
div[data-testid="stMetric"] label { 
    color: var(--text-muted) !important; 
    font-size: 0.75rem !important; 
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { 
    color: var(--text-primary) !important; 
    font-size: 1.8rem !important; 
    font-weight: 700 !important; 
}

/* Tabs */
button[data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-bottom: 2px solid transparent !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.2s ease !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent-blue) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
    background: transparent !important;
}
button[data-baseweb="tab"]:hover {
    color: var(--accent-blue-light) !important;
}

/* Expander / Collapsible */
details {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    margin-bottom: 1rem !important;
    transition: all 0.2s ease !important;
}
details:hover {
    border-color: var(--accent-blue-light) !important;
    box-shadow: var(--shadow-sm) !important;
}
summary {
    font-weight: 600 !important;
    color: var(--text-primary) !important;
    padding: 0.75rem 1rem !important;
}

/* Divider */
hr { 
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
}

/* Plotly chart backgrounds */
.js-plotly-plot .plotly { background: transparent !important; }

/* Custom card component */
.nova-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: var(--shadow-sm);
    transition: all 0.2s ease;
}
.nova-card:hover {
    box-shadow: var(--shadow-md);
}

.risk-banner {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
    background: var(--bg-card);
    box-shadow: var(--shadow-lg);
}
.risk-critical { border-top: 6px solid #ef4444; background: linear-gradient(135deg, #ffffff, #fef2f2); }
.risk-high { border-top: 6px solid #f59e0b; background: linear-gradient(135deg, #ffffff, #fffbeb); }
.risk-moderate { border-top: 6px solid #f97316; background: linear-gradient(135deg, #ffffff, #fff7ed); }
.risk-low { border-top: 6px solid #10b981; background: linear-gradient(135deg, #ffffff, #ecfdf5); }

.pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.pill-red { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
.pill-green { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
.pill-blue { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
.pill-amber { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }

.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--accent-blue);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--accent-blue-light);
    display: inline-block;
}

.mono { font-family: 'JetBrains Mono', monospace; }

.alert-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-left: 4px solid #ef4444;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #991b1b;
}

.action-box {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-left: 4px solid #10b981;
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
    font-size: 0.85rem;
    color: #065f46;
}

.shap-card {
    background: var(--bg-card);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
}

.shap-positive-bar {
    background: linear-gradient(90deg, #fef2f2, #fee2e2);
    border-left: 4px solid #ef4444;
}

.shap-negative-bar {
    background: linear-gradient(90deg, #eff6ff, #dbeafe);
    border-left: 4px solid #3b82f6;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD ARTIFACTS
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="🫀 Loading NovaHeart Engine...")
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
# SHAP EXPLAINER SETUP
# ─────────────────────────────────────────
@st.cache_resource(show_spinner="📊 Initializing SHAP explainer...")
def init_shap_explainer():
    # Create a background dataset for SHAP (use dummy data with median values)
    background_df = pd.DataFrame([{col: 0 for col in FEATURE_ORDER}])
    explainer = shap.TreeExplainer(model, background_df, feature_perturbation="interventional")
    return explainer

try:
    shap_explainer = init_shap_explainer()
    SHAP_AVAILABLE = True
except Exception as e:
    st.warning(f"SHAP explainer initialization note: {str(e)[:100]}...")
    SHAP_AVAILABLE = False

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
    return {"CRITICAL": "#dc2626", "HIGH": "#d97706", "MODERATE": "#ea580c", "LOW": "#059669"}.get(cls, "#64748b")


def gauge_chart(prob):
    pct = prob * 100
    color = "#dc2626" if pct >= 65 else "#d97706" if pct >= 40 else "#059669"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 44, "color": "#1e293b", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"color": "#64748b"}, "tickwidth": 1},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#f8fafc",
            "borderwidth": 1,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.1)"},
                {"range": [40, 65], "color": "rgba(245,158,11,0.08)"},
                {"range": [65, 100], "color": "rgba(239,68,68,0.08)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.75, "value": pct},
        },
    ))
    fig.update_layout(
        margin=dict(t=20, b=10, l=20, r=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#1e293b"},
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
        fillcolor='rgba(37,99,235,0.15)',
        line=dict(color='#2563eb', width=2),
        marker=dict(color='#2563eb', size=6),
    ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(visible=True, range=[0, 100], gridcolor='#e2e8f0', tickfont=dict(color='#64748b', size=9)),
            angularaxis=dict(gridcolor='#e2e8f0', tickfont=dict(color='#475569', size=10)),
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=50, r=50),
        height=280,
        font=dict(family='Inter', color='#1e293b'),
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
            colorscale=[[0, '#2563eb'], [0.5, '#3b82f6'], [1, '#60a5fa']],
            showscale=False,
        ),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=480,
        margin=dict(t=10, b=10, l=120, r=20),
        xaxis=dict(gridcolor='#e2e8f0', color='#64748b'),
        yaxis=dict(color='#475569', tickfont=dict(size=11)),
        font=dict(family='Inter', color='#1e293b'),
    )
    return fig


def create_shap_waterfall(shap_values, base_value, features_df, top_n=10):
    """Create a waterfall chart visualization for SHAP values"""
    
    # Get feature names and SHAP values
    feature_names = features_df.columns.tolist()
    shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
    
    # Create dataframe of feature contributions
    contributions = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_vals,
        'abs_shap': np.abs(shap_vals)
    })
    
    # Sort by absolute contribution and get top N
    contributions = contributions.sort_values('abs_shap', ascending=False).head(top_n)
    
    # Calculate cumulative contributions
    base_prob = base_value
    cumulative = base_prob
    contributions['cumulative_start'] = cumulative
    
    waterfall_data = []
    for idx, row in contributions.iterrows():
        cumulative += row['shap_value']
        waterfall_data.append({
            'feature': row['feature'],
            'shap_value': row['shap_value'],
            'start': contributions.loc[idx, 'cumulative_start'],
            'end': cumulative,
            'is_positive': row['shap_value'] > 0
        })
    
    final_prob = cumulative
    
    # Create Plotly waterfall chart
    fig = go.Figure()
    
    # Add base value bar
    fig.add_trace(go.Bar(
        name='Base Value',
        x=['Base Value'],
        y=[base_prob],
        marker_color='#94a3b8',
        text=[f'{base_prob:.3f}'],
        textposition='outside',
        width=0.6
    ))
    
    # Add contribution bars
    for item in waterfall_data:
        color = '#ef4444' if item['is_positive'] else '#3b82f6'
        fig.add_trace(go.Bar(
            name=item['feature'],
            x=[item['feature']],
            y=[item['shap_value']],
            marker_color=color,
            text=[f"+{item['shap_value']:.3f}" if item['is_positive'] else f"{item['shap_value']:.3f}"],
            textposition='outside',
            width=0.6,
            base=[item['start']]
        ))
    
    # Add final prediction bar
    fig.add_trace(go.Bar(
        name='Final Prediction',
        x=['Final Prediction'],
        y=[final_prob],
        marker_color='#2563eb',
        text=[f'{final_prob:.3f}'],
        textposition='outside',
        width=0.6
    ))
    
    fig.update_layout(
        title=dict(
            text="Clinical Evidence Breakdown — SHAP Feature Contributions",
            font=dict(size=14, color="#1e293b", family="Inter"),
            x=0.5
        ),
        barmode='overlay',
        showlegend=False,
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title="Features",
            tickangle=-45,
            tickfont=dict(size=10, color="#475569"),
            gridcolor='#e2e8f0'
        ),
        yaxis=dict(
            title="f(x) = Prediction Score",
            titlefont=dict(color="#475569"),
            tickfont=dict(color="#64748b"),
            gridcolor='#e2e8f0'
        ),
        font=dict(family='Inter'),
        margin=dict(t=60, b=80, l=40, r=40)
    )
    
    return fig, contributions


def create_shap_summary_table(contributions):
    """Create a styled summary table of SHAP contributions"""
    
    contributions['direction'] = contributions['shap_value'].apply(
        lambda x: '↑ Increases Risk' if x > 0 else '↓ Decreases Risk'
    )
    contributions['impact'] = contributions['shap_value'].apply(
        lambda x: f"+{x:.4f}" if x > 0 else f"{x:.4f}"
    )
    
    # Style with colors
    def color_impact(val):
        if '+' in str(val):
            return 'color: #dc2626; font-weight: 600'
        elif '-' in str(val):
            return 'color: #2563eb; font-weight: 600'
        return ''
    
    styled = contributions[['feature', 'impact', 'direction', 'abs_shap']].head(10)
    styled.columns = ['Feature', 'Impact', 'Direction', '|Impact|']
    styled['Impact'] = styled['Impact'].apply(lambda x: f"{float(x):+.4f}" if isinstance(x, (int, float)) else x)
    styled['|Impact|'] = styled['|Impact|'].apply(lambda x: f"{x:.4f}")
    
    return styled


# ─────────────────────────────────────────
# SIDEBAR — PATIENT DATA ENTRY (Collapsible sections)
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem;'>
        <div style='font-size:2.5rem;'>🫀</div>
        <div style='font-size:1.2rem; font-weight:700; letter-spacing:-0.5px; color:#1e293b;'>NOVAHEART</div>
        <div style='font-size:0.7rem; color:#64748b; letter-spacing:1.5px; text-transform:uppercase;'>Clinical Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("👤 Demographics", expanded=True):
        age = st.slider("Age (years)", 20, 90, 55)
        sex = st.selectbox("Sex", ["Male", "Female"])
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 75.0, step=0.5)
        height = st.number_input("Height / Length (cm)", 100.0, 220.0, 170.0, step=0.5)
        bmi = round(weight / (height / 100) ** 2, 2)
        st.caption(f"📊 BMI: **{bmi}** kg/m²")

    with st.expander("💓 Vital Signs", expanded=True):
        bp = st.slider("Blood Pressure — Systolic (mmHg)", 70, 220, 120)
        pr = st.slider("Pulse Rate (bpm)", 40, 180, 75)

    with st.expander("🫀 Cardiac Function", expanded=True):
        ef_tte = st.slider("Ejection Fraction — EF-TTE (%)", 10, 80, 55)
        fc = st.selectbox("NYHA Function Class", [0, 1, 2, 3, 4], index=1)
        region_rwma = st.selectbox("Region RWMA", [0, 1, 2, 3, 4])

    with st.expander("🧪 Lab Values", expanded=False):
        fbs = st.number_input("Fasting Blood Sugar (mg/dL)", 50.0, 500.0, 100.0)
        cr = st.number_input("Creatinine (mg/dL)", 0.3, 15.0, 1.0, step=0.1)
        tg = st.number_input("Triglycerides (mg/dL)", 30.0, 800.0, 150.0)
        ldl = st.number_input("LDL Cholesterol (mg/dL)", 20.0, 300.0, 120.0)
        hdl = st.number_input("HDL Cholesterol (mg/dL)", 10.0, 120.0, 45.0)
        bun = st.number_input("BUN (mg/dL)", 3.0, 150.0, 15.0)
        esr = st.number_input("ESR (mm/hr)", 1.0, 150.0, 15.0)
        hb = st.number_input("Hemoglobin (g/dL)", 5.0, 20.0, 14.0, step=0.1)
        k = st.number_input("Potassium K (mEq/L)", 2.0, 8.0, 4.2, step=0.1)
        na = st.number_input("Sodium Na (mEq/L)", 110.0, 170.0, 140.0)
        wbc = st.number_input("WBC (cells/µL)", 1000.0, 30000.0, 7500.0, step=100.0)
        lymph = st.number_input("Lymphocytes (%)", 5.0, 80.0, 30.0)
        neut = st.number_input("Neutrophils (%)", 10.0, 95.0, 60.0)
        plt_val = st.number_input("Platelets (×10³/µL)", 50.0, 800.0, 250.0)

    with st.expander("📋 Comorbidities", expanded=False):
        col1s, col2s = st.columns(2)
        with col1s:
            dm = st.checkbox("Diabetes (DM)")
            htn = st.checkbox("Hypertension")
            obesity = st.checkbox("Obesity")
            crf = st.checkbox("Chronic Renal Failure")
            cva = st.checkbox("CVA / Stroke")
            chf = st.checkbox("CHF")
        with col2s:
            fh = st.checkbox("Family History")
            smoker = st.checkbox("Current Smoker")
            ex_smoker = st.checkbox("Ex-Smoker")
            airway = st.checkbox("Airway Disease")
            thyroid = st.checkbox("Thyroid Disease")
            dlp = st.checkbox("Dyslipidemia")

    with st.expander("📊 Symptoms & ECG", expanded=False):
        col3s, col4s = st.columns(2)
        with col3s:
            typical_cp = st.checkbox("Typical Chest Pain")
            dyspnea = st.checkbox("Dyspnea")
            atypical = st.checkbox("Atypical Chest Pain")
            nonanginal = st.checkbox("Nonanginal Pain")
            exertional = st.checkbox("Exertional CP")
            lowth = st.checkbox("Low Threshold Angina")
            edema = st.checkbox("Edema")
            weak_pp = st.checkbox("Weak Peripheral Pulse")
            lung_rales = st.checkbox("Lung Rales")
        with col4s:
            sys_mur = st.checkbox("Systolic Murmur")
            dia_mur = st.checkbox("Diastolic Murmur")
            q_wave = st.checkbox("Q Wave")
            st_elev = st.checkbox("ST Elevation")
            st_dep = st.checkbox("ST Depression")
            t_inv = st.checkbox("T-wave Inversion")
            lvh = st.checkbox("LVH")
            poor_r = st.checkbox("Poor R Progression")

    with st.expander("❤️ Valvular Disease", expanded=False):
        vhd_label = st.selectbox("Valvular Heart Disease", ["None", "Mild", "Moderate", "Severe"])
        vhd_map = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
        vhd_val = vhd_map[vhd_label]

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🔬 Run Cardiac Analysis", use_container_width=True)


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
st.markdown("""
<div style='display:flex; align-items:center; gap:1rem; padding: 0.5rem 0 1.5rem;'>
    <span style='font-size:2.5rem;'>🫀</span>
    <div>
        <div style='font-size:1.8rem; font-weight:700; letter-spacing:-0.5px; color:#1e293b;'>NovaHeart <span style='color:#2563eb;'>CDSS</span></div>
        <div style='font-size:0.75rem; color:#64748b; letter-spacing:1px; text-transform:uppercase;'>Cardiovascular Risk Stratification · CatBoost ML Engine · Z-Alizadeh Sani Dataset</div>
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
        c4.metric("SHAP", "✓ Active", "Explainable AI")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">How To Use</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='nova-card'>
            <ol style='color:#475569; line-height:2; margin:0; padding-left:1.2rem;'>
                <li>Fill in the patient's clinical profile in the <b style='color:#2563eb;'>left sidebar</b> — all sections are collapsible for easy navigation</li>
                <li>Enter vitals, lab values, symptoms, and ECG findings</li>
                <li>Press <b style='color:#2563eb;'>Run Cardiac Analysis</b> to generate risk assessment</li>
                <li>Review the risk stratification, SHAP explanations, alerts, and care plan across detailed tabs</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Risk Classification Matrix</div>', unsafe_allow_html=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        for col, label, pct, color, bg in [
            (rc1, "LOW", "< 40%", "#059669", "#ecfdf5"),
            (rc2, "MODERATE", "40–65%", "#ea580c", "#fff7ed"),
            (rc3, "HIGH", "65–85%", "#d97706", "#fffbeb"),
            (rc4, "CRITICAL", "≥ 85%", "#dc2626", "#fef2f2"),
        ]:
            col.markdown(f"""
            <div style='background:{bg}; border:1px solid {color}30; border-radius:16px; padding:1rem; text-align:center;'>
                <div style='font-size:0.65rem; letter-spacing:1.5px; text-transform:uppercase; color:{color}; font-weight:700;'>{label}</div>
                <div style='font-size:1.4rem; font-weight:700; color:#1e293b; margin-top:0.3rem;'>{pct}</div>
            </div>
            """, unsafe_allow_html=True)

    with tab_about:
        st.markdown("""
        <div class='nova-card'>
            <p style='color:#475569; line-height:1.8;'>
            <b style='color:#1e293b;'>NovaHeart CDSS</b> is an advanced cardiovascular risk stratification system powered by
            a CatBoost Gradient Boosting classifier, optimized via Optuna to maximize clinical Recall — minimizing
            dangerous false negatives. The model was trained on the <b style='color:#2563eb;'>Z-Alizadeh Sani Dataset</b>,
            which includes detailed echocardiographic, lab, ECG, and clinical markers including Ejection Fraction and NYHA Classification.
            </p>
            <p style='color:#475569; line-height:1.8; margin-top:0.5rem;'>
            A <b style='color:#1e293b;'>KNN Imputer</b> provides a defensive preprocessing layer so incomplete patient profiles
            are safely handled without crashing the pipeline. Feature schema is locked to prevent training-serving skew.
            </p>
            <p style='color:#475569; line-height:1.8; margin-top:0.5rem;'>
            <b style='color:#2563eb;'>🔬 SHAP (SHapley Additive exPlanations)</b> provides interpretable AI by showing exactly 
            which clinical features contributed to the prediction and by how much — enabling truly transparent clinical decision support.
            </p>
            <hr>
            <p style='color:#ef4444; font-size:0.8rem; margin-top:1rem;'>
            ⚠️ For clinical decision support only. Not a substitute for physician judgment.
            </p>
        </div>
        """, unsafe_allow_html=True)

else:
    # ─────────────────────────────────────────
    # PREDICTION FLOW
    # ─────────────────────────────────────────
    with st.spinner("🫀 Running inference pipeline..."):
        patient_df = build_patient_df(patient_inputs)
        risk_prob = model.predict_proba(patient_df)[0][1]
        care = compute_care_plan(patient_inputs, risk_prob)
        risk_cls = care["risk_class"]
        r_color = risk_color(risk_cls)
        
        # Compute SHAP values
        shap_contributions = None
        shap_fig = None
        shap_table = None
        if SHAP_AVAILABLE:
            try:
                shap_values = shap_explainer.shap_values(patient_df)
                expected_value = shap_explainer.expected_value
                shap_fig, shap_contributions = create_shap_waterfall(shap_values, expected_value, patient_df, top_n=8)
                shap_table = create_shap_summary_table(shap_contributions)
            except Exception as e:
                st.warning(f"SHAP computation note: {str(e)[:100]}")
                SHAP_AVAILABLE = False

    # Banner
    banner_cls = {"CRITICAL": "risk-critical", "HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}[risk_cls]
    st.markdown(f"""
    <div class='risk-banner {banner_cls}'>
        <div style='font-size:0.7rem; letter-spacing:2px; text-transform:uppercase; color:{r_color}; font-weight:700; margin-bottom:0.4rem;'>Risk Classification</div>
        <div style='font-size:3rem; font-weight:800; color:{r_color}; letter-spacing:-1px;'>{risk_cls}</div>
        <div style='font-size:0.9rem; color:#475569; margin-top:0.3rem;'>CAD Probability: <span style='color:{r_color}; font-weight:700; font-size:1.1rem; font-family:JetBrains Mono;'>{risk_prob*100:.2f}%</span></div>
    </div>
    """, unsafe_allow_html=True)

    # Updated tabs with SHAP as first tab
    tabs = st.tabs(["📊 SHAP Explanations", "📈 Risk Overview", "🩺 Clinical Report", "🧪 Lab Panel", "🔍 Feature Analysis"])

    # ── TAB 0: SHAP EXPLANATIONS (NEW) ──
    with tabs[0]:
        if SHAP_AVAILABLE and shap_fig is not None:
            st.markdown("""
            <div class='shap-card'>
                <div style='display:flex; align-items:center; gap:0.5rem; margin-bottom:1rem;'>
                    <span style='font-size:1.5rem;'>🔬</span>
                    <div>
                        <div style='font-size:0.7rem; font-weight:700; letter-spacing:1px; color:#2563eb; text-transform:uppercase;'>Explainable AI</div>
                        <div style='font-size:0.85rem; color:#64748b;'>Understanding what drives each prediction</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.plotly_chart(shap_fig, use_container_width=True, config={"displayModeBar": False})
            
            # Add interpretation guide
            with st.expander("📖 How to interpret this chart"):
                st.markdown("""
                <div style='padding: 0.5rem 0;'>
                    <ul style='color:#475569; line-height:1.8;'>
                        <li><span style='color:#ef4444; font-weight:600;'>Red bars</span> increase the risk prediction (↑ CAD Probability)</li>
                        <li><span style='color:#3b82f6; font-weight:600;'>Blue bars</span> decrease the risk prediction (↓ CAD Probability)</li>
                        <li>The <b>base value</b> is the average prediction across all patients</li>
                        <li>The <b>final prediction</b> is the patient-specific risk score after all contributions</li>
                        <li>Longer bars = larger impact on the final decision</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Feature impact summary table
            if shap_table is not None:
                st.markdown('<div class="section-header" style="margin-top:1rem;">🔑 Top Feature Impacts</div>', unsafe_allow_html=True)
                
                # Create styled HTML table
                table_html = """
                <table style='width:100%; border-collapse:collapse; background:#ffffff; border-radius:12px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,0.05);'>
                    <thead>
                        <tr style='background:#f8fafc; border-bottom:2px solid #e2e8f0;'>
                            <th style='padding:12px 16px; text-align:left; font-size:0.75rem; font-weight:700; color:#1e293b; text-transform:uppercase; letter-spacing:0.5px;'>Feature</th>
                            <th style='padding:12px 16px; text-align:center; font-size:0.75rem; font-weight:700; color:#1e293b; text-transform:uppercase; letter-spacing:0.5px;'>Impact</th>
                            <th style='padding:12px 16px; text-align:left; font-size:0.75rem; font-weight:700; color:#1e293b; text-transform:uppercase; letter-spacing:0.5px;'>Direction</th>
                            <th style='padding:12px 16px; text-align:center; font-size:0.75rem; font-weight:700; color:#1e293b; text-transform:uppercase; letter-spacing:0.5px;'>|Impact|</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in shap_table.iterrows():
                    direction_color = "#dc2626" if "Increases" in row['Direction'] else "#2563eb"
                    impact_color = "#dc2626" if "+" in str(row['Impact']) else "#2563eb"
                    impact_sign = "▲" if "+" in str(row['Impact']) else "▼"
                    
                    table_html += f"""
                        <tr style='border-bottom:1px solid #f1f5f9;'>
                            <td style='padding:10px 16px; font-size:0.85rem; font-weight:500; color:#1e293b;'>{row['Feature']}</td>
                            <td style='padding:10px 16px; text-align:center; font-size:0.85rem; font-weight:600; color:{impact_color};'>{impact_sign} {row['Impact']}</td>
                            <td style='padding:10px 16px; font-size:0.8rem; color:{direction_color};'>{row['Direction']}</td>
                            <td style='padding:10px 16px; text-align:center; font-size:0.8rem; font-family:monospace; color:#64748b;'>{row['|Impact|']}</td>
                        </tr>
                    """
                
                table_html += """
                    </tbody>
                </table>
                """
                
                st.markdown(table_html, unsafe_allow_html=True)
                
                st.info("💡 **Clinical Insight**: Features with larger |Impact| values have the strongest influence on the prediction. Red/increasing features may represent modifiable risk factors.")
        else:
            st.warning("SHAP explanations are currently unavailable. Ensure the model is properly configured with SHAP support.")
            st.info("To enable SHAP: Make sure 'shap' is installed and the model supports TreeExplainer.")

    # ── TAB 1: RISK OVERVIEW ──
    with tabs[1]:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown('<div class="section-header">Risk Probability Gauge</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(risk_prob), use_container_width=True, config={"displayModeBar": False})

            m1, m2, m3 = st.columns(3)
            m1.metric("EF-TTE", f"{ef_tte}%", "Normal ≥55%" if ef_tte >= 55 else "⚠ Reduced")
            m2.metric("NYHA Class", f"Class {fc}")
            m3.metric("BP", f"{bp} mmHg", "Normal" if bp < 130 else "⚠ Elevated")

        with c2:
            st.markdown('<div class="section-header">Biomarker Radar</div>', unsafe_allow_html=True)
            st.plotly_chart(radar_chart(patient_inputs), use_container_width=True, config={"displayModeBar": False})

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
    with tabs[2]:
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
    with tabs[3]:
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
                sc = "#059669" if status == "Normal" else "#dc2626"
                st.markdown(f"""
                <div class='nova-card' style='padding:1rem; margin-bottom:0.7rem;'>
                    <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5rem;'>
                        <span style='font-size:0.75rem; color:#64748b;'>{name}</span>
                        <span style='font-size:0.65rem; color:{sc}; background:{sc}10; padding:2px 8px; border-radius:999px; border:1px solid {sc}30;'>{status}</span>
                    </div>
                    <div style='font-size:1.3rem; font-weight:700; color:#1e293b; font-family:JetBrains Mono;'>{val} <span style='font-size:0.7rem; color:#64748b; font-family:Inter;'>{unit}</span></div>
                    <div style='background:#e2e8f0; border-radius:4px; height:4px; margin-top:0.6rem;'>
                        <div style='background:{sc}; width:{pct}%; height:4px; border-radius:4px;'></div>
                    </div>
                    <div style='display:flex; justify-content:space-between; font-size:0.65rem; color:#94a3b8; margin-top:3px;'>
                        <span>Ref: {low}–{high}</span><span>{unit}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB 4: FEATURE ANALYSIS ──
    with tabs[4]:
        st.markdown('<div class="section-header">Numeric Feature Value Profile</div>', unsafe_allow_html=True)
        st.plotly_chart(feature_bar_chart(patient_inputs), use_container_width=True, config={"displayModeBar": False})

        st.markdown('<div class="section-header" style="margin-top:0.5rem;">Active Categorical Flags</div>', unsafe_allow_html=True)
        cat_flags = {k: v for k, v in patient_inputs.items() if k in CATEGORICAL_COLS}
        on_flags = [k for k, v in cat_flags.items() if v and v != 0]
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
<div style='text-align:center; padding:2rem 0 1rem; color:#94a3b8; font-size:0.7rem; letter-spacing:0.5px; border-top:1px solid #e2e8f0; margin-top:2rem;'>
NOVAHEART CDSS · FOR CLINICAL DECISION SUPPORT ONLY · NOT A SUBSTITUTE FOR MEDICAL JUDGMENT
</div>
""", unsafe_allow_html=True)