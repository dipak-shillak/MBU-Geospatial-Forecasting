import streamlit as st
import pandas as pd
import numpy as np
import os
import pydeck as pdk

# ================= SAFE PROPHET IMPORT =================
try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Sentinel-Aadhaar | UIDAI",
    layout="wide",
    page_icon="🛰️"
)

st.title("🛰️ Sentinel-Aadhaar")
st.caption(
    "Predictive Governance • Mandatory Biometric Updates • Zero Exclusion\n\n"
    "📌 *Transforms biometric backlog into fiscal, logistical & policy insights*"
)

# ================= DATA LOADER =================
@st.cache_data
def load_data(uploaded_files=None):
    files = [
        "api_data_aadhar_biometric_0_500000.csv",
        "api_data_aadhar_biometric_500000_1000000.csv",
        "api_data_aadhar_biometric_1000000_1500000.csv",
        "api_data_aadhar_biometric_1500000_1861108.csv"
    ]

    if uploaded_files:
        return pd.concat([pd.read_csv(f) for f in uploaded_files]), "Manual Upload"

    if all(os.path.exists(f) for f in files):
        return pd.concat([pd.read_csv(f) for f in files]), "Auto-Merged UIDAI Segments"

    return None, None

df_raw, mode = load_data()

# ================= HANDLE MISSING DATA =================
if df_raw is None:
    st.sidebar.warning("📂 UIDAI segments not found locally")
    uploads = st.sidebar.file_uploader(
        "Upload UIDAI CSVs",
        type="csv",
        accept_multiple_files=True
    )
    if uploads:
        df_raw, mode = load_data(uploads)
    else:
        st.info("⬅️ Upload UIDAI biometric datasets to begin analysis.")
        st.stop()

# ================= STANDARDIZATION =================
df = df_raw.copy()
df.columns = df.columns.str.strip().str.lower()

df["bio_age_5_17"] = pd.to_numeric(
    df.get("bio_age_5_17", 0),
    errors="coerce"
).fillna(0)

if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

# ================= SIDEBAR =================
st.sidebar.success(f"Mode: {mode}")

state = st.sidebar.selectbox(
    "Select State",
    sorted(df["state"].dropna().unique())
)

district = st.sidebar.selectbox(
    "Select District",
    sorted(df[df["state"] == state]["district"].dropna().unique())
)

dist_df = df[
    (df["state"] == state) &
    (df["district"] == district)
]

# ================= CORE METRICS =================
SATURATION_TARGET = 0.35
COI_PER_PERSON = 1000  # Cost of Inaction

total_pending = int(dist_df["bio_age_5_17"].sum())
saturation_gap = int(total_pending * SATURATION_TARGET)

state_total = df[df["state"] == state]["bio_age_5_17"].sum()
state_districts = max(df[df["state"] == state]["district"].nunique(), 1)

state_avg_gap = int((state_total * SATURATION_TARGET) / state_districts)
gap_delta = saturation_gap - state_avg_gap

coi_val = saturation_gap * COI_PER_PERSON

# ================= KPI DASHBOARD =================
c1, c2, c3 = st.columns(3)

c1.metric(
    "🎒 Children Pending MBU",
    f"{total_pending:,}"
)

c2.metric(
    "❗ Saturation Gap",
    f"{saturation_gap:,}",
    delta=f"{gap_delta:+,} vs State Avg",
    delta_color="inverse",
    help="Gap from 35% saturation benchmark vs state average"
)

c3.metric(
    "💸 Welfare Risk / Month",
    f"₹ {coi_val/1e7:.2f} Cr",
    help="Estimated DBT value at risk due to biometric exclusion"
)

# ================= ANALYSIS TABS =================
t1, t2, t3, t4, t5 = st.tabs([
    "📈 XAI Forecast",
    "💸 Fiscal Impact",
    "🚐 Logistics",
    "👥 Personas",
    "🗺️ Map View"
])

# ================= TAB 1: XAI FORECAST =================
with t1:
    st.subheader("Explainable AI Demand Forecast")

    st.caption(
        "📡 *Inclusion Difficulty Index measures deviation between predicted "
        "and actual enrollment — higher means harder-to-serve populations.*"
    )

    if PROPHET_OK and "date" in dist_df.columns and dist_df["date"].nunique() > 5:
        ts = (
            dist_df.groupby("date")["bio_age_5_17"]
            .sum()
            .reset_index()
            .rename(columns={"date": "ds", "bio_age_5_17": "y"})
        )

        model = Prophet(yearly_seasonality=True)
        model.fit(ts)

        future = model.make_future_dataframe(periods=6, freq="M")
        forecast = model.predict(future)

        st.line_chart(forecast.set_index("ds")["yhat"])

        inclusion_idx = int(
            abs(forecast["yhat"][:len(ts)] - ts["y"]).mean()
        )

        st.metric(
            "📡 Inclusion Difficulty Index",
            inclusion_idx
        )
    else:
        st.info("Insufficient historical data for explainable forecasting.")

# ================= TAB 2: FISCAL IMPACT =================
with t2:
    st.subheader("Potential DBT Leakage")

    st.caption(
        "💡 *Biometric updates protect welfare funds — not just identity.*"
    )

    st.write(f"🌾 **PM-Kisan Risk:** ₹ {(coi_val * 0.4) / 1e7:.2f} Cr")
    st.write(f"🎓 **Scholarships Risk:** ₹ {(coi_val * 0.3) / 1e7:.2f} Cr")
    st.write(f"🍚 **PDS Rations Risk:** ₹ {(coi_val * 0.3) / 1e7:.2f} Cr")

# ================= TAB 3: LOGISTICS =================
with t3:
    st.subheader("Mobile Van Optimization")

    st.caption("🚐 *Assumes 1 van processes ~150 biometric updates/day*")

    VAN_CAP = 150
    # Sprint Calculation
    vans = int(np.ceil(saturation_gap / (VAN_CAP * 7)))

    st.metric(
        "🚐 Vans Required (7-Day Sprint)",
        vans
    )

    if vans > 5:
        st.error("🚨 Hard-to-Reach Zone — Emergency Deployment Needed")
    else:
        st.success("✅ Routine Coverage Sufficient")

# ================= TAB 4: PERSONAS =================
with t4:
    st.subheader("Data-Driven Citizen Personas")

    st.info(
        "🎒 **Ages 5–7:** First-time school enrollment & Mid-Day Meal eligibility"
    )

    st.warning(
        "🎓 **Ages 15–17:** Mandatory Biometric Update (MBU) for scholarships & DBT"
    )

# ================= TAB 5: MAP VIEW =================
with t5:
    st.subheader("🗺️ Dynamic Saturation & Logistics Map")

    st.caption(
        "📍 *Dynamic map using state-centroid jittering. "
        "UIDAI raw files provide pincodes, not exact coordinates.*"
    )

    state_centers = {
        "Andaman & Nicobar Islands": [11.74, 92.74],
        "Bihar": [25.09, 85.31],
        "Delhi": [28.61, 77.20],
        "Andhra Pradesh": [15.91, 79.74],
        "Haryana": [29.05, 76.08]
    }

    # Dynamic Center Fallback
    if "latitude" not in df.columns or "longitude" not in df.columns:
        center = state_centers.get(state, [20.59, 78.96])
        df["latitude"] = center[0] + np.random.uniform(-0.1, 0.1, len(df))
        df["longitude"] = center[1] + np.random.uniform(-0.1, 0.1, len(df))

    map_df = (
        df[df["state"] == state]
        .groupby(["district", "latitude", "longitude"], as_index=False)
        ["bio_age_5_17"]
        .sum()
    )

    map_df["gap"] = (map_df["bio_age_5_17"] * SATURATION_TARGET).astype(int)

    layer = pdk.Layer(
        "ScatterplotLayer",
        map_df,
        get_position="[longitude, latitude]",
        get_radius="gap * 1.5",
        get_fill_color=[220, 38, 38, 160],
        pickable=True
    )

    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=map_df["latitude"].mean(),
                longitude=map_df["longitude"].mean(),
                zoom=7
            ),
            tooltip={
                "text": "District: {district}\nSaturation Gap: {gap}"
            }
        )
    )

# ================= EXECUTIVE SUMMARY =================
st.markdown("---")
st.subheader("📄 Executive Summary")

st.write(
    f"**Problem:** ₹ {coi_val/1e7:.2f} Cr/month welfare risk due to biometric backlog."
)

st.write(
    f"**Insight:** Saturation Gap highlights unmet biometric demand in **{district}**."
)

st.write(
    f"**Action:** Deploy **{vans}** mobile vans to protect **{saturation_gap:,}** children."
)

st.caption("Sentinel-Aadhaar | UIDAI Hackathon Final Submission | 2026")