
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── Allow importing from src/ ─────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from traffic_control import (
    compute_signal_timing, recommend_routes,
    generate_alerts, control_summary, CONGESTION_LABELS,
)
from locations import LOCATION_NAMES, get_name, add_names_to_df

# ── Sklearn ───────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Control System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiselect label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important; font-size: 12px;
    letter-spacing: 0.05em; text-transform: uppercase;
}
.metric-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 20px 24px; text-align: center;
}
.metric-label { font-size: 12px; color: #64748b; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 6px; }
.metric-value { font-size: 32px; font-weight: 600; color: #0f172a; line-height: 1; }
.metric-sub   { font-size: 12px; color: #94a3b8; margin-top: 4px; }
.section-title {
    font-size: 20px; font-weight: 600; color: #0f172a;
    border-left: 4px solid #3b82f6; padding-left: 12px; margin: 28px 0 16px;
}
.signal-box {
    background: #1e293b; border-radius: 16px; padding: 24px; text-align: center; margin: 8px 0;
}
.signal-circle {
    width: 70px; height: 70px; border-radius: 50%;
    margin: 8px auto; display: flex; align-items: center;
    justify-content: center; font-size: 28px;
}
.signal-active-red    { background: #ef4444; box-shadow: 0 0 20px #ef4444; }
.signal-active-yellow { background: #f59e0b; box-shadow: 0 0 20px #f59e0b; }
.signal-active-green  { background: #22c55e; box-shadow: 0 0 20px #22c55e; }
.signal-off           { background: #374151; }
.signal-label { color: #94a3b8; font-size: 13px; margin-top: 4px; }
.alert-critical { background: #fef2f2; border-left: 4px solid #ef4444; border-radius: 8px; padding: 14px 16px; margin: 8px 0; }
.alert-warning  { background: #fffbeb; border-left: 4px solid #f59e0b; border-radius: 8px; padding: 14px 16px; margin: 8px 0; }
.alert-title { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
.alert-body  { font-size: 13px; color: #475569; }
.route-card  { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 16px; margin: 8px 0; }
.route-rank  { font-size: 22px; font-weight: 700; color: #3b82f6; }
.route-body  { font-size: 13px; color: #475569; margin-top: 4px; }
.badge-high   { background:#fef2f2; color:#dc2626; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-medium { background:#fffbeb; color:#d97706; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }
.badge-low    { background:#f0fdf4; color:#16a34a; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:500; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def loc_options():
    return [f"{lid} — {name}" for lid, name in sorted(LOCATION_NAMES.items())]

def parse_loc(selected: str) -> int:
    return int(selected.split(" — ")[0])

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    df = pd.read_csv("data/traffic_dataset_100k.csv")
    df["timestamp"]    = pd.to_datetime(df["timestamp"])
    df["hour"]         = df["timestamp"].dt.hour
    df["day"]          = df["timestamp"].dt.day
    df["month"]        = df["timestamp"].dt.month
    df["weekday"]      = df["timestamp"].dt.weekday
    df["weekday_name"] = df["timestamp"].dt.day_name()
    df["is_weekend"]   = df["weekday"].isin([5, 6]).astype(int)
    def tod(h):
        if 5 <= h < 12:    return "Morning"
        elif 12 <= h < 17: return "Afternoon"
        elif 17 <= h < 21: return "Evening"
        else:              return "Night"
    df["time_of_day"]        = df["hour"].apply(tod)
    df["speed_volume_ratio"] = (df["average_speed"] / df["traffic_volume"].replace(0, np.nan)).round(4)
    df["dist_from_ref_km"]   = (np.sqrt((df["latitude"]-13.0827)**2+(df["longitude"]-80.2707)**2)*111).round(2)
    df["congestion_encoded"] = df["congestion_level"].map({"Low":0,"Medium":1,"High":2})
    df = add_names_to_df(df)
    return df

@st.cache_resource
def train_models(df):
    feature_cols = ["traffic_volume","average_speed","hour","weekday",
                    "is_weekend","speed_volume_ratio","dist_from_ref_km","month","location_id"]
    X        = df[feature_cols].fillna(df[feature_cols].median())
    y        = df["congestion_encoded"]
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        results[name] = {
            "model": model,
            "accuracy":    accuracy_score(y_test, y_pred),
            "auc":         roc_auc_score(y_test, y_proba, multi_class="ovr"),
            "cv_accuracy": cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy", n_jobs=-1).mean(),
            "report":      classification_report(y_test, y_pred, target_names=["Low","Medium","High"], output_dict=True),
            "cm":          confusion_matrix(y_test, y_pred),
            "y_test": y_test, "y_pred": y_pred, "y_proba": y_proba,
        }
    rf_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": results["Random Forest"]["model"].feature_importances_
    }).sort_values("importance", ascending=False)
    return results, scaler, feature_cols, rf_imp

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚦 Traffic Control")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Overview", "🔍 EDA", "🤖 Predictions",
        "📈 Model Comparison", "🚦 Traffic Control",
    ])
    st.markdown("---")
    st.markdown("### Filters")
    df_raw       = load_and_preprocess()
    sel_cong     = st.multiselect("Congestion Level", ["Low","Medium","High"], default=["Low","Medium","High"])
    sel_hours    = st.slider("Hour Range", 0, 23, (0, 23))
    sel_locs_raw = st.multiselect("Locations", loc_options(), default=[])
    sel_locs     = [parse_loc(s) for s in sel_locs_raw]
    st.markdown("---")
    st.caption("B.TECH. Final Year Project\nTraffic Congestion Analysis\n& Control System")

df = df_raw.copy()
df = df[df["congestion_level"].isin(sel_cong)]
df = df[df["hour"].between(sel_hours[0], sel_hours[1])]
if sel_locs:
    df = df[df["location_id"].isin(sel_locs)]

COLORS = {"High":"#ef4444","Medium":"#f59e0b","Low":"#22c55e"}
PT     = "plotly_white"

# ═════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═════════════════════════════════════════════
if page == "📊 Overview":
    st.title("Traffic Congestion Analysis Dashboard")
    st.caption("100,000 traffic records · 50 locations · Thoothukudi district, Tamil Nadu")

    c1,c2,c3,c4 = st.columns(4)
    total_records = len(df)
    avg_traffic = df['traffic_volume'].mean() if not df.empty else 0
    avg_traffic = 0 if pd.isna(avg_traffic) else int(avg_traffic)
    avg_speed = df['average_speed'].mean() if not df.empty else 0.0
    avg_speed = round(avg_speed, 1) if not pd.isna(avg_speed) else 0.0
    high_cong_count = len(df[df['congestion_level']=='High'])
    high_cong_pct = (high_cong_count / total_records * 100) if total_records > 0 else 0.0
    high_cong_pct = round(high_cong_pct, 1)
    for col,label,val,sub in zip([c1,c2,c3,c4],
        ["Total Records","Avg Traffic Volume","Avg Speed","High Congestion"],
        [f"{total_records:,}", avg_traffic, avg_speed, f"{high_cong_pct}%"],
        ["filtered dataset","vehicles / interval","km/h","of filtered records"]):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div><div class="metric-value">{val}</div>
            <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Congestion Distribution</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        cc = df["congestion_level"].value_counts().reset_index()
        cc.columns = ["Level","Count"]
        fig = px.pie(cc, names="Level", values="Count", color="Level",
                     color_discrete_map=COLORS, hole=0.5, template=PT)
        fig.update_layout(height=320, margin=dict(t=20,b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        hourly = df.groupby(["hour","congestion_level"])["traffic_volume"].mean().reset_index()
        fig = px.bar(hourly, x="hour", y="traffic_volume", color="congestion_level",
                     color_discrete_map=COLORS, barmode="stack", template=PT,
                     labels={"traffic_volume":"Avg Volume","hour":"Hour","congestion_level":"Congestion"})
        fig.update_layout(height=320, margin=dict(t=20,b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Traffic Volume Over Time</div>', unsafe_allow_html=True)
    daily = df.groupby(df["timestamp"].dt.date)["traffic_volume"].mean().reset_index()
    daily.columns = ["date","avg_volume"]
    fig = px.line(daily, x="date", y="avg_volume", template=PT)
    fig.update_traces(line_color="#3b82f6", line_width=2)
    fig.update_layout(height=280, margin=dict(t=10,b=10))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Geographic Distribution</div>', unsafe_allow_html=True)
    geo = df.groupby(["location_id","location_name"]).agg(
        lat=("latitude","mean"), lon=("longitude","mean"),
        volume=("traffic_volume","mean"),
        dominant=("congestion_level", lambda x: x.value_counts().index[0])
    ).reset_index()
    fig = px.scatter_mapbox(geo, lat="lat", lon="lon", color="dominant", size="volume",
                            color_discrete_map=COLORS, hover_name="location_name",
                            hover_data={"volume":":.0f","lat":False,"lon":False},
                            mapbox_style="carto-positron", zoom=11, template=PT)
    fig.update_layout(height=450, margin=dict(t=10,b=10))
    st.plotly_chart(fig, width='stretch')

# ═════════════════════════════════════════════
# PAGE 2 — EDA
# ═════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")

    st.markdown('<div class="section-title">Feature Distributions</div>', unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="traffic_volume", color="congestion_level", nbins=40,
                           color_discrete_map=COLORS, barmode="overlay", opacity=0.7, template=PT)
        fig.update_layout(height=300, title="Traffic Volume Distribution", margin=dict(t=40,b=10))
        st.plotly_chart(fig, width='stretch')
    with c2:
        fig = px.histogram(df, x="average_speed", color="congestion_level", nbins=40,
                           color_discrete_map=COLORS, barmode="overlay", opacity=0.7, template=PT)
        fig.update_layout(height=300, title="Speed Distribution", margin=dict(t=40,b=10))
        st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Speed vs Volume</div>', unsafe_allow_html=True)
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig = px.scatter(sample, x="traffic_volume", y="average_speed", color="congestion_level",
                     color_discrete_map=COLORS, hover_name="location_name", opacity=0.5, template=PT)
    fig.update_layout(height=380, margin=dict(t=10,b=10))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Congestion Heatmap: Hour × Day</div>', unsafe_allow_html=True)
    pivot = (df.groupby(["weekday_name","hour"])["congestion_encoded"].mean().reset_index()
               .pivot(index="weekday_name", columns="hour", values="congestion_encoded")
               .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]))
    fig = px.imshow(pivot, color_continuous_scale="RdYlGn_r", template=PT, aspect="auto",
                    labels={"x":"Hour of Day","y":"Day","color":"Avg Congestion"})
    fig.update_layout(height=320, margin=dict(t=10,b=10))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Top 10 Busiest Locations</div>', unsafe_allow_html=True)
    top10 = (df.groupby(["location_id","location_name"])["traffic_volume"]
               .mean().sort_values(ascending=False).head(10).reset_index())
    fig = px.bar(top10, x="location_name", y="traffic_volume",
                 color="traffic_volume", color_continuous_scale="Blues",
                 template=PT, text_auto=".0f")
    fig.update_layout(height=380, margin=dict(t=10,b=10), coloraxis_showscale=False,
                      xaxis_tickangle=-30)
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df[["traffic_volume","average_speed","hour","weekday",
               "is_weekend","dist_from_ref_km","congestion_encoded"]].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, template=PT, aspect="auto")
    fig.update_layout(height=400, margin=dict(t=10,b=10))
    st.plotly_chart(fig, width='stretch')

# ═════════════════════════════════════════════
# PAGE 3 — PREDICTIONS
# ═════════════════════════════════════════════
elif page == "🤖 Predictions":
    st.title("Congestion Prediction")
    results, scaler, feature_cols, _ = train_models(df_raw)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    st.info(f"Best model: **{best_name}** — Accuracy: {results[best_name]['accuracy']*100:.1f}%", icon="🏆")

    st.markdown('<div class="section-title">Manual Prediction</div>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        traffic_vol = st.slider("Traffic Volume", 10, 500, 250)
        avg_speed   = st.slider("Average Speed (km/h)", 10, 80, 45)
        hour_input  = st.slider("Hour of Day", 0, 23, 8)
    with c2:
        wd_input    = st.selectbox("Day", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
        wd_num      = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"].index(wd_input)
        is_wknd     = 1 if wd_num >= 5 else 0
        month_input = st.selectbox("Month", list(range(1,13)),
                                   format_func=lambda m: pd.Timestamp(2025,m,1).strftime("%B"))
        loc_sel     = st.selectbox("Location", loc_options())
        loc_id      = parse_loc(loc_sel)
    with c3:
        model_ch    = st.selectbox("Model", list(results.keys()))
        svr         = round(avg_speed / max(traffic_vol,1), 4)
        loc_lat     = df_raw[df_raw["location_id"]==loc_id]["latitude"].mean()
        loc_lon     = df_raw[df_raw["location_id"]==loc_id]["longitude"].mean()
        dist        = round(np.sqrt((loc_lat-13.0827)**2+(loc_lon-80.2707)**2)*111, 2)
        st.metric("Speed/Volume Ratio", f"{svr:.4f}")
        st.metric("Distance from Ref (km)", f"{dist:.1f}")

    if st.button("🔮 Predict Congestion", width='stretch'):
        inp      = np.array([[traffic_vol, avg_speed, hour_input, wd_num,
                              is_wknd, svr, dist, month_input, loc_id]])
        pred     = results[model_ch]["model"].predict(scaler.transform(inp))[0]
        proba    = results[model_ch]["model"].predict_proba(scaler.transform(inp))[0]
        pred_lbl = {0:"Low",1:"Medium",2:"High"}[pred]
        badge    = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high"}[pred_lbl]
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                    padding:24px;text-align:center;margin:16px 0;">
            <div style="font-size:13px;color:#64748b;margin-bottom:4px;">{get_name(loc_id)}</div>
            <div style="font-size:14px;color:#64748b;margin-bottom:8px;">Predicted Congestion Level</div>
            <span class="{badge}" style="font-size:22px;padding:8px 28px;">{pred_lbl}</span>
        </div>""", unsafe_allow_html=True)
        for col,lvl,prob in zip(st.columns(3), ["Low","Medium","High"], proba):
            col.metric(f"{lvl} probability", f"{prob*100:.1f}%")

# ═════════════════════════════════════════════
# PAGE 4 — MODEL COMPARISON
# ═════════════════════════════════════════════
elif page == "📈 Model Comparison":
    st.title("Model Comparison")
    with st.spinner("Training models..."):
        results, scaler, feature_cols, rf_imp = train_models(df_raw)

    st.markdown('<div class="section-title">Performance Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame([{"Model":name,"Accuracy":f"{r['accuracy']*100:.2f}%",
                              "ROC-AUC":f"{r['auc']:.4f}","CV Accuracy":f"{r['cv_accuracy']*100:.2f}%"}
                             for name,r in results.items()])
    st.dataframe(summary.set_index("Model"), width='stretch')

    st.markdown('<div class="section-title">Accuracy Comparison</div>', unsafe_allow_html=True)
    mdf = pd.DataFrame([{"Model":name,"Accuracy":r["accuracy"]*100,
                          "ROC-AUC":r["auc"]*100,"CV Accuracy":r["cv_accuracy"]*100}
                         for name,r in results.items()])
    fig = px.bar(mdf.melt(id_vars="Model",var_name="Metric",value_name="Score (%)"),
                 x="Model", y="Score (%)", color="Metric",
                 barmode="group", template=PT, text_auto=".1f")
    fig.update_layout(height=360, margin=dict(t=10,b=10))
    st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Confusion Matrices</div>', unsafe_allow_html=True)
    for i,(name,r) in zip(range(3), results.items()):
        with st.columns(3)[i]:
            fig = px.imshow(r["cm"], text_auto=True,
                            x=["Low","Medium","High"], y=["Low","Medium","High"],
                            color_continuous_scale="Blues", template=PT,
                            labels={"x":"Predicted","y":"Actual"})
            fig.update_layout(height=280, title=name, margin=dict(t=40,b=10), coloraxis_showscale=False)
            st.plotly_chart(fig, width='stretch')

    st.markdown('<div class="section-title">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    fig = px.bar(rf_imp, x="importance", y="feature", orientation="h",
                 color="importance", color_continuous_scale="Blues", template=PT, text_auto=".3f")
    fig.update_layout(height=360, margin=dict(t=10,b=10),
                      yaxis={"categoryorder":"total ascending"}, coloraxis_showscale=False)
    st.plotly_chart(fig, width='stretch')

# ═════════════════════════════════════════════
# PAGE 5 — TRAFFIC CONTROL
# ═════════════════════════════════════════════
elif page == "🚦 Traffic Control":
    st.title("Intelligent Traffic Control System")
    st.caption("Smart signal timing · Route recommendation · Real-time alerts")

    ctrl_hour = st.slider("Select Hour to Analyse", 0, 23, 8, format="%d:00")

    # Network status
    summary = control_summary(df_raw, ctrl_hour)
    st.markdown('<div class="section-title">Network Status</div>', unsafe_allow_html=True)
    for col,label,val,sub,color in zip(
        st.columns(5),
        ["Network Health","High Congestion","Medium Congestion","Avg Speed","Avg Volume"],
        [f"{summary['network_health']}%", summary['high_congestion'],
         summary['med_congestion'], summary['avg_speed'], summary['avg_volume']],
        [summary['status'],"locations","locations","km/h","vehicles"],
        ["#0f172a","#ef4444","#f59e0b","#0f172a","#0f172a"]
    ):
        col.markdown(f"""<div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value" style="color:{color}">{val}</div>
            <div class="metric-sub">{sub}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    left, mid, right = st.columns(3)

    # Signal Timing
    with left:
        st.markdown('<div class="section-title">🚥 Smart Signal Timing</div>', unsafe_allow_html=True)
        sig_sel   = st.selectbox("Junction", loc_options(), key="sig_loc")
        sig_loc   = parse_loc(sig_sel)
        sig_vol   = st.slider("Current Volume", 10, 500, 250, key="sig_vol")
        sig_speed = st.slider("Current Speed (km/h)", 10, 80, 40, key="sig_speed")
        sig_cong  = st.selectbox("Current Congestion", ["Low","Medium","High"], key="sig_cong")

        if st.button("⚡ Compute Signal Timing", width='stretch'):
            timing  = compute_signal_timing(sig_cong, sig_vol, sig_speed)
            is_high = sig_cong == "High"
            is_med  = sig_cong == "Medium"
            st.markdown(f"""
            <div class="signal-box">
                <div style="color:#94a3b8;font-size:13px;margin-bottom:12px;">{get_name(sig_loc)}</div>
                <div class="signal-circle {'signal-active-red' if is_high else 'signal-off'}">🔴</div>
                <div class="signal-label">RED — {timing['red_time']}s</div>
                <div class="signal-circle {'signal-active-yellow' if is_med else 'signal-off'}" style="margin-top:8px;">🟡</div>
                <div class="signal-label">YELLOW — {timing['yellow_time']}s</div>
                <div class="signal-circle {'signal-active-green' if not is_high else 'signal-off'}" style="margin-top:8px;">🟢</div>
                <div class="signal-label">GREEN — {timing['green_time']}s</div>
                <div style="color:#94a3b8;font-size:12px;margin-top:14px;">
                    Cycle: {timing['cycle_time']}s &nbsp;|&nbsp; Green efficiency: {timing['green_efficiency']}%
                </div>
            </div>
            <div style="background:#f0fdf4;border-left:3px solid #22c55e;border-radius:6px;
                        padding:10px 12px;margin-top:10px;font-size:13px;color:#166534;">
                {timing['recommendation']}
            </div>""", unsafe_allow_html=True)

    # Route Recommendation
    with mid:
        st.markdown('<div class="section-title">🗺️ Route Recommendation</div>', unsafe_allow_html=True)
        route_sel = st.selectbox("Current Location", loc_options(), key="route_loc")
        route_loc = parse_loc(route_sel)

        if st.button("🔍 Find Alternate Routes", width='stretch'):
            rd        = recommend_routes(df_raw, route_loc, ctrl_hour)
            badge_cls = {"Low":"badge-low","Medium":"badge-medium","High":"badge-high"}
            st.markdown(f"""
            <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px 16px;margin-bottom:12px;">
                <div style="font-size:12px;color:#64748b;">Current location</div>
                <div style="font-weight:600;margin:2px 0;">{get_name(route_loc)}</div>
                <div><span class="{badge_cls[rd['current_congestion']]}">{rd['current_congestion']}</span>
                    &nbsp; Vol: {rd['current_volume']} &nbsp; Speed: {rd['current_speed']} km/h
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("**Top alternate routes:**")
            for i, r in enumerate(rd["alternate_routes"], 1):
                st.markdown(f"""
                <div class="route-card">
                    <div style="display:flex;align-items:center;gap:10px;">
                        <div class="route-rank">#{i}</div>
                        <div><b>{get_name(r['location_id'])}</b> &nbsp;
                            <span class="{badge_cls[r['congestion']]}">{r['congestion']}</span></div>
                    </div>
                    <div class="route-body">Speed: {r['avg_speed']} km/h &nbsp;|&nbsp;
                        Volume: {r['avg_volume']} &nbsp;|&nbsp; Distance: {r['distance_km']} km</div>
                    <div style="font-size:12px;margin-top:6px;">{r['advice']}</div>
                </div>""", unsafe_allow_html=True)

            map_df = pd.DataFrame(rd["alternate_routes"])
            map_df["name"] = map_df["location_id"].apply(get_name)
            fig = px.scatter_mapbox(map_df, lat="lat", lon="lon", color="congestion",
                                    size=[20]*len(map_df), color_discrete_map=COLORS,
                                    hover_name="name", mapbox_style="carto-positron",
                                    zoom=11, template=PT)
            fig.update_layout(height=250, margin=dict(t=5,b=5))
            st.plotly_chart(fig, width='stretch')

    # Alert System
    with right:
        st.markdown('<div class="section-title">🚨 Real-Time Alerts</div>', unsafe_allow_html=True)
        if st.button("🔔 Scan All Locations", width='stretch'):
            alerts = generate_alerts(df_raw, ctrl_hour, top_n=10)
            if not alerts:
                st.success("✅ No alerts. Network is clear.")
            else:
                crit = [a for a in alerts if a["severity"]=="Critical"]
                warn = [a for a in alerts if a["severity"]=="Warning"]
                st.markdown(f"**{len(crit)} Critical &nbsp;|&nbsp; {len(warn)} Warnings**")
                for a in alerts:
                    css = "alert-critical" if a["severity"]=="Critical" else "alert-warning"
                    icon = "🚨" if a["severity"]=="Critical" else "⚠️"
                    st.markdown(f"""
                    <div class="{css}">
                        <div class="alert-title">{icon} {get_name(a['location_id'])} — {a['severity']}</div>
                        <div class="alert-body">{"<br>".join(a['alerts'])}<br>
                            Vol: {a['avg_volume']} &nbsp;|&nbsp; Speed: {a['avg_speed']} km/h
                            &nbsp;|&nbsp; {a['congestion']}</div>
                        <div style="font-size:12px;margin-top:6px;color:#475569;">{a['action']}</div>
                    </div>""", unsafe_allow_html=True)

    # Full alert map
    st.markdown('<div class="section-title">Alert Map — All Locations</div>', unsafe_allow_html=True)
    geo_all = df_raw[df_raw["hour"]==ctrl_hour].groupby(["location_id","location_name"]).agg(
        lat=("latitude","mean"), lon=("longitude","mean"),
        volume=("traffic_volume","mean"),
        congestion=("congestion_level", lambda x: x.value_counts().index[0])
    ).reset_index()
    fig = px.scatter_mapbox(geo_all, lat="lat", lon="lon", color="congestion", size="volume",
                            color_discrete_map=COLORS, hover_name="location_name",
                            hover_data={"volume":":.0f","lat":False,"lon":False},
                            mapbox_style="carto-positron", zoom=11, template=PT,
                            title=f"Traffic Status at {ctrl_hour}:00")
    fig.update_layout(height=450, margin=dict(t=40,b=10))
    st.plotly_chart(fig, width='stretch')
