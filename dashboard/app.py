import streamlit as st

st.set_page_config(page_title="SignalForge", page_icon="◉", layout="wide", initial_sidebar_state="expanded")
METRICS=[("Fraud risk","Calibrated","classifier"),("Graph risk","Enabled","shared entities"),("Anomaly model","Isolation Forest","novelty"),("Expected loss","Modeled","regression"),("Decision actions","3","allow / review / block"),("Active learning","Enabled","uncertainty"),("Drift monitor","PSI","population"),("Batch path","PySpark","offline"),("API","FastAPI","real-time"),("Feature parity","Shared","online / batch"),("Synthetic data","100%","portfolio"),("Auto-enforcement","Off","evaluation only")]
SIGNALS=[("Classifier confidence",.82),("Graph evidence",.76),("Anomaly evidence",.67),("Expected-loss pressure",.71),("Decision margin",.64)]

st.markdown("""<style>
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Arial,sans-serif;color:#1d1d1f}.stApp{background:#f5f5f7}[data-testid="stHeader"]{background:transparent}[data-testid="stSidebar"]{background:#fff;border-right:1px solid #e5e5ea}.block-container{max-width:1500px;padding:2rem 2.4rem 4rem}.hero{background:linear-gradient(135deg,#fff,#f7fbff);border:1px solid #e5e5ea;border-radius:32px;padding:38px 42px;margin-bottom:24px;box-shadow:0 14px 36px rgba(0,0,0,.045)}.eyebrow{color:#0071e3;font-size:.78rem;font-weight:700;letter-spacing:.11em;text-transform:uppercase}.hero h1{font-size:3.35rem;letter-spacing:-.052em;margin:.22rem 0 .55rem}.hero p{max-width:900px;color:#6e6e73;font-size:1.12rem;line-height:1.55}.pill{display:inline-block;background:#eef6ff;color:#0066cc;border:1px solid #d8eaff;border-radius:999px;padding:.42rem .78rem;margin:.55rem .35rem 0 0;font-size:.76rem;font-weight:650}[data-testid="stMetric"]{background:#fff;border:1px solid #e5e5ea;border-radius:24px;padding:18px 20px;box-shadow:0 8px 26px rgba(0,0,0,.035);min-height:116px}[data-testid="stMetricLabel"]{color:#6e6e73;font-weight:600}[data-testid="stMetricValue"]{font-size:1.9rem;font-weight:700}.stTabs [data-baseweb="tab"]{background:#fff;border:1px solid #e5e5ea;border-radius:999px;padding:8px 16px}.card{background:#fff;border:1px solid #e5e5ea;border-radius:22px;padding:18px 20px}.note{background:#fff;border:1px solid #e5e5ea;border-radius:18px;padding:14px 18px;color:#6e6e73}</style>""",unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## SignalForge"); st.caption("Fraud & Abuse Decision Intelligence"); st.divider(); st.markdown("**Overview**\n\nDecision health\n\nModel stack\n\nAdversarial monitoring\n\nReview queue"); st.divider(); st.caption("Synthetic / offline evaluation")
st.markdown("""<div class="hero"><div class="eyebrow">Fraud & Abuse Decision Intelligence</div><h1>SignalForge</h1><p>Calibrated fraud risk, graph signals, anomaly detection, expected loss, and cost-aware actioning.</p><span class="pill">Calibration</span><span class="pill">Graph risk</span><span class="pill">Expected loss</span><span class="pill">Active learning</span></div>""",unsafe_allow_html=True)
for s in range(0,len(METRICS),4):
    cols=st.columns(4)
    for c,(l,v,n) in zip(cols,METRICS[s:s+4]): c.metric(l,v,n)
st.subheader("Decision health")
l,r=st.columns([1.15,.85],gap="large")
with l:
    for name,val in SIGNALS: st.progress(val,text=f"{name} · {val:.0%}")
with r: st.markdown('<div class="card"><b>Decision economics</b><br><br><span style="color:#6e6e73">The model score is evidence. Expected loss, graph context, anomaly evidence, customer friction, and review cost determine the modeled action.</span></div>',unsafe_allow_html=True)
t1,t2,t3,t4=st.tabs(["Decision health","Model stack","Adversarial monitoring","Review queue"])
with t1: st.dataframe([{"Action":"ALLOW","Purpose":"low modeled cost"},{"Action":"REVIEW","Purpose":"uncertainty / exposure"},{"Action":"BLOCK","Purpose":"highest modeled risk"}],use_container_width=True,hide_index=True)
with t2: st.dataframe([{"Layer":"Classifier","Method":"Gradient Boosting + calibration"},{"Layer":"Graph","Method":"interpretable shared-entity model"},{"Layer":"Anomaly","Method":"Isolation Forest"},{"Layer":"Expected loss","Method":"Gradient Boosting Regressor"}],use_container_width=True,hide_index=True)
with t3:
    for name,val in SIGNALS: st.progress(val,text=name)
with t4: st.info("Review prioritization is synthetic and advisory. No customer, payment, device, or production telemetry is included.")
st.markdown('<div class="note"><b>Evaluation boundary.</b> KPI values are descriptive product-state labels or clearly marked synthetic/illustrative defaults. They are not production fraud-performance claims.</div>',unsafe_allow_html=True)
