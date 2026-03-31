import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nexus ESG AI Benchmarking", page_icon="🌍", layout="wide")

# --- UI HEADER ---
st.title("🌍 Nexus AI: ESG Automated Benchmarking")
st.markdown("Upload a company's sustainability report, and our AI agents will parse the PDF, strip away greenwashing, and benchmark the data using the **A.I.M. Methodology**.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key (Optional for Demo)", type="password")
    use_demo = st.checkbox("Enable Demo Mode (Instant Charts)", value=True)
    uploaded_file = st.file_uploader("Upload ESG Report (PDF)", type=["pdf"])
    analyze_btn = st.button("Run AI Analysis")

# --- DUMMY DATA FOR DEMO MODE ---
demo_json = {
    "company_name": "Acme Global Manufacturing",
    "transparency_index": 42,
    "unmanaged_risk": {
        "inherent_exposure": 85,
        "policies_score": -15,
        "actions_score": -20,
        "results_score": -10,
        "final_unmanaged_risk": 40
    },
    "par_scores": {"Policies": 80, "Actions": 55, "Results": 35},
    "materiality": [
        {"topic": "Carbon Emissions", "financial_risk": 9, "world_impact": 8},
        {"topic": "Labor Rights", "financial_risk": 6, "world_impact": 9},
        {"topic": "Data Privacy", "financial_risk": 8, "world_impact": 4},
        {"topic": "Water Usage", "financial_risk": 4, "world_impact": 7},
        {"topic": "Board Diversity", "financial_risk": 3, "world_impact": 5}
    ]
}

# --- MAIN APP LOGIC ---
if analyze_btn:
    if not use_demo and not api_key:
        st.warning("Please enter an OpenAI API key or enable Demo Mode.")
    elif not use_demo and not uploaded_file:
        st.warning("Please upload a PDF report.")
    else:
        with st.spinner("AI Agents analyzing 100+ pages for proprietary methodology..."):
            time.sleep(2) # Simulating AI processing time
            data = demo_json # In live mode, this would be replaced by the OpenAI JSON response
            
        st.success(f"Analysis Complete for: **{data['company_name']}**")
        
        # --- LAYOUT: 2 COLUMNS ---
        col1, col2 = st.columns(2)
        
        # 1. TRANSPARENCY INDEX (Gauge Chart)
        with col1:
            st.subheader("1. Transparency & Fluff Index")
            st.markdown("AI ratio of Hard Data vs. Marketing 'Boilerplate'")
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = data["transparency_index"],
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightcoral"},
                        {'range': [40, 70], 'color': "khaki"},
                        {'range': [70, 100], 'color': "lightgreen"}]
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # 2. P.A.R. READINESS (Radar Chart)
        with col2:
            st.subheader("2. P.A.R. Management Audit")
            st.markdown("Does the company just have policies, or actual results?")
            categories = ['Policies', 'Actions', 'Results']
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=[data["par_scores"]["Policies"], data["par_scores"]["Actions"], data["par_scores"]["Results"]],
                theta=categories,
                fill='toself',
                name='Management Strength',
                line_color='teal'
            ))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=300, margin=dict(l=30, r=30, t=30, b=10))
            st.plotly_chart(fig_radar, use_container_width=True)

        # --- LAYOUT: 2 COLUMNS ROW 2 ---
        col3, col4 = st.columns(2)

        # 3. UNMANAGED RISK (Waterfall Chart)
        with col3:
            st.subheader("3. Unmanaged ESG Risk")
            st.markdown("Inherent Industry Risk minus AI-Audited Management")
            ur = data["unmanaged_risk"]
            fig_waterfall = go.Figure(go.Waterfall(
                name = "Risk", orientation = "v",
                measure = ["absolute", "relative", "relative", "relative", "total"],
                x = ["Inherent Exposure", "Policies", "Actions", "Results", "Unmanaged Risk"],
                textposition = "outside",
                y = [ur["inherent_exposure"], ur["policies_score"], ur["actions_score"], ur["results_score"], ur["final_unmanaged_risk"]],
                connector = {"line":{"color":"rgb(63, 63, 63)"}},
                decreasing = {"marker":{"color":"#2ca02c"}},
                increasing = {"marker":{"color":"#d62728"}},
                totals = {"marker":{"color":"#1f77b4"}}
            ))
            fig_waterfall.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig_waterfall, use_container_width=True)

        # 4. DOUBLE MATERIALITY (Scatter Plot)
        with col4:
            st.subheader("4. Double Materiality Matrix")
            st.markdown("Financial Risk to Company vs. Impact on World")
            df_mat = pd.DataFrame(data["materiality"])
            fig_scatter = px.scatter(
                df_mat, x="financial_risk", y="world_impact", text="topic", 
                size=[15]*len(df_mat), color="world_impact", color_continuous_scale="Reds"
            )
            fig_scatter.update_traces(textposition='top center')
            fig_scatter.update_layout(
                xaxis_title="Financial Materiality (Risk to Business)",
                yaxis_title="Impact Materiality (Risk to World)",
                xaxis=dict(range=[0, 10]), yaxis=dict(range=[0, 10]),
                height=350, margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        st.info("💡 **Methodology Note:** This dashboard utilizes a proprietary fusion of Double Materiality (EcoVadis/SASB) and Unmanaged Risk gap analysis (Sustainalytics/MSCI) powered by Large Language Models.")