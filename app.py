import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import time
import PyPDF2
import requests
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Nexus ESG AI Benchmarking", page_icon="🌍", layout="wide")

# --- UI HEADER ---
st.title("🌍 Nexus AI: Enterprise ESG Audit")
st.markdown("AI-powered analysis of 50-page reports using the **Nexus A.I.M. Methodology**. Benchmarking E, S, G factors, product footprints, and climate resilience.")

# --- HELPER FUNCTION: TEST API KEY ---
def test_nvidia_key(key):
    clean_key = key.strip()
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {clean_key}", "Accept": "application/json"}
    payload = {
      "model": "meta/llama-3.3-70b-instruct",
      "messages":[{"role": "user", "content": "Hello"}],
      "max_tokens": 10
    }
    try:
        response = requests.post(invoke_url, headers=headers, json=payload)
        if response.status_code == 200:
            return True, "✅ Connection Successful! Llama 3.3 70B is ready."
        else:
            return False, f"❌ Failed (Code {response.status_code}): {response.text}"
    except Exception as e:
        return False, f"❌ Connection Error: {e}"

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("NVIDIA NIM API Key", type="password")
    
    if st.button("🔌 Test API Connection"):
        if not api_key:
            st.warning("Please paste an API key first.")
        else:
            with st.spinner("Testing..."):
                success, msg = test_nvidia_key(api_key)
                if success: st.success(msg)
                else: st.error(msg)
                
    st.divider()
    use_demo = st.checkbox("Enable Demo Mode (Instant Charts)", value=False)
    uploaded_file = st.file_uploader("Upload ESG Report (PDF)", type=["pdf"])
    analyze_btn = st.button("Run Deep AI Analysis (50 Pages)", type="primary")

# --- DUMMY DATA FOR DEMO MODE ---
demo_json = {
    "company_name": "Acme Global Corp",
    "kpi_quality": "Average",
    "overall_score": 68,
    "esg_scores": {"environmental": 75, "social": 60, "governance": 82},
    "esg_justifications": {
        "environmental": "Strong commitments to renewable energy, but lacking in water management.",
        "social": "Good diversity metrics, but supply chain labor auditing is weak.",
        "governance": "Excellent board independence and transparent executive pay."
    },
    "product_analysis":[
        {"category": "Solar Components", "type": "Green", "percentage": 35},
        {"category": "Legacy Manufacturing", "type": "Carbon-Intensive", "percentage": 45},
        {"category": "Consulting Services", "type": "Neutral", "percentage": 20}
    ],
    "climate_resilience": 70,
    "investment_threats":[
        "High exposure to carbon pricing regulations in EU markets.",
        "Supply chain concentration risk in water-stressed regions.",
        "Potential labor union strikes due to stagnant wages in legacy divisions."
    ],
    "transparency_index": 65,
    "unmanaged_risk": {
        "inherent_exposure": 80, 
        "policies_score": -20, 
        "actions_score": -15, 
        "results_score": -15, 
        "final_unmanaged_risk": 30
    },
    "materiality":[
        {"topic": "Carbon Emissions", "financial_risk": 9, "world_impact": 8},
        {"topic": "Labor Rights", "financial_risk": 6, "world_impact": 9},
        {"topic": "Supply Chain", "financial_risk": 8, "world_impact": 6}
    ]
}

# --- HELPER FUNCTION: EXTRACT 50 PAGES ---
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    num_pages = min(50, len(reader.pages))
    for i in range(num_pages):
        text += reader.pages[i].extract_text() + "\n"
    return text

# --- HELPER FUNCTION: CALL NVIDIA Llama 3 ---
def analyze_esg_report(text, key):
    clean_key = key.strip()
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {clean_key}", "Accept": "application/json"}
    
    prompt = f"""
    You are a Senior ESG Investment Auditor. I am providing you with up to 50 pages of a company's sustainability report.
    You must evaluate the company using the "Nexus A.I.M. Methodology" (Assess, Integrate, Measure).
    
    METHODOLOGY RULES:
    1. ASSESS: Grade Environmental, Social, and Governance (E, S, G) independently from 0-100.
    2. KPI QUALITY: Classify the company as "Laggard" (0-40), "Average" (41-75), or "Leader" (76-100).
    3. PRODUCT BIFURCATION: Estimate the percentage of their business/revenue split between "Green", "Carbon-Intensive", and "Neutral" products.
    4. THREATS: Identify the top 3 investment threats (e.g., climate transition risk, supply chain issues).
    5. DOUBLE MATERIALITY & UNMANAGED RISK: Grade financial vs impact risks.
    
    OUTPUT FORMAT: You MUST return ONLY a valid JSON object. No markdown, no explanations. 
    Exact JSON structure required:
    {{
        "company_name": "Name",
        "kpi_quality": "Average",
        "overall_score": 75,
        "esg_scores": {{"environmental": 80, "social": 60, "governance": 70}},
        "esg_justifications": {{"environmental": "1 sentence reason", "social": "1 sentence reason", "governance": "1 sentence reason"}},
        "product_analysis":[
            {{"category": "Product A", "type": "Green", "percentage": 40}},
            {{"category": "Product B", "type": "Carbon-Intensive", "percentage": 60}}
        ],
        "climate_resilience": 65,
        "investment_threats":["Threat 1", "Threat 2", "Threat 3"],
        "transparency_index": 50,
        "unmanaged_risk": {{"inherent_exposure": 80, "policies_score": -15, "actions_score": -20, "results_score": -10, "final_unmanaged_risk": 35}},
        "materiality":[
            {{"topic": "Topic 1", "financial_risk": 8, "world_impact": 7}},
            {{"topic": "Topic 2", "financial_risk": 6, "world_impact": 9}}
        ]
    }}
    
    Report Text:
    {text}
    """
    
    payload = {
      "model": "meta/llama-3.3-70b-instruct",
      "messages":[
          {"role": "system", "content": "You are a precise data extraction AI. You only output valid JSON."},
          {"role": "user", "content": prompt}
      ],
      "temperature": 0.1,
      "max_tokens": 2048,
      "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")
        
    result_text = response.json()['choices'][0]['message']['content'].strip()
    result_text = re.sub(r'^```json\s*', '', result_text, flags=re.IGNORECASE)
    result_text = re.sub(r'^```\s*', '', result_text)
    result_text = re.sub(r'\s*```$', '', result_text)
    
    return json.loads(result_text)

# --- MAIN APP LOGIC ---
if analyze_btn:
    if use_demo:
        with st.spinner("Loading Demo Data..."):
            time.sleep(1)
            data = demo_json
    else:
        if not api_key:
            st.error("❌ Please enter your NVIDIA API key.")
            st.stop()
        if not uploaded_file:
            st.error("❌ Please upload a PDF report.")
            st.stop()
            
        with st.spinner("📖 Reading 50 Pages of PDF text..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
            except Exception as e:
                st.error(f"❌ PDF Error: {e}")
                st.stop()
            
        with st.spinner("🧠 Llama 3.3 is applying the Nexus Methodology... (Takes 20-40 seconds)"):
            try:
                data = analyze_esg_report(pdf_text, api_key)
            except Exception as e:
                st.error(f"❌ AI Parsing Error: {e}")
                st.stop()
            
    st.success(f"✅ Audit Complete: **{data['company_name']}**")
    
    # --- ROW 1: KPI METRIC CARDS ---
    st.markdown("### 📊 High-Level ESG Performance")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(label="Overall AI Score", value=f"{data['overall_score']}/100", delta=data['kpi_quality'])
    kpi2.metric(label="🌿 Environmental", value=data['esg_scores']['environmental'])
    kpi3.metric(label="🤝 Social", value=data['esg_scores']['social'])
    kpi4.metric(label="⚖️ Governance", value=data['esg_scores']['governance'])
    
    st.divider()

    # --- ROW 2: ESG BREAKDOWN & PRODUCT BIFURCATION ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### E, S, G Pillar Breakdown")
        df_esg = pd.DataFrame({
            "Pillar": ["Environmental", "Social", "Governance"],
            "Score": [data['esg_scores']['environmental'], data['esg_scores']['social'], data['esg_scores']['governance']]
        })
        fig_bar = px.bar(
            df_esg, 
            x="Pillar", 
            y="Score", 
            color="Pillar", 
            color_discrete_map={"Environmental":"#2ca02c", "Social":"#1f77b4", "Governance":"#ff7f0e"}
        )
        fig_bar.update_layout(
            height=300, 
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        with st.expander("📝 Read AI Pillar Justifications"):
            st.markdown(f"**Environmental:** {data['esg_justifications']['environmental']}")
            st.markdown(f"**Social:** {data['esg_justifications']['social']}")
            st.markdown(f"**Governance:** {data['esg_justifications']['governance']}")

    with col2:
        st.markdown("#### Product Portfolio: Green vs Carbon-Intensive")
        df_prod = pd.DataFrame(data["product_analysis"])
        fig_pie = px.pie(
            df_prod, 
            values='percentage', 
            names='category', 
            color='type',
            color_discrete_map={"Green":"#2ca02c", "Neutral":"#c7c7c7", "Carbon-Intensive":"#d62728"},
            hole=0.4
        )
        fig_pie.update_layout(
            height=300, 
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # --- ROW 3: DOUBLE MATERIALITY & UNMANAGED RISK ---
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### Double Materiality Matrix")
        df_mat = pd.DataFrame(data["materiality"])
        fig_scatter = px.scatter(
            df_mat, 
            x="financial_risk", 
            y="world_impact", 
            text="topic", 
            size=[15]*len(df_mat), 
            color="world_impact", 
            color_continuous_scale="Reds"
        )
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(
            xaxis_title="Financial Materiality (Risk to Business)",
            yaxis_title="Impact Materiality (Risk to World)",
            xaxis=dict(range=[0, 10]), 
            yaxis=dict(range=[0, 10]),
            height=350, 
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col4:
        st.markdown("#### Unmanaged ESG Risk (Industry vs Actions)")
        ur = data["unmanaged_risk"]
        fig_waterfall = go.Figure(go.Waterfall(
            name="Risk", 
            orientation="v", 
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=["Inherent Risk", "Policies", "Actions", "Results", "Unmanaged Risk"], 
            textposition="outside",
            y=[ur["inherent_exposure"], ur["policies_score"], ur["actions_score"], ur["results_score"], ur["final_unmanaged_risk"]],
            connector={"line":{"color":"rgb(63, 63, 63)"}}, 
            decreasing={"marker":{"color":"#2ca02c"}}, 
            increasing={"marker":{"color":"#d62728"}}, 
            totals={"marker":{"color":"#1f77b4"}}
        ))
        fig_waterfall.update_layout(
            height=350, 
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    st.divider()

    # --- ROW 4: THREATS & DOWNLOAD ---
    st.markdown("### ⚠️ Investment Threats & Climate Resilience")
    st.metric("Climate Crisis Resilience Score", f"{data['climate_resilience']}/100")
    
    st.error("**Top 3 Red Flags / Threats to consider before investing:**")
    for threat in data["investment_threats"]:
        st.markdown(f"- {threat}")

    st.divider()
    
    # --- EXPORT REPORT FEATURE ---
    report_text = f"""
    ================================================
    NEXUS AI - ESG AUDIT REPORT
    ================================================
    Company: {data['company_name']}
    Overall Rating: {data['kpi_quality']} (Score: {data['overall_score']}/100)
    Climate Resilience: {data['climate_resilience']}/100
    
    --- E, S, G BREAKDOWN ---
    Environmental: {data['esg_scores']['environmental']} - {data['esg_justifications']['environmental']}
    Social: {data['esg_scores']['social']} - {data['esg_justifications']['social']}
    Governance: {data['esg_scores']['governance']} - {data['esg_justifications']['governance']}
    
    --- TOP INVESTMENT THREATS ---
    """
    for t in data['investment_threats']: report_text += f"\n- {t}"
    
    st.download_button(
        label="📥 Download AI Audited Report (TXT)",
        data=report_text,
        file_name=f"{data['company_name'].replace(' ', '_')}_ESG_Audit.txt",
        mime="text/plain",
        type="primary"
    )
