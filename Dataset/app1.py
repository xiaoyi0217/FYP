import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import requests
import json
import os

# ==== Page & CSS Setup ==== #
st.set_page_config(
    page_title="MindEase - Social Anxiety Tracker",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found! Using default styling.")

local_css("Dataset/style.css")

# ==== SIMPLE TEXT‑BASED AUTHENTICATION ==== #
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.sidebar.title("🔐 Login")
    user = st.sidebar.text_input("Username")
    pwd  = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Log in"):
        if {"kuan":"kuan","muru":"muru","xiaoyi":"xiaoyi"}.get(user) == pwd:
            st.session_state.logged_in = True
            st.session_state.username = user
        else:
            st.sidebar.error("❌ Invalid credentials")
    if not st.session_state.logged_in:
        st.stop()

st.sidebar.write(f"👋 Hello, **{st.session_state.username}**")

# APP LOGIC BELOW #
# ==== API KEY SETUP (Hybrid) ==== #
api_key = st.secrets.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
if not api_key:
    st.error("OpenRouter API key is missing! Add it to Streamlit secrets or set as environment variable.")
    st.stop()

# ==== OpenRouter AI Helper ==== #
# Define the function OUTSIDE the if block
def get_openrouter_response(prompt):
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            },
            data=json.dumps({
            "model": "deepseek/deepseek-chat:free",  # DeepSeek Model
            "messages": [
                {"role": "user", "content": prompt}
                ]
            })
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"OpenRouter API error: {e}")
    return None
# ==== Load ML Models ==== #
@st.cache_resource
def load_models():
    with open('Dataset/production_kmeans_model.pkl','rb') as f: kmeans = pickle.load(f)
    with open('Dataset/kmeans_scaler.pkl','rb')       as f: scaler = pickle.load(f)
    with open('Dataset/best_xgb_model.pkl','rb')      as f: xgb    = pickle.load(f)
    return kmeans, scaler, xgb

kmeans_model, kmeans_scaler, xgb_model = load_models()

# ==== Encoding Mappings ==== #
EDU_LEVEL_MAP       = {"Bachelor's":0,"Doctorate":1,"High School":2,"Master's":3}
SOCIOECONOMIC_MAP   = {'High':0,'Low':1,'Middle':2}
AGE_CAT_MAP        = {'Adult':0,'Senior':1,'Youth':2}
USAGE_INTENSITY_MAP = {'Low':1,'Medium':2,'High':0}
SOCIAL_ANX_MAP      = {0:'High',1:'Low',2:'Medium'}
PLOT_MAP            = {'Low':1,'Medium':2,'High':3}

FEATURE_NAMES = [
    'Likes Received (per post)',
    'Comments Received (per post)',
    'Peer Comparison Frequency (1-10)',
    'Socioeconomic Status',
    'Education Level',
    'Body Image Impact (1-10)',
    'Sleep Quality Impact (1-10)',
    'Self Confidence Impact (1-10)',
    'Cyberbullying Experience (1-10)',
    'Anxiety Levels (1-10)',
    'Age Category',
    'Total Social Interaction',
    'Usage Intensity',
    'Usage Anxiety Interaction'
]

DATA_FILE = "Dataset/user_data.csv"
if not os.path.exists(DATA_FILE):
    cols = ['username'] + FEATURE_NAMES + ['Social Anxiety Category','Cluster','Timestamp']
    pd.DataFrame(columns=cols).to_csv(DATA_FILE, index=False)

def home_page():
    st.title("🏠 Welcome to MindEase")
    st.write("""
        MindEase combines AI-driven classification and clustering to help you track
        and manage social anxiety. Log daily metrics, view your progress, and get tips.
    """)

def new_entry_page():
    st.header("➕ Log Your Daily Metrics")
    with st.form("entry_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            education    = st.selectbox("Education Level", list(EDU_LEVEL_MAP.keys()))
            socio  = st.selectbox("Socioeconomic Status", list(SOCIOECONOMIC_MAP.keys()))
            peer   = st.slider("Peer Comparison (1-10)",1,10,5)
            body   = st.slider("Body Image Impact (1-10)",1,10,5)
            sleep_quality  = st.slider("Sleep Quality Impact (1-10)",1,10,5)
            conf   = st.slider("Self Confidence Impact (1-10)",1,10,5)
        with c2:
            cyber = st.slider("Cyberbullying Experience (1-10)",1,10,1)
            anxiety=st.slider("Anxiety Levels (1-10)",1,10,5)
            age_cat   = st.selectbox("Age Category", list(AGE_CAT_MAP.keys()))
            total_int = st.number_input("Total Social Interaction",0,10000,10)
            sm_usage_hrs   = st.number_input("Daily SM Usage (hrs)",0.0,24.0,3.0)

        if st.form_submit_button("Submit"):
            df_existing = pd.read_csv(DATA_FILE)
            likes = pd.to_numeric(df_existing['Likes Received (per post)'], errors='coerce').mean()
            comments = pd.to_numeric(df_existing['Comments Received (per post)'], errors='coerce').mean()
            likes = likes if not np.isnan(likes) else 500
            comments = comments if not np.isnan(comments) else 249
            ua_interaction = sm_usage_hrs * anxiety
            if sm_usage_hrs < 2:
                usage_intensity_label = 'Low'
            elif sm_usage_hrs < 5:
                usage_intensity_label = 'Medium'
            else:
                usage_intensity_label = 'High'

            raw_feat = np.array([
                likes, comments,peer,
                SOCIOECONOMIC_MAP[socio], EDU_LEVEL_MAP[education],
                body, sleep_quality,conf, cyber, anxiety,
                AGE_CAT_MAP[age_cat], total_int,
                USAGE_INTENSITY_MAP[usage_intensity_label], ua_interaction
            ]).reshape(1, -1)

            pred        = xgb_model.predict(raw_feat)[0]
            pred_label  = SOCIAL_ANX_MAP[pred]
            cluster_id  = int(kmeans_model.predict(kmeans_scaler.transform(raw_feat))[0])

            entry = pd.DataFrame([{
                'username':           st.session_state.username,
                'Education Level':    EDU_LEVEL_MAP[education],
                'Socioeconomic Status':SOCIOECONOMIC_MAP[socio],
                'Peer Comparison Frequency (1-10)':    peer,
                'Body Image Impact (1-10)':  body,
                'Sleep Quality Impact (1-10)':sleep_quality,
                'Self Confidence Impact (1-10)':    conf,
                'Cyberbullying Experience (1-10)':      cyber,
                'Anxiety Levels (1-10)':      anxiety,
                'Age Category':       AGE_CAT_MAP[age_cat],
                'Total Interaction':  total_int,
                'Usage Intensity':    USAGE_INTENSITY_MAP[usage_intensity_label],
                'Usage-Anxiety':      ua_interaction,
                'Likes Received (per post)':     likes,
                'Comments Received (per post)':  comments,
                'Social Anxiety Category':  pred_label,
                'cluster':            cluster_id,
                'Timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            entry.to_csv(DATA_FILE, mode='a', index=False, header=False)
            st.success(f"Logged! Risk: **{pred_label}**, Cluster: **{cluster_id}**")

def progress_page():
    st.header("📈 Your Progress Over Time")
    df = pd.read_csv(DATA_FILE, parse_dates=['Timestamp'])
    user_df = df[df['username'] == st.session_state.username]

    if user_df.empty:
        st.warning("No entries yet. Log some data first!")
        return

    latest = user_df.sort_values('Timestamp').iloc[-1]
    st.subheader("Latest Entry")
    st.write(f"- **Social Anxiety Category:**{latest['Social Anxiety Category']}")
    st.write(f"- **Cluster:** {latest['Cluster']}")
    st.write(f"- **Anxiety Level:** {latest['Anxiety Levels (1-10)']}")
    st.write(f"- **When:** {latest['Timestamp']}")

    st.subheader("Social Anxiety Over Time")
    user_df['anx_num'] = user_df['Social Anxiety Category'].map(PLOT_MAP)
    fig_anx = px.line(user_df, x='Timestamp', y='anx_num', markers=True,
                      labels={'anx_num':'Social Anxiety (1=Low,2=Medium,3=High)'})
    st.plotly_chart(fig_anx, use_container_width=True)

    st.subheader("Risk Distribution")
    dist = user_df['Social Anxiety Category'].value_counts().reset_index()
    dist.columns = ['Risk','Count']
    fig2 = px.bar(dist, x='Risk', y='Count', text_auto=True)
    st.plotly_chart(fig2, use_container_width=True)

def insights_page():
    st.header("🔍 Insights & Tips")
    df = pd.read_csv(DATA_FILE, parse_dates=['Timestamp'])
    user_df = df[df['username'] == st.session_state.username]

    if user_df.empty:
        st.warning("No data to analyze. Please add entries first.")
        return

    latest = user_df.sort_values('Timestamp').iloc[-1]
    sa_cat   = latest['Social Anxiety Category']
    st.markdown(f"### Current Social Anxiety: **{sa_cat}** (Cluster {latest['Cluster']})")
    if sa_cat == 'High':
        st.warning("Consider grounding exercises, breaks, and social support.")
    elif sa_cat == 'Medium':
        st.info("Try setting SM limits and positive self-talk.")
    else:
        st.success("Great! Maintain healthy habits and support peers.")

    if st.button("Get AI-Powered Recommendations"):
        prompt = (
            "- You are an ai so be friendly, say who are you and then said about on their result.\n"
            f"Here is the user name: {st.session_state.username}"
            "- You are a concise mental health coach,make it funny,make it you are an ai.\n"
            f"- You need to say the User Social Anxiety level: {sa_cat}.\n"
            f"- You need to say the Anxiety level: {latest['Anxiety Levels (1-10)']}.\n"
            "- Provide exactly 3 bullet points of actionable tips based on their result everytime need to be different."
        )
        response = get_openrouter_response(prompt)
        if response:
            st.markdown("#### AI-Powered Recommendations")
            st.markdown(response)

def expert_dashboard():
    st.header("👩‍⚕️ Expert Dashboard")
    df = pd.read_csv(DATA_FILE, parse_dates=['Timestamp'])
    patients = [u for u in df['username'].unique() if u != 'xiaoyi']
    if not patients:
        return st.info("No patient data available yet.")
    patient = st.selectbox("Select a patient", patients)
    patient_df = df[df['username'] == patient]

    if patient_df.empty:
        return st.warning(f"No data found for {patient}.")
    latest = patient_df.sort_values('Timestamp').iloc[-1]

    st.subheader(f"Latest for {patient}")
    st.write(f"- **Social Anxiety:** {latest['Social Anxiety Category']}")
    st.write(f"- **Cluster:** {latest['Cluster']}")
    st.write(f"- **Anxiety Level:** {latest['Anxiety Levels (1-10)']}")
    st.write(f"- **When:** {latest['Timestamp']}")

    st.subheader("Social Anxiety Over Time")
    patient_df['anx_num'] = patient_df['Social Anxiety Category'].map(PLOT_MAP)
    st.plotly_chart(px.line(patient_df, x='Timestamp', y='anx_num', markers=True), use_container_width=True)
    st.subheader("Cluster Distribution")
    dist = patient_df['Cluster'].value_counts().reset_index()
    dist.columns=['Cluster','Count']
    st.plotly_chart(px.bar(dist, x='Cluster', y='Count', text_auto=True), use_container_width=True)

def logout_page():
    st.header("🔒 Log Out")
    if st.button("🚪 Log out"):
        for key in ["logged_in","username"]:
            st.session_state.pop(key, None)
        st.query_params.clear()
        st.success("You’ve been logged out.")
        st.stop()

# ==== Sidebar Navigation & Routing ==== #
st.sidebar.header("Navigation")
if st.session_state.username == "xiaoyi":
    menu = {
        "🏠 Home": home_page,
        "👩‍⚕️ Expert Dashboard": expert_dashboard,
        "🔒 Logout": logout_page
    }
else:
    menu = {
        "🏠 Home": home_page,
        "➕ New Entry": new_entry_page,
        "📈 Progress Dashboard": progress_page,
        "🔍 Insights & Tips": insights_page,
        "🔒 Logout": logout_page
    }

choice = st.sidebar.radio("Go to", list(menu.keys()))
menu[choice]()

# ==== Footer ==== #
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888;'>MindEase © 2025 - Educational Project</div>",
    unsafe_allow_html=True
)