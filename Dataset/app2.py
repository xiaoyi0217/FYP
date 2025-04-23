import streamlit as st
import pickle
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from datetime import datetime
import requests
import os
from PIL import Image

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

local_css("style.css")

def get_ollama_response(server_url, model_name, prompt, max_tokens=256, temperature=0.7):
    url = server_url + "/v1/completions"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {"Content-Type": "application/json"}
    try:
        resp = requests.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        return resp.json()['choices'][0]['text'].strip()
    except Exception as e:
        st.error(f"Ollama API error: {e}")
        return None

@st.cache_resource
def load_models():
    with open('production_kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('kmeans_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('xgb_model.pkl', 'rb') as f:
        xgb = pickle.load(f)
    return kmeans, scaler, xgb

kmeans_model, kmeans_scaler, xgb_model = load_models()

# ==== Encoding Mappings ==== #
URBAN_RURAL_MAP = {'Rural': 0, 'Urban': 1}
GENDER_MAP = {'Female': 0, 'Male': 1}
SM_FREQ_MAP = {'Daily': 0, 'Monthly': 1, 'Rarely': 2, 'Weekly': 3}
EDU_LEVEL_MAP = {"Bachelor's": 0, "Doctorate": 1, "High School": 2, "Master's": 3}
COUNTRY_MAP = {
    'Brunei': 0, 'Cambodia': 1, 'East Timor': 2, 'Indonesia': 3, 'Laos': 4,
    'Malaysia': 5, 'Myanmar': 6, 'Philippines': 7, 'Singapore': 8,
    'Thailand': 9, 'Vietnam': 10
}
SOCIOECONOMIC_MAP = {'High': 0, 'Low': 1, 'Middle': 2}
STATE_MAP = {
    'Bali': 0, 'Bandar Seri Begawan': 1, 'Bangkok': 2, 'Battambang': 3, 'Baucau': 4,
    'Cebu': 5, 'Chiang Mai': 6, 'Da Nang': 7, 'Davao': 8, 'Dili': 9,
    'Hanoi': 10, 'Ho Chi Minh City': 11, 'Jakarta': 12, 'Johor Bahru': 13,
    'Kuala Belait': 14, 'Kuala Lumpur': 15, 'Luang Prabang': 16,
    'Mandalay': 17, 'Manila': 18, 'Naypyidaw': 19, 'Pakse': 20,
    'Penang': 21, 'Phnom Penh': 22, 'Phuket': 23, 'Siem Reap': 24,
    'Singapore': 25, 'Suai': 26, 'Surabaya': 27, 'Tutong': 28,
    'Vientiane': 29, 'Yangon': 30
}
AGE_CAT_MAP = {'Adult': 0, 'Senior': 1, 'Youth': 2}
USAGE_INTENSITY_MAP = {'Low': 1, 'Medium': 2, 'High': 0}
PLATFORM_MAP = {'Facebook': 0, 'Instagram': 1, 'TikTok': 2, 'Twitter': 3, 'WeChat': 4}
ANXIETY_LABELS = {0: 'High', 1: 'Low', 2: 'Medium'}

COUNTRY_STATE_MAPPING = {
    'Brunei': ['Bandar Seri Begawan', 'Kuala Belait', 'Tutong'],
    'Cambodia': ['Phnom Penh', 'Siem Reap', 'Battambang'],
    'East Timor': ['Dili', 'Baucau', 'Suai'],
    'Indonesia': ['Jakarta', 'Surabaya', 'Bali'],
    'Laos': ['Vientiane', 'Luang Prabang', 'Pakse'],
    'Malaysia': ['Kuala Lumpur', 'Penang', 'Johor Bahru'],
    'Myanmar': ['Naypyidaw', 'Mandalay', 'Yangon'],
    'Philippines': ['Manila', 'Cebu', 'Davao'],
    'Singapore': ['Singapore'],
    'Thailand': ['Bangkok', 'Chiang Mai', 'Phuket'],
    'Vietnam': ['Hanoi', 'Ho Chi Minh City', 'Da Nang']
}

# ==== Data Storage Setup ==== #
# The CSV header does not include Likes Received and Comments Received as these are computed averages.
DATA_FILE = "user_data.csv"
if not os.path.exists(DATA_FILE):
    cols = [
        'Urban/Rural', 'Gender', 'Frequency of SM Use', 'Education Level', 'Country',
        'Socioeconomic Status', 'State', 'Peer Comparison', 'Body Image Impact',
        'Sleep Quality Impact', 'Self Confidence', 'Cyberbullying', 'Anxiety Level',
        'Age Category', 'Total Interaction', 'Usage Intensity', 'Usage-Anxiety',
        'Most Used SM Platform', 'Likes Received', 'Comments Received',
        'predicted_class', 'cluster', 'timestamp'
    ]
    pd.DataFrame(columns=cols).to_csv(DATA_FILE, index=False)

# ==== Header ==== #
col1, col2 = st.columns([1, 4])
with col1:
    try:
        st.image("mindease_logo.png", width=100)
    except:
        st.write("🧠")
with col2:
    st.markdown("""
        <div class='header'>
            <h1>MindEase Social Anxiety Tracker</h1>
            <p>Track your social anxiety over time with AI-driven insights</p>
        </div>
    """, unsafe_allow_html=True)

# ==== Sidebar Navigation ==== #
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "➕ New Entry", "📈 Progress Dashboard", "🔍 Insights & Tips"])

# ==== Home Page ==== #
if page == "🏠 Home":
    st.title("Welcome to MindEase")
    st.write("""
        MindEase combines the power of AI-driven classification and clustering to help you understand and manage your social anxiety.
        Log daily metrics, view your progress over time, and receive personalized guidance to improve your well-being.
    """)
    st.markdown("Explore the sidebar to begin your journey!")

# ==== New Entry Page ==== #
elif page == "➕ New Entry":
    st.header("Log Your Daily Metrics")
    with st.form("entry_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            urban_rural = st.selectbox("Area Type", list(URBAN_RURAL_MAP.keys()))
            country     = st.selectbox("Country", list(COUNTRY_MAP.keys()))
            state       = st.selectbox("State", COUNTRY_STATE_MAPPING[country])
            gender      = st.selectbox("Gender", list(GENDER_MAP.keys()))
            sm_freq     = st.selectbox("Social Media Frequency", list(SM_FREQ_MAP.keys()))
            education   = st.selectbox("Education Level", list(EDU_LEVEL_MAP.keys()))
            socio       = st.selectbox("Socioeconomic Status", list(SOCIOECONOMIC_MAP.keys()))
            platform    = st.selectbox("Most Used SM Platform", list(PLATFORM_MAP.keys()))
        with c2:
            peer           = st.slider("Peer Comparison (1–10)", 1, 10, 5)
            body           = st.slider("Body Image Impact (1–10)", 1, 10, 5)
            sleep_quality  = st.slider("Sleep Quality Impact (1–10)", 1, 10, 5)
            conf           = st.slider("Self Confidence Impact (1–10)", 1, 10, 5)
            cyber          = st.slider("Cyberbullying Experience (1–10)", 1, 10, 1)
            anxiety        = st.slider("Anxiety Levels (1–10)", 1, 10, 5)
            age_cat        = st.selectbox("Age Category", list(AGE_CAT_MAP.keys()))
            total_int      = st.number_input("Total Social Interactions", min_value=0, max_value=1785, value=10)
            sm_usage_hours = st.number_input("Daily SM Usage (hrs)", min_value=0.0, max_value=24.0, value=3.0)
        submitted = st.form_submit_button("Submit")

        def safe_mean(series, default):
            val = pd.to_numeric(series, errors='coerce').mean()
            return default if pd.isna(val) else val

        if submitted:
            df_existing = pd.read_csv(DATA_FILE)
            likes = safe_mean(df_existing['Likes Received'], 500) if ('Likes Received' in df_existing.columns and not df_existing.empty) else 500
            comments = safe_mean(df_existing['Comments Received'], 249) if ('Comments Received' in df_existing.columns and not df_existing.empty) else 249

            ua_interaction = sm_usage_hours * anxiety  # Usage-Anxiety Interaction
            if sm_usage_hours < 2:
                usage_intensity = 'Low'
            elif sm_usage_hours < 5:
                usage_intensity = 'Medium'
            else:
                usage_intensity = 'High'

            usage_intensity = usage_intensity.strip()
            usage_intensity = USAGE_INTENSITY_MAP.get(usage_intensity, 0)  # Fallback to 'Low' if invalid

            # Build the feature array for prediction (15 features).
            raw = np.array([
                SM_FREQ_MAP[sm_freq],       # Frequency of SM Use
                likes,                      # Average Likes Received
                comments,                   # Average Comments Received
                SOCIOECONOMIC_MAP[socio],   # Socioeconomic Status
                EDU_LEVEL_MAP[education],   # Education Level
                STATE_MAP[state],           # State (mapped)
                body,                       # Body Image Impact
                sleep_quality,              # Sleep Quality Impact
                conf,                       # Self Confidence Impact
                cyber,                      # Cyberbullying Experience
                anxiety,                    # Anxiety Levels
                AGE_CAT_MAP[age_cat],       # Age Category
                total_int,                  # Total Social Interaction
                usage_intensity,            # Usage Intensity
                ua_interaction              # Usage-Anxiety Interaction (added feature)
            ])
            features = raw.reshape(1, -1)

            # Generate predictions from the loaded models.
            pred = xgb_model.predict(features)[0]
            pred_label = ANXIETY_LABELS.get(pred, "Unknown")
            cluster = int(kmeans_model.predict(kmeans_scaler.transform(features))[0])

            # Save the new entry.
            entry = pd.DataFrame([{  
                'Urban/Rural': urban_rural,
                'Gender': gender,
                'Frequency of SM Use': sm_freq,
                'Education Level': education,
                'Country': country,
                'Socioeconomic Status': socio,
                'State': state,
                'Peer Comparison': peer,
                'Body Image Impact': body,
                'Sleep Quality Impact': sleep_quality,
                'Self Confidence': conf,
                'Cyberbullying': cyber,
                'Anxiety Level': anxiety,
                'Age Category': age_cat,
                'Total Interaction': total_int,
                'Usage Intensity': usage_intensity,
                'Usage-Anxiety': ua_interaction,
                'Most Used SM Platform': platform,
                'Likes Received': likes,
                'Comments Received': comments,
                'predicted_class': pred_label,
                'cluster': cluster,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }])
            # Append the new entry to the CSV.
            entry.to_csv(DATA_FILE, mode='a', header=False, index=False)
            st.success(f"Entry logged! Risk: **{pred_label}**, Cluster: **{cluster}**.")

# ==== Progress Dashboard Page ==== #
elif page == "📈 Progress Dashboard":
    st.header("Your Progress Over Time")
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    if df.empty:
        st.warning("No entries logged yet. Please add new entries to view your progress.")
    else:
        # Show Latest Entry Summary.
        latest = df.sort_values('timestamp').iloc[-1]
        st.subheader("Latest Entry Summary")
        st.write(f"**Risk Category:** {latest['predicted_class']}")
        st.write(f"**Cluster:** {latest['cluster']}")
        st.write(f"**Anxiety Level:** {latest['Anxiety Level']}")
        st.write(f"**Timestamp:** {latest['timestamp']}")

        st.subheader("Anxiety Levels Over Time")
        fig1 = px.line(df, x='timestamp', y='Anxiety Level', markers=True)
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Risk Category Distribution")
        counts = df['predicted_class'].value_counts().reset_index()
        counts.columns = ['Risk Category', 'Count']
        fig2 = px.bar(counts, x='Risk Category', y='Count', text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

        # --- PCA Visualization ---
        # Create a dataframe for PCA using the 15 prediction features.
        df_pca = df.copy()
        if 'Likes Received' not in df_pca.columns:
            df_pca['Likes Received'] = 0
        if 'Comments Received' not in df_pca.columns:
            df_pca['Comments Received'] = 0

        # Map the categorical features as in prediction.
        df_pca['Frequency of SM Use'] = df_pca['Frequency of SM Use'].map(SM_FREQ_MAP)
        df_pca['Socioeconomic Status'] = df_pca['Socioeconomic Status'].map(SOCIOECONOMIC_MAP)
        df_pca['Education Level'] = df_pca['Education Level'].map(EDU_LEVEL_MAP)
        df_pca['State'] = df_pca['State'].map(STATE_MAP)
        df_pca['Age Category'] = df_pca['Age Category'].map(AGE_CAT_MAP)
        df_pca['Usage Intensity'] = df_pca['Usage Intensity'].map(USAGE_INTENSITY_MAP)

        # Define the 15 features used for the scaler.
        features_pca = [
            'Frequency of SM Use',
            'Likes Received',
            'Comments Received',
            'Socioeconomic Status',
            'Education Level',
            'State',
            'Body Image Impact',
            'Sleep Quality Impact',
            'Self Confidence',
            'Cyberbullying',
            'Anxiety Level',
            'Age Category',
            'Total Interaction',
            'Usage Intensity',
            'Usage-Anxiety'
        ]
        X = df_pca[features_pca].fillna(0)
        X_scaled = kmeans_scaler.transform(X)

        if X_scaled.shape[0] < 2:
            st.warning("Insufficient data to perform PCA visualization. Please log more entries.")
        else:
            pca = PCA(n_components=2)
            pcs = pca.fit_transform(X_scaled)
            df_pca['PC1'], df_pca['PC2'] = pcs[:, 0], pcs[:, 1]
    
            st.subheader("Cluster Visualization (PCA)")
            df_pca['cluster'] = df_pca['cluster'].astype(str)
            fig3 = px.scatter(df_pca, x='PC1', y='PC2', color='cluster')
            st.plotly_chart(fig3, use_container_width=True)

# ==== Insights & Tips Page ==== #
elif page == "🔍 Insights & Tips":
    st.header("Personalized Insights & Tips")
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    if df.empty:
        st.warning("No data to analyze. Please add entries to receive insights.")
    else:
        latest = df.iloc[-1]
        risk = latest['predicted_class'].lower()
        colors = {'high': 'red', 'medium': 'orange', 'low': 'green'}
        color = colors.get(risk, 'green')
        st.markdown(f"""
            <div class='risk-card {color}'>
                <h3>{latest['predicted_class']} Risk</h3>
                <p>Cluster: {latest['cluster']}</p>
            </div>
        """, unsafe_allow_html=True)
        if latest['predicted_class'] == 'High':
            st.warning("Consider taking frequent breaks, practicing grounding techniques, and seeking support.")
        elif latest['predicted_class'] == 'Medium':
            st.info("Try setting limits on social media, offline activities, and positive self-talk.")
        else:
            st.success("Maintain healthy habits and support others.")
        # Provide AI-powered recommendations on demand
        if st.button("Get AI-Powered Recommendations"):
            prompt = (
                "- You are a concise mental health coach.\n"
                "- You are a friendly and funny assistant.  \n"
                "— Only output the final, concise actionable tips.  \n"
                "— Do NOT include any internal reasoning.  \n\n"
                f"User risk level: {latest['predicted_class']}.\n"
                f"Anxiety level: {latest['Anxiety Level']}.\n"
                "- Output exactly 3 bullet points, no explanations."
            )
            server_url = "http://127.0.0.1:11434"
            model_name = "deepseek-r1:8b"
            response = get_ollama_response(server_url, model_name, prompt)
            if response:
                st.markdown("### AI-Powered Recommendations")
                st.markdown(response)

# ==== Footer ==== #
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color: #888;'>MindEase © 2025 - Educational Project</div>",
    unsafe_allow_html=True
)

# To run, execute:
# streamlit run app.py）


print(df_pca[features_pca].dtypes)
print(df_pca[features_pca].isna().sum())