import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Aqua Sight AI",
    page_icon="🌊",
    layout="wide"
)

@st.cache_resource
def build_model():
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'Nitrogen': np.random.uniform(0, 100, n),
        'Phosphorus': np.random.uniform(0, 100, n),
        'pH': np.random.uniform(6, 9, n),
        'Temperature': np.random.uniform(10, 35, n)
    })
    
    conditions = (
        (df['Nitrogen'] > 40) & 
        (df['Phosphorus'] > 40) & 
        (df['Temperature'] > 20)
    )
    df['Risk'] = np.where(conditions, 1, 0)
    
    X = df[['Nitrogen', 'Phosphorus', 'pH', 'Temperature']]
    y = df['Risk']
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    return model, df

model, dataset = build_model()

try:
    st.sidebar.image("logo.png", width=100)
except:
    pass

st.sidebar.title("🌊 Aqua Sight AI")
st.sidebar.markdown("**Configure Sensor Parameters:**")

n_in = st.sidebar.slider("Nitrogen (mg/L)", 0, 100, 45)
p_in = st.sidebar.slider("Phosphorus (mg/L)", 0, 100, 45)
t_in = st.sidebar.slider("Temperature (°C)", 10, 35, 25)
ph_in = st.sidebar.slider("pH Level", 0.0, 14.0, 7.2)

st.sidebar.markdown("---")
st.sidebar.info("Adjust sliders to simulate different water conditions.")

st.title("Aqua Sight AI: Eutrophication Monitor")
st.info("""
**Definition:** Eutrophication is a process where water bodies become overly enriched with minerals and nutrients, 
leading to excessive algae growth (algal blooms). This depletes oxygen and kills aquatic life.
""")

prediction = model.predict([[n_in, p_in, ph_in, t_in]])
probs = model.predict_proba([[n_in, p_in, ph_in, t_in]])
confidence = probs[0][prediction[0]] * 100

st.divider()
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Analysis Result")
    
    if prediction[0] == 1:
        st.error(f"🚨 CRITICAL ALERT: High Eutrophication Risk Detected")
        st.metric("AI Confidence", f"{confidence:.2f}%")
        
        st.markdown("### ⚠️ Recommended Actions:")
        st.warning("""
        1. **Immediate Aeration:** Deploy oxygenators to prevent fish death.
        2. **Nutrient Control:** Halt agricultural runoff in upstream fields.
        3. **Algaecide Application:** Apply eco-friendly treatments immediately.
        """)
        
    else:
        st.success(f"✅ STATUS: Water is Safe")
        st.metric("AI Confidence", f"{confidence:.2f}%")
        
        st.markdown("### 👍 Recommended Actions:")
        st.markdown("""
        1. **Routine Monitoring:** Continue sensor logging every 24 hours.
        2. **Preservation:** Maintain vegetative buffers around the lake.
        """)

with col2:
    st.subheader("Sensor Readings")
    st.write(f"**Nitrogen:** {n_in} mg/L")
    st.write(f"**Phosphorus:** {p_in} mg/L")
    st.write(f"**Temperature:** {t_in} °C")
    st.write(f"**pH:** {ph_in}")

st.divider()

st.subheader("📊 Training Data Sample")
st.markdown("This is a view of the **Synthetic Dataset** used to train the Random Forest logic.")
st.dataframe(dataset.head(10), use_container_width=True)

st.divider()
st.subheader("🛰️ Future Scope: Satellite Integration")
st.markdown("""
To scale **Aqua Sight AI** for city-wide monitoring, we propose the following upgrades:
* **Sentinel-2 Satellite Connection:** Using APIs to fetch real-time satellite imagery of the lake.
* **NDVI Analysis:** Using Computer Vision to calculate the "Greenness Index" of the water from space.
* **IoT Hardware:** Connecting this dashboard to physical Raspberry Pi sensors floating in the water.
""")