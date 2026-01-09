import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="Avito Car Price Predictor - Improved",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Dark Theme (GitHub-style)
st.markdown("""
    <style>
    /* Root colors */
    :root {
        --bg-primary: #0d1117;
        --bg-secondary: #161b22;
        --bg-tertiary: #21262d;
        --text-primary: #c9d1d9;
        --text-secondary: #8b949e;
        --accent: #58a6ff;
        --accent-orange: #fb8500;
        --accent-purple: #79c0ff;
    }
    
    /* Main app background */
    [data-testid="stAppViewContainer"] {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        padding: 2rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #30363d;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        background-color: #0d1117;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #c9d1d9 !important;
        text-shadow: none !important;
    }
    
    /* Text */
    p, span, div {
        color: #c9d1d9;
    }
    
    /* Subheader */
    .stSubheader {
        color: #c9d1d9 !important;
        background: #161b22 !important;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #58a6ff;
        margin-bottom: 1.5rem;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput, .stTextInput {
        background-color: #161b22;
    }
    
    input[type="number"], input[type="text"], select {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
    }
    
    input[type="number"]:focus, input[type="text"]:focus, select:focus {
        border-color: #58a6ff !important;
        box-shadow: 0 0 0 3px rgba(88, 166, 255, 0.1) !important;
    }
    
    /* Selectbox styling */
    [data-testid="stSelectbox"] {
        background-color: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #fb8500 0%, #ffb703 100%);
        color: #000000 !important; /* force black text */
        font-weight: bold;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 15px rgba(251, 133, 0, 0.3);
        transition: all 0.3s ease;
        font-size: 1.1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(251, 133, 0, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Prediction box */
    .prediction-box {
        padding: 2.5rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
        color: #ffffff;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 32px rgba(31, 110, 251, 0.4);
        border: 1px solid #30363d;
    }
    
    .prediction-box h1 {
        color: #ffffff !important;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        padding: 1.5rem;
        border-radius: 12px;
        background: #161b22;
        color: #c9d1d9;
        margin: 1rem 0;
        border-left: 4px solid #58a6ff;
        border: 1px solid #30363d;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        border-left: 4px solid #fb8500;
        border-color: #30363d;
        box-shadow: 0 6px 20px rgba(88, 166, 255, 0.2);
        transform: translateY(-2px);
    }
    
    .info-box h4 {
        margin: 0 0 1rem 0;
        color: #58a6ff;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .info-box p {
        margin: 0.5rem 0;
        font-size: 0.95rem;
        color: #c9d1d9;
    }
    
    .info-box b {
        color: #79c0ff;
    }
    
    /* Divider */
    hr {
        border-color: #30363d !important;
        margin: 2rem 0 !important;
    }
    
    /* Metrics */
    [data-testid="metric-container"] {
        background: #161b22;
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid #30363d;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="metric-container"] label {
        color: #8b949e;
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 10px;
        background-color: #161b22 !important;
        border-left: 4px solid #58a6ff !important;
        color: #c9d1d9 !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    
    .stAlert > div {
        color: #c9d1d9 !important;
    }
    
    /* Info message */
    .stInfo {
        background-color: #0d1117 !important;
        border-left: 4px solid #58a6ff !important;
    }
    
    /* Column containers */
    .stColumn {
        background-color: transparent;
    }
    
    /* Markdown */
    [data-testid="stMarkdownContainer"] {
        color: #c9d1d9;
    }
    
    /* Footer */
    footer {
        display: none;
    }
    
    /* Code blocks */
    code {
        background-color: #161b22;
        color: #79c0ff;
        padding: 2px 6px;
        border-radius: 4px;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0d1117;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }
    </style>
""", unsafe_allow_html=True)

# Define the Ensemble Model class (must be defined before loading pickle)
class EnsembleModel:
    def __init__(self, xgb_model, lgbm_model, catboost_model, gbr_model):
        self.xgb_model = xgb_model
        self.lgbm_model = lgbm_model
        self.catboost_model = catboost_model
        self.gbr_model = gbr_model
    
    def predict(self, X):
        pred_xgb = self.xgb_model.predict(X)
        pred_lgbm = self.lgbm_model.predict(X)
        pred_catboost = self.catboost_model.predict(X)
        pred_gbr = self.gbr_model.predict(X)
        
        # Average predictions
        return (pred_xgb + pred_lgbm + pred_catboost + pred_gbr) / 4

# Load model and preprocessing objects
@st.cache_resource
def load_models():
    try:
        # Get the directory of this script and construct paths relative to project root
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        models_dir = os.path.join(project_root, 'models')
        
        with open(os.path.join(models_dir, 'car_price_model_ensemble.pkl'), 'rb') as f:
            model = pickle.load(f)
        with open(os.path.join(models_dir, 'scaler_ensemble.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(models_dir, 'feature_names_ensemble.pkl'), 'rb') as f:
            feature_names = pickle.load(f)
        with open(os.path.join(models_dir, 'model_metadata_ensemble.pkl'), 'rb') as f:
            model_metadata = pickle.load(f)
        return model, scaler, feature_names, model_metadata
    except FileNotFoundError as e:
        st.error(f"⚠️ Model files not found: {e}")
        return None, None, None, None

# Load dataset for brand and model mapping
@st.cache_resource
def load_dataset():
    try:
        # Get the directory of this script and construct path to data file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        csv_path = os.path.join(project_root, 'data', 'avito_final_results.csv')
        
        df = pd.read_csv(csv_path)
        df = df[df['Marque'] != 'autre']  # Remove 'autre' brand
        # Get unique brands and their models
        brand_model_mapping = {}
        for marque in df['Marque'].unique():
            if pd.notna(marque):
                models = sorted(df[df['Marque'] == marque]['Modèle'].dropna().unique().tolist())
                brand_model_mapping[marque] = models
        return brand_model_mapping
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return {}

model, scaler, feature_names, model_metadata = load_models()
brand_model_mapping = load_dataset()

# Header
st.title("🚗 Avito Car Price Predictor - Improved")
st.markdown("### Predict the price of your used car in Morocco (Ensemble Model)")
st.markdown("---")

# Sidebar - Model Information
with st.sidebar:
    st.header("📊 Model Information")
    if model_metadata:
        st.metric("Model Type", "Ensemble")
        st.metric("R² Score", "87.53%", help="Model explains 87.53% of price variance")
        st.metric("MAE", "22,281 DH", help="Mean Absolute Error")
        st.metric("RMSE", "33,803 DH", help="Root Mean Squared Error")
        st.info(f"🗓️ Training Date: {model_metadata.get('training_date', 'December 2025')}")
        
        st.markdown("---")
        st.markdown("### 🎯 Ensemble Architecture")
        st.markdown("""
        **Combines 4 Models:**
        1. XGBoost (87.22% R²)
        2. LightGBM (87.18% R²)
        3. CatBoost (87.07% R²)
        4. Gradient Boosting (86.26% R²)
        
        **Averaging:** Equal weight predictions
        """)
    
    st.markdown("---")
    st.markdown("### 📖 How to use")
    st.markdown("""
    1. Fill in car details
    2. Click 'Predict Price'
    3. Get instant estimate
    """)

if model is None:
    st.stop()

# Main content - Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("🔧 Vehicle Specifications")
    
    # Numeric inputs
    kilometrage = st.number_input(
        "Kilométrage (km)", 
        min_value=0, 
        max_value=400000, 
        value=50000,
        step=1000,
        help="Total distance traveled by the vehicle"
    )
    
    year = st.number_input(
        "Year (Année-Modèle)", 
        min_value=1990, 
        max_value=datetime.now().year, 
        value=2020,
        help="Year of manufacture"
    )
    
    puissance_fiscale = st.number_input(
        "Puissance fiscale (CV)", 
        min_value=1, 
        max_value=15, 
        value=6,
        help="Fiscal horsepower"
    )
    
    nombre_portes = st.selectbox(
        "Nombre de portes",
        options=[3, 5],
        index=1
    )

with col2:
    st.subheader("🏷️ Vehicle Details")
    
    # Brand selection
    marque = st.selectbox(
        "Marque (Brand)", 
        options=sorted(brand_model_mapping.keys()),
        help="Select the car brand from the available list"
    )
    
    # Model selection based on selected brand
    if marque and marque in brand_model_mapping:
        available_models = brand_model_mapping[marque]
        modele = st.selectbox(
            "Modèle (Model)", 
            options=available_models,
            help="Select the model for the chosen brand"
        )
    else:
        modele = "N/A"
    
    # Categorical inputs
    boite_vitesses = st.selectbox(
        "Boite de vitesses",
        options=["Manuelle", "Automatique", "Semi-automatique"],
        index=0
    )
    
    type_carburant = st.selectbox(
        "Type de carburant",
        options=["Diesel", "Essence", "Hybride", "Electrique"],
        index=0
    )
    
    origine = st.selectbox(
        "Origine",
        options=["Maroc", "Europe", "unknown"],
        index=0
    )
    
    premiere_main = st.selectbox(
        "Première main",
        options=["Oui", "Non"],
        index=1
    )
    
    etat = st.selectbox(
        "État",
        options=["Excellent", "Bon", "Moyen", "Mauvais"],
        index=1
    )

# Calculate derived features
age = datetime.now().year - year
mileage_per_year = kilometrage / (age + 1) if age >= 0 else 0
age_squared = age ** 2

# Prediction button
st.markdown("---")
predict_button = st.button("🎯 Predict Price", use_container_width=True)

if predict_button:
    with st.spinner("Calculating price..."):
        try:
            # Create base dataframe with all features initialized to 0
            input_data = pd.DataFrame(0, index=[0], columns=feature_names)
            
            # Set numeric features
            input_data['Kilométrage'] = kilometrage
            input_data['Age'] = age
            input_data['Puissance fiscale'] = puissance_fiscale
            input_data['Nombre de portes'] = nombre_portes
            input_data['mileage_per_year'] = mileage_per_year
            input_data['age_squared'] = age_squared
            input_data['Year'] = year
            
            # Calculate additional engineered features for improved model
            if 'power_age_ratio' in input_data.columns:
                input_data['power_age_ratio'] = puissance_fiscale / (age + 1)
            
            # Luxury brand indicator
            luxury_brands = ['Mercedes-Benz', 'BMW', 'Audi', 'Porsche', 'Jaguar', 'Land Rover', 'Lexus']
            if 'is_luxury' in input_data.columns:
                input_data['is_luxury'] = 1 if marque in luxury_brands else 0
            
            # Popular model indicator
            if 'is_popular_model' in input_data.columns:
                input_data['is_popular_model'] = 0
            
            # Mileage categories
            if 'mileage_category_low' in input_data.columns:
                input_data['mileage_category_low'] = 1 if kilometrage < 50000 else 0
            if 'mileage_category_medium' in input_data.columns:
                input_data['mileage_category_medium'] = 1 if 50000 <= kilometrage < 100000 else 0
            
            # Interaction features
            if 'year_power' in input_data.columns:
                input_data['year_power'] = year * puissance_fiscale
            if 'age_mileage_interaction' in input_data.columns:
                input_data['age_mileage_interaction'] = age * kilometrage / 10000
            
            # Age bins
            if 'age_0_3' in input_data.columns:
                input_data['age_0_3'] = 1 if age <= 3 else 0
            if 'age_4_7' in input_data.columns:
                input_data['age_4_7'] = 1 if 3 < age <= 7 else 0
            if 'age_8_plus' in input_data.columns:
                input_data['age_8_plus'] = 1 if age > 7 else 0
            
            # Target encoding features
            if 'brand_target_enc' in input_data.columns:
                if 'brand_encodings' in model_metadata and marque in model_metadata['brand_encodings']:
                    input_data['brand_target_enc'] = model_metadata['brand_encodings'][marque]
                else:
                    input_data['brand_target_enc'] = model_metadata.get('global_mean_price', 280644)
            
            if 'model_target_enc' in input_data.columns:
                model_key = f"{marque}_{modele}"
                if 'model_encodings' in model_metadata and model_key in model_metadata['model_encodings']:
                    input_data['model_target_enc'] = model_metadata['model_encodings'][model_key]
                else:
                    input_data['model_target_enc'] = model_metadata.get('global_mean_price', 280644)
            
            # Frequency encoding features
            if 'brand_freq' in input_data.columns:
                if 'brand_frequencies' in model_metadata and marque in model_metadata['brand_frequencies']:
                    input_data['brand_freq'] = model_metadata['brand_frequencies'][marque]
                else:
                    input_data['brand_freq'] = 0.01
            
            if 'model_freq' in input_data.columns:
                model_key = f"{marque}_{modele}"
                if 'model_frequencies' in model_metadata and model_key in model_metadata['model_frequencies']:
                    input_data['model_freq'] = model_metadata['model_frequencies'][model_key]
                else:
                    input_data['model_freq'] = 0.005
            
            # One-hot encoded features
            # Boite de vitesses
            if boite_vitesses == "Manuelle":
                if 'Boite de vitesses_Manuelle' in input_data.columns:
                    input_data['Boite de vitesses_Manuelle'] = 1
            elif boite_vitesses == "Semi-automatique":
                if 'Boite de vitesses_Semi-automatique' in input_data.columns:
                    input_data['Boite de vitesses_Semi-automatique'] = 1
            
            # Type de carburant
            if type_carburant == "Essence":
                if 'Type de carburant_Essence' in input_data.columns:
                    input_data['Type de carburant_Essence'] = 1
            elif type_carburant == "Hybride":
                if 'Type de carburant_Hybride' in input_data.columns:
                    input_data['Type de carburant_Hybride'] = 1
            elif type_carburant == "Electrique":
                if 'Type de carburant_Electrique' in input_data.columns:
                    input_data['Type de carburant_Electrique'] = 1
            
            # Origine
            if origine == "Maroc":
                if 'Origine_Maroc' in input_data.columns:
                    input_data['Origine_Maroc'] = 1
            elif origine == "unknown":
                if 'Origine_unknown' in input_data.columns:
                    input_data['Origine_unknown'] = 1
            
            # Première main
            if premiere_main == "Oui":
                if 'Première main_Oui' in input_data.columns:
                    input_data['Première main_Oui'] = 1
            
            # État
            if etat == "Bon":
                if 'État_Bon' in input_data.columns:
                    input_data['État_Bon'] = 1
            elif etat == "Excellent":
                if 'État_Excellent' in input_data.columns:
                    input_data['État_Excellent'] = 1
            elif etat == "Moyen":
                if 'État_Moyen' in input_data.columns:
                    input_data['État_Moyen'] = 1
            elif etat == "Mauvais":
                if 'État_Mauvais' in input_data.columns:
                    input_data['État_Mauvais'] = 1
            
            # Scale numeric features
            numeric_features_to_scale = ['Kilométrage', 'Age', 'Puissance fiscale', 'Nombre de portes', 
                                        'mileage_per_year', 'age_squared']
            
            # Add additional numeric features if they exist
            additional_numeric = ['power_age_ratio', 'year_power', 'age_mileage_interaction',
                                'brand_target_enc', 'model_target_enc', 'brand_freq', 'model_freq']
            
            for feat in additional_numeric:
                if feat in input_data.columns:
                    numeric_features_to_scale.append(feat)
            
            # Only scale features that exist
            features_to_scale = [f for f in numeric_features_to_scale if f in input_data.columns]
            input_data[features_to_scale] = scaler.transform(input_data[features_to_scale])
            
            # Make prediction
            prediction = model.predict(input_data)[0]
            
            # Display results
            st.markdown("---")
            st.markdown(f"""
                <div class="prediction-box">
                    <h1>💰 Estimated Price</h1>
                    <h1 style="font-size: 3rem; margin: 1rem 0; color: #FFD700;">{prediction:,.0f} DH</h1>
                    <p style="font-size: 1.2rem;">≈ ${prediction/10:.0f} USD</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display input summary
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                    <div class="info-box">
                        <h4>📋 Vehicle Info</h4>
                        <p><b>Brand:</b> {}</p>
                        <p><b>Model:</b> {}</p>
                        <p><b>Year:</b> {}</p>
                        <p><b>Age:</b> {} years</p>
                    </div>
                """.format(marque, modele, year, age), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                    <div class="info-box">
                        <h4>🔧 Specifications</h4>
                        <p><b>Mileage:</b> {:,} km</p>
                        <p><b>Power:</b> {} CV</p>
                        <p><b>Doors:</b> {}</p>
                        <p><b>Fuel:</b> {}</p>
                    </div>
                """.format(kilometrage, puissance_fiscale, nombre_portes, type_carburant), 
                unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                    <div class="info-box">
                        <h4>✨ Condition</h4>
                        <p><b>State:</b> {}</p>
                        <p><b>Gearbox:</b> {}</p>
                        <p><b>Origin:</b> {}</p>
                        <p><b>First Hand:</b> {}</p>
                    </div>
                """.format(etat, boite_vitesses, origine, premiere_main), unsafe_allow_html=True)
            
            # Price range estimate
            st.markdown("---")
            st.info(f"""
                💡 **Price Range Estimate:** {prediction*0.9:,.0f} DH - {prediction*1.1:,.0f} DH
                
                This estimate is based on the ensemble model's average error (MAE: 22,281 DH). 
                The ensemble model achieves 87.53% accuracy by combining XGBoost, LightGBM, CatBoost, 
                and Gradient Boosting predictions.
                
                **Note:** Actual market prices may vary based on additional factors like:
                - Specific model variant and trim level
                - Detailed service history and maintenance records
                - Additional equipment and options
                - Local market conditions and demand
                - Seller motivation and negotiation
            """)
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.info("Please ensure all fields are filled correctly and try again.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🚗 Avito Car Price Predictor - Improved Ensemble Model</p>
        <p style='font-size: 0.9rem;'>Model trained on Moroccan used car data | December 2025</p>
        <p style='font-size: 0.85rem;'>Ensemble: XGBoost + LightGBM + CatBoost + Gradient Boosting</p>
    </div>
""", unsafe_allow_html=True)
