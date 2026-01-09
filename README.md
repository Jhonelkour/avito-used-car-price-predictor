# 🚗 Avito Car Price Predictor

A comprehensive machine learning project for scraping, analyzing, and predicting used car prices on the Moroccan market (Avito.ma). This project includes data collection via web scraping, exploratory data analysis, model training with ensemble methods, and an interactive web application for real-time price predictions.

## 🎯 Features

### 🔍 Web Scraping

- **Automated Data Collection**: Selenium-based scraper for Avito.ma car listings
- **Robust Design**: Handles dynamic content and pagination (up to 500 pages)
- **14 Data Fields**: URL, Title, Price, and 11 technical specifications
- **Progressive Saving**: Data saved incrementally to prevent loss

### 📊 Machine Learning

- **Ensemble Models**: Both ensemble and improved model variants
- **Advanced Feature Engineering**: Age calculations, mileage per year, frequency encoding
- **Multiple Algorithms**: Gradient Boosting with hyperparameter tuning
- **Comprehensive Metrics**: R², MAE, RMSE with cross-validation

### 🎨 Interactive Web App

- **Real-time Predictions**: Instant price estimates with confidence intervals
- **Modern Dark UI**: GitHub-inspired professional design
- **Dynamic Filtering**: Brand/model selection based on scraped data
- **Multi-currency Display**: Prices in DH and USD
- **Detailed Insights**: Feature importance and price factors

### 🧹 EDA & Cleaning

- **Exploratory Analysis**: Assessed distributions, relationships, and target behavior
- **Cleaning Completed**: Handled missing values and treated outliers where appropriate
- **Categorical Normalization**: Standardized labels (e.g., fuel, gearbox, brand/model)
- **Model-Ready Data**: Prepared features for encoding/scaling in training

## 📁 Project Structure

```
Avito/
│
├── data/
│   └── avito_final_results.csv     # Scraped car listings dataset
│
├── models/
│   ├── car_price_model_ensemble.pkl    # Ensemble model
│   ├── car_price_model_improved.pkl    # Improved model variant
│   ├── scaler_ensemble.pkl             # Feature scaler (ensemble)
│   ├── scaler_improved.pkl             # Feature scaler (improved)
│   ├── feature_names_ensemble.pkl      # Feature list (ensemble)
│   ├── feature_names_improved.pkl      # Feature list (improved)
│   ├── model_metadata_ensemble.pkl     # Model metrics (ensemble)
│   └── model_metadata_improved.pkl     # Model metrics (improved)
│
├── notebooks/
│   ├── 00_scraping.ipynb          # Web scraping notebook
│   └── 01_analysis.ipynb          # EDA, data analysis & ML
│
├── src/
│   ├── app.py                     # Streamlit web application
│   └── update_metadata.py         # Model metadata utility
│
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Chrome browser (for scraping)

### Installation

1. Clone or download this repository
2. Navigate to the project directory:

```bash
cd Avito
```

3. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

4. Install required packages:

```bash
pip install -r requirements.txt
```

### Running the Application

Run the Streamlit app from the project root:

```bash
streamlit run src/app.py
```

Or from the src directory:

```bash
cd src
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`

## 🎨 Using the Application

### Web Scraper (notebooks/00_scraping.ipynb)

1. Open the scraping notebook in Jupyter
2. Configure scraping parameters (pages to scrape)
3. Run the cells to collect data from Avito.ma
4. Data is saved progressively to `data/avito_final_results.csv`

**Scraped Fields**:

- URL, Title, Price
- Kilométrage (Mileage)
- Première main (First hand)
- Carburant (Fuel type)
- Boîte de vitesse (Gearbox)
- Année-Modèle (Model year)
- Origine (Origin)
- Puissance fiscale (Fiscal power)
- Nombre de portes (Number of doors)
- État (Condition)
- Marque (Brand)
- Modèle (Model)

### Exploratory Data Analysis (EDA) (notebooks/01_analysis.ipynb)

The EDA phase helps you understand data quality, distributions, and key relationships before modeling.

1. Open the analysis notebook in Jupyter
2. Run the cells in the "EDA" section
3. Review plots and notes to guide feature engineering and modeling

**What you'll explore**:
- Target distribution (Price) with optional log-scale view
- Feature distributions: mileage, year, fiscal power, doors
- Missing values overview and imputation decisions
- Correlations (Pearson/Spearman) and multicollinearity checks
- Price relationships vs. year, mileage, brand/model, fuel, gearbox
- Outlier detection (IQR/z-score) and handling approach
- Brand/model frequency and dataset coverage

Outputs are displayed inline in the notebook and summarized to inform the next modeling steps.

Status: EDA and data cleaning have been completed in `notebooks/01_analysis.ipynb`.

### Data Analysis & Model Training (notebooks/01_analysis.ipynb)

1. **Data Exploration**: Statistical analysis and visualizations
2. **Data Cleaning**: Handle missing values and outliers
3. **Feature Engineering**: Create derived features (age, mileage/year)
4. **Model Training**: Train ensemble and improved models
5. **Model Evaluation**: Cross-validation and performance metrics
6. **Model Export**: Save trained models to `models/` directory

### Price Prediction App (src/app.py)

1. **Enter Vehicle Details**:

   - Kilométrage (mileage in km)
   - Année-Modèle (year of manufacture)
   - Puissance fiscale (CV)
   - Nombre de portes
   - Marque and Modèle (dynamic dropdown)
   - Carburant (Diesel, Essence, Hybride, Électrique)
   - Boîte de vitesse (Automatique, Manuelle)
   - Origine (Maroc-Occasion, Dédouanée-Occasion, WW-Maroc, Importée Neuve)
   - Première main (Oui/Non)
   - État (Bon état, Très bon état, Neuf, etc.)
2. **Click "🔮 Predict Price"**: Get instant price estimate
3. **Review Results**:

   - Estimated price in DH and USD
   - Price range (confidence interval ±10%)
   - Vehicle summary
   - Feature importance insights
   - Model performance metrics

## 🔍 Technical Details

### Data Collection

**Technology Stack**:

- `undetected_chromedriver`: Bypass bot detection
- `Selenium WebDriver`: Browser automation
- `Pandas`: Data manipulation and CSV export

**Scraping Strategy**:

- Iterates through up to 500 pages
- Locates car cards by CSS class selectors
- Clicks "Voir plus" (Show more) buttons to expand specifications
- Implements robust error handling and retry logic
- Progressive CSV saving to prevent data loss

### Data Cleaning

- **Missing Values**: Imputed or dropped based on feature relevance
- **Outliers**: Identified via IQR/z-score; capped or filtered as needed
- **Types & Ranges**: Cast numeric fields and validated sensible ranges (price, mileage, year)
- **Categorical Cleanup**: Normalized categories (fuel, gearbox, brand/model) for consistency
- **Deduplication**: Removed duplicates and obvious scraping artifacts
- **Training Prep**: Ensured compatibility with encoding and scaling steps

### Machine Learning

**Feature Engineering**:

- **Numeric Features**: Kilométrage, Age (calculated), Puissance fiscale, Nombre de portes
- **Derived Features**: Mileage per year, Age squared
- **Categorical Features**: Marque, Modèle, Carburant, Boîte de vitesse, Origine, Première main, État
- **Encoding**:
  - Frequency encoding for high-cardinality (Brand/Model)
  - One-hot encoding for low-cardinality features

**Model Architecture**:

- **Algorithm**: Gradient Boosting Regressor
- **Scaling**: StandardScaler for numeric features
- **Cross-Validation**: K-fold for robust evaluation
- **Hyperparameter Tuning**: RandomizedSearchCV
- **Ensemble Approach**: Multiple model variants for comparison

**Performance Metrics**:

- R² Score (Coefficient of Determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- Cross-validation scores

### Web Application

**Framework**: Streamlit with custom CSS
**Theme**: Dark mode (GitHub-inspired)
**Features**:

- Responsive layout with wide mode
- Dynamic brand/model filtering based on dataset
- Real-time predictions with loading animations
- Expandable sections for details
- Error handling and user feedback

## 🛠️ Technologies Used

### Data Science & ML

- **Python 3.8+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning models and preprocessing
- **Matplotlib & Seaborn**: Data visualization

### Web Scraping

- **Selenium**: Browser automation
- **undetected-chromedriver**: Anti-detection for web scraping

### Web Application

- **Streamlit**: Interactive web framework
- **Custom CSS**: Modern UI styling

## 📊 Model Performance

Check the application sidebar for live metrics from the trained models:

- **R² Score**: ~0.85-0.90 (actual performance visible in app)
- **MAE**: Displayed in the application
- **Training Dataset**: 5,000+ Avito.ma car listings
- **Features**: 20+ engineered features

## ⚠️ Disclaimer

This tool provides price **estimates** based on historical data and should be used as a guide only. Actual market prices may vary due to:

- Specific vehicle condition, maintenance history, and documentation
- Market demand fluctuations and seasonal variations
- Negotiation factors and seller urgency
- Geographic location and availability
- Additional features, modifications, or accessories not captured in the model
- Recent market events or economic changes

**Note**: Web scraping should be performed responsibly and in compliance with Avito.ma's terms of service.

## 📝 License

This project is for educational and research purposes only.

## 🤝 Contributing

Contributions are welcome! Please feel free to:

- Report bugs or issues
- Suggest new features or improvements
- Submit pull requests with enhancements
- Share feedback on model performance

## 📧 Contact

For questions, feedback, or collaboration opportunities, please open an issue in the repository.

---

**Built with ❤️ using Python, Selenium, Scikit-learn, and Streamlit**
