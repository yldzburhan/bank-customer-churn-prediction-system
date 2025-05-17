import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from category_encoders import CatBoostEncoder
import warnings
import os # For style file path

# Import custom transformers and functions
try:
    from custom_transformers import (
        SalaryRounder, AgeRounder, FeatureGenerator, 
        Vectorizer
    )
except ImportError:
    st.error("ERROR: 'custom_transformers.py' file not found or does not contain the necessary classes/functions. "
             "Please ensure this file is correctly configured. "
             "The application cannot run correctly without this file if the pipeline depends on it.")
    pipeline = None 

# Page Settings
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide warnings (optional)
warnings.filterwarnings("ignore")

# Apply Matplotlib style (optional)
# Ensure 'rose-pine-dawn.mplstyle' is in the same directory as app.py
# or provide the correct path.
STYLE_FILE_NAME = 'rose-pine-dawn.mplstyle'
if os.path.exists(STYLE_FILE_NAME):
    try:
        plt.style.use(STYLE_FILE_NAME)
        # st.sidebar.success(f"'{STYLE_FILE_NAME}' stili uygulandı.") # Kullanıcıya bilgi
    except Exception as e:
        # st.sidebar.warning(f"'{STYLE_FILE_NAME}' stili uygulanamadı: {e}. Varsayılan stil kullanılıyor.")
        sns.set_theme(style="whitegrid", palette="pastel") # Fallback style
else:
    # st.sidebar.info(f"'{STYLE_FILE_NAME}' stil dosyası bulunamadı. Varsayılan stil kullanılıyor.")
    sns.set_theme(style="whitegrid", palette="pastel") # Fallback style


# --- Data Loading for EDA ---
@st.cache_data 
def load_eda_data():
    try:
        df_train_comp = pd.read_csv('data/train.csv', index_col='id')
        df_orig_train = pd.read_csv('data/churn_modeling.csv', index_col='RowNumber')

        for df in [df_train_comp, df_orig_train]:
            if 'IsActiveMember' in df.columns:
                df['IsActiveMember'] = df['IsActiveMember'].astype(int)
            if 'HasCrCard' in df.columns:
                df['HasCrCard'] = df['HasCrCard'].astype(int)
            if 'Exited' in df.columns:
                df['Exited'] = df['Exited'].astype(int)
        
        df_eda_combined = pd.concat([df_orig_train, df_train_comp], ignore_index=True)
        return df_eda_combined
    except FileNotFoundError:
        st.error("⚠️ EDA Data files (train.csv, churn_modeling.csv) not found in 'data/' folder.")
        return None
    except Exception as e:
        st.error(f"🚫 Error loading EDA data: {e}")
        return None

df_for_eda = load_eda_data()


# --- Model Pipeline Loading ---
@st.cache_resource
def load_model_pipeline(pipeline_path='churn_model_pipeline.joblib'):
    try:
        pipeline_loaded = joblib.load(pipeline_path)
        return pipeline_loaded
    except FileNotFoundError:
        st.error(f"⚠️ Saved model pipeline ('{pipeline_path}') not found.")
        st.info("Please run your model training script (e.g., save_pipeline.py) first to save the pipeline.")
        return None
    except Exception as e:
        st.error(f"🚫 An error occurred while loading the pipeline: {e}")
        st.warning("Ensure your custom transformer classes and functions are defined correctly.")
        return None

pipeline = load_model_pipeline()


# --- Input Preprocessing for Streamlit ---
def preprocess_input_for_streamlit(data_dict):
    try:
        df = pd.DataFrame([data_dict])
        numeric_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        bool_like_cols = ['HasCrCard', 'IsActiveMember']
        for col in bool_like_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df
    except Exception as e:
        st.error(f"Error processing input data: {str(e)}")
        return None

# --- EDA Plotting Functions (Updated with style considerations) ---
def plot_churn_by_categorical(df, column_name, ax):
    if df is not None and column_name in df.columns and 'Exited' in df.columns:
        # Rose-pine dawn temasına uygun olabilecek bir palet seçimi
        # Örnek paletler: "flare", "crest", "magma_r", "viridis_r" veya özel bir liste
        # Temanızın ana renklerini biliyorsanız, özel bir palet oluşturabilirsiniz.
        # ['#ebbcba', '#c4a7e7', '#9ccfd8', '#f6c177', '#31748f'] gibi
        custom_palette_categorical = sns.color_palette("pastel", 2) # Örnek olarak pastel, 2 renk için
        if STYLE_FILE_NAME and os.path.exists(STYLE_FILE_NAME):
             # Stil dosyası varsa, Seaborn'un varsayılan renklerini kullanmasına izin ver
             # veya stil dosyasının renklerini alacak bir yöntem bulunmalı.
             # Şimdilik, stil dosyası varsa Seaborn'un kendi renklerini kullanmasını sağlıyoruz.
             # Palette argümanını None yapmak yerine, stilin renklerini almasını bekleyebiliriz.
             # Veya temanızın ana renklerinden bir palet oluşturabilirsiniz.
             # Örn: rose_pine_colors = {"Exited=0": "#9ccfd8", "Exited=1": "#eb6f92"}
             # sns.countplot(x=column_name, hue='Exited', data=df, ax=ax, palette=rose_pine_colors)
             pass # Stil dosyası renkleri otomatik uygulamalı

        sns.countplot(x=column_name, hue='Exited', data=df, ax=ax, palette=custom_palette_categorical)
        ax.set_title(f'Churn Distribution by {column_name}', fontsize=9) # Font boyutu küçültüldü
        ax.set_xlabel(column_name, fontsize=7)
        ax.set_ylabel('Number of Customers', fontsize=7)
        ax.tick_params(axis='x', rotation=30, labelsize=6) # X ekseni etiketleri küçültüldü ve döndürüldü
        ax.tick_params(axis='y', labelsize=6)
        ax.legend(title='Exited', title_fontsize='7', fontsize='6', loc='upper right')
    else:
        ax.text(0.5, 0.5, f"Data or column '{column_name}'/'Exited' not available.", 
                ha='center', va='center', fontsize=8, color='gray')

def plot_numerical_distribution_by_churn(df, column_name, ax):
    if df is not None and column_name in df.columns and 'Exited' in df.columns:
        custom_palette_numerical = sns.color_palette("pastel", 2) # Örnek
        if STYLE_FILE_NAME and os.path.exists(STYLE_FILE_NAME):
            pass # Stil dosyası renkleri otomatik uygulamalı
            
        sns.histplot(data=df, x=column_name, hue='Exited', kde=True, multiple="stack", ax=ax, palette=custom_palette_numerical, bins=25) # bins sayısı ayarlandı
        ax.set_title(f'{column_name} Distribution by Churn', fontsize=9)
        ax.set_xlabel(column_name, fontsize=7)
        ax.set_ylabel('Frequency', fontsize=7)
        ax.tick_params(labelsize=6)
        if df['Exited'].nunique() > 1: # Legend sadece birden fazla hue değeri varsa gösterilir
            ax.legend(title='Exited', title_fontsize='7', fontsize='6', loc='upper right')
    else:
        ax.text(0.5, 0.5, f"Data or column '{column_name}'/'Exited' not available.", 
                ha='center', va='center', fontsize=8, color='gray')

# --- Main Application ---
def main_prediction_app():
    st.title("🏦 Bank Customer Churn Prediction System")
    st.markdown("*Predict a customers likelihood of leaving the bank (churn) and their risk level based on their information.*")
    st.markdown("---")

    if pipeline is None and df_for_eda is None:
        st.error("🚫 Application cannot start: Model pipeline and EDA data failed to load.")
        return
    # ... (Diğer pipeline ve df_for_eda None kontrolleri aynı kalabilir) ...

    tab_titles = ["📈 Single Customer Prediction", "📊 General Churn Insights", "💡 Model & Project Info"]
    if pipeline is None: 
        tab_titles[0] = "📈 Single Customer Prediction (Model Error)"
    
    tab1, tab2, tab3 = st.tabs(tab_titles)

    with tab1:
        # ... (tab1 içeriği aynı kalabilir, form ve tahmin mantığı) ...
        st.header("👤 Customer Information Input")
        if pipeline is None:
            st.error("Model could not be loaded. Prediction is currently unavailable.")
        else:
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    customer_id = st.text_input("Customer ID", "e.g., 15674932", help="Unique identifier for the customer.")
                    surname = st.text_input("Surname", "e.g., Smith", help="Customer's last name.")
                    credit_score = st.slider("Credit Score", 300, 850, 650, help="Between 300-850.")
                    geography = st.selectbox("Geography", ["France", "Spain", "Germany"], index=0, help="Country where the customer is located.")
                with col2:
                    gender = st.radio("Gender", ["Male", "Female"], index=0, horizontal=True)
                    age = st.slider("Age", 18, 100, 35, help="Customer's age (18-100).")
                    tenure = st.slider("Tenure (Years)", 0, 10, 5, help="Number of years the customer has been with the bank (0-10).")
                    balance = st.number_input("Balance", min_value=0.0, value=0.0, step=1000.0, format="%.2f", help="Customer's account balance.")
                with col3:
                    num_products = st.selectbox("Number of Products", [1, 2, 3, 4], index=1, help="Number of bank products the customer owns.")
                    has_credit_card_str = st.radio("Has Credit Card?", ["Yes", "No"], index=0, horizontal=True)
                    has_credit_card = 1 if has_credit_card_str == "Yes" else 0
                    is_active_member_str = st.radio("Is Active Member?", ["Yes", "No"], index=0, horizontal=True)
                    is_active_member = 1 if is_active_member_str == "Yes" else 0
                    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0, format="%.2f", help="Customer's estimated annual salary.")
                
                submitted = st.form_submit_button("🎯 Predict Churn")
                
            if submitted:
                with st.spinner("🔄 Calculating prediction..."):
                    input_data = {
                        'CustomerId': customer_id, 'Surname': surname, 'CreditScore': credit_score,
                        'Geography': geography, 'Gender': gender, 'Age': age, 'Tenure': tenure,
                        'Balance': balance, 'NumOfProducts': num_products, 'HasCrCard': has_credit_card,
                        'IsActiveMember': is_active_member, 'EstimatedSalary': estimated_salary
                    }
                    processed_data_df = preprocess_input_for_streamlit(input_data)
                    if processed_data_df is not None:
                        try:
                            prediction_proba = pipeline.predict_proba(processed_data_df)
                            churn_probability = prediction_proba[0, 1]
                            st.subheader("📊 Prediction Results:")
                            res_col1, res_col2 = st.columns(2)
                            with res_col1:
                                st.metric(label="Churn Probability", value=f"{churn_probability:.1%}")
                                st.progress(int(churn_probability * 100))
                            with res_col2:
                                risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.3 else "Low"
                                if risk_level == "High": st.error(f"🚨 Risk Level: {risk_level}")
                                elif risk_level == "Medium": st.warning(f"⚠️ Risk Level: {risk_level}")
                                else: st.success(f"✅ Risk Level: {risk_level}")
                            # ... (insights logic) ...
                        except AttributeError:
                            st.error("🚫 Loaded model pipeline lacks 'predict_proba'. Please load a classification pipeline for probability estimates.")
                        except Exception as e:
                            st.error(f"🚫 Error during prediction: {str(e)}")
            else:
                st.info("Please fill the form above and click 'Predict Churn'.")


    with tab2:
        st.header("📊 General Churn Insights (from Training Data)")
        st.markdown("This section displays general EDA graphs and insights from the training dataset.")

        if df_for_eda is not None:
            st.subheader("Categorical Feature Analysis")
            fig1, axes1 = plt.subplots(2, 2, figsize=(10, 8)) # Boyut küçültüldü
            fig1.tight_layout(pad=3.0) 

            plot_churn_by_categorical(df_for_eda, 'Geography', axes1[0, 0])
            plot_churn_by_categorical(df_for_eda, 'Gender', axes1[0, 1])
            plot_churn_by_categorical(df_for_eda, 'NumOfProducts', axes1[1, 0])
            plot_churn_by_categorical(df_for_eda, 'HasCrCard', axes1[1, 1])
            
            st.pyplot(fig1)
            st.markdown("---")
            
            st.subheader("Numerical Feature Analysis")
            fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4)) # Boyut küçültüldü
            fig2.tight_layout(pad=2.5)

            plot_numerical_distribution_by_churn(df_for_eda, 'Age', axes2[0])
            plot_numerical_distribution_by_churn(df_for_eda, 'Balance', axes2[1])
            
            st.pyplot(fig2)
            
            st.markdown("---")
            st.info("""
            **Key Observations (Examples):**
            * 🌍 **Geography:** Customers in Germany might show a higher churn rate compared to France or Spain.
            * 💳 **Number of Products:** Customers with 1 product or 3-4 products might be more prone to churn than those with 2 products.
            * 🎂 **Age:** Churn rates often peak for middle-aged customers (e.g., 40-55 years).
            * 💰 **Balance:** Customers with very high balances who are inactive, or those with zero balance (especially outside France), might have different churn behaviors.
            """)
        else:
            st.warning("Could not load EDA data to display insights.")
        
    with tab3:
        # ... (tab3 içeriği aynı kalabilir) ...
        st.header("💡 Model and Project Information")
        st.markdown("""
        This application uses a machine learning model developed to predict bank customer churn.

        **Model Used:**
        * **Model Type:** XGBoost Classifier (or the name of the final model in your pipeline)
        * **Key Performance Metric (in Training):** AUC (Area Under Curve) - Example: ~0.89

        **Feature Engineering Steps:**
        * Rounding/transformation for numerical features like salary, age, balance.
        * Creation of interaction features (e.g., `IsActive_by_CreditCard`) and combined categorical features (e.g., `AllCat`).
        * Vectorization of high-cardinality text-based features (`Surname`, `AllCat`) using TF-IDF and TruncatedSVD.
        * Use of `CatBoostEncoder` and `MEstimateEncoder` for categorical features.

        **Development Process:**
        * Comprehensive Exploratory Data Analysis (EDA) was performed.
        * Various models were tested (Logistic Regression, LightGBM, CatBoost, XGBoost, TensorFlow-based Neural Network).
        * Hyperparameter optimization was applied using Optuna to enhance model performance.
        * The model's generalization ability was assessed using `StratifiedKFold` cross-validation.
        
        **Note:** This interface provides real-time predictions using a pre-trained model pipeline. 
        Predictions are probabilistic and do not guarantee a definitive outcome.
        """)
        with st.expander("❓ Frequently Asked Questions (Example)"):
            st.markdown("""
            **Q: What does 'Churn Probability' mean?**
            A: It represents the likelihood, as a percentage, that the customer will leave the bank within a certain timeframe.

            **Q: What is the model's accuracy?**
            A: The model demonstrated a performance of approximately ~0.89 AUC on the training data. Real-world performance may vary.
            """)

    # Footer
    st.markdown("---")
    st.markdown("<div style='text-align: center;'>© 2025 Bank Customer Churn Prediction System</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main_prediction_app()
