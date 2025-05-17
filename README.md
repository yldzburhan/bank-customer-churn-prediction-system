# Bank Customer Churn Prediction 🏦

This project encompasses a machine learning model developed to predict bank customer churn and an interactive web application that utilizes this model. The project aims to predict the likelihood of churn based on customers' demographic information, account activities, and their relationship with the bank.

[Screenshot or GIF of the Bank Customer Churn Prediction App]

## 🎯 Project Aim

* To identify the main factors influencing customer churn.
* To proactively detect customers at high risk of churning.
* To provide data-driven insights for customer retention strategies.
* To make a trained machine learning model accessible to users via an interactive web interface.

## 📂 Project Structure

```text
bank-customer-churn-prediction/
├── data/                     # Folder containing the datasets
│   ├── churn_modeling.csv    # Original churn dataset (additional source for training)
│   ├── test.csv              # Competition test dataset
│   └── train.csv             # Competition training dataset
├── venv/                     # Virtual environment folder (optional, can be excluded with .gitignore)
├── __pycache__/              # Python cache files (optional, can be excluded with .gitignore)
├── app.py                    # Streamlit web application script
├── churn_model_pipeline.joblib # Trained and saved model pipeline
├── custom_transformers.py    # Custom Scikit-learn transformers and functions
├── models.ipynb              # Jupyter Notebook (data analysis, model development, and training)
├── README.md                 # This file
├── requirements.txt          # Required Python libraries
└── save_pipeline.py          # Script to train and save the model pipeline
```

## ✨ Key Features and Process

The project includes the following main steps:

1.  **Data Loading and Merging:**
    * Data is loaded from `train.csv` and `churn_modeling.csv` (original dataset) and merged to create a comprehensive training set.
2.  **Exploratory Data Analysis (EDA):**
    * Analysis of feature distributions, relationships between features, and their correlation with churn.
    * Key insights are derived through visualizations (histograms, box plots, count plots, etc.). These analyses are detailed in `models.ipynb`.
3.  **Feature Engineering (in `custom_transformers.py` and `models.ipynb`):**
    * **Numerical Feature Transformations:** Features like salary, age, and balance are processed using transformers such as `SalaryRounder`, `AgeRounder`, and `BalanceRounder`. `Nullify` sets 0 values in 'Balance' to NaN.
    * **New Feature Generation:** New and interactive features like `IsActive_by_CreditCard`, `Products_Per_Tenure`, `ZeroBalance`, and `AgeCat` are derived using `FeatureGenerator`.
    * **Vectorization for High-Cardinality Features:** TF-IDF followed by TruncatedSVD is applied to features like `Surname`, `EstimatedSalary`, `CreditScore`, `CustomerId`, and the combined `AllCat` using the custom `Vectorizer` class.
    * **Categorical Feature Encoding:** `MEstimateEncoder` and `CatBoostEncoder` (depending on the pipeline) are used for features like `Geography` and `Gender`.
    * **Removal of Unnecessary Columns:** `FeatureDropper` is used to remove raw processed features or IDs.
4.  **Model Development and Training (in `models.ipynb` and `save_pipeline.py`):**
    * Various classification algorithms were experimented with (Logistic Regression, XGBoost, LightGBM, CatBoost, TensorFlow-based Neural Network).
    * The best-performing model (an XGBoost pipeline is exemplified in this repository) was selected.
    * **Hyperparameter Optimization:** `Optuna` was used to optimize the hyperparameters of the chosen model.
    * **Cross-Validation:** The model's performance and generalization ability were evaluated using `StratifiedKFold`.
    * The best pipeline, including all preprocessing steps, was saved to `churn_model_pipeline.joblib`.
5.  **Streamlit Web Application (`app.py`):**
    * Provides an interactive form for users to input customer information.
    * Loads the saved `churn_model_pipeline.joblib` to make predictions.
    * Displays the churn probability and risk level (Low, Medium, High).
    * Presents some basic EDA graphs and project information in tabs.

## 🛠️ Technologies and Libraries Used

* **Python 3.x** (Specifically 3.9, 3.10, or 3.11 recommended for TensorFlow compatibility)
* **Core Libraries:**
    * `pandas` & `numpy` (Data manipulation and numerical operations)
    * `scikit-learn` (Pipeline creation, preprocessing, modeling, metrics)
    * `matplotlib` & `seaborn` (Data visualization)
* **Modeling Libraries:**
    * `xgboost` (XGBClassifier)
    * `lightgbm` (LGBMClassifier - among experimented models)
    * `catboost` (CatBoostClassifier - among experimented models)
    * `tensorflow` (For the custom `TensorFlower` neural network class - among experimented models)
    * `category-encoders` (CatBoostEncoder, MEstimateEncoder)
* **Other Tools:**
    * `streamlit` (Interactive web application interface)
    * `joblib` (Saving and loading the model pipeline)
    * `optuna` (Hyperparameter optimization)
    * `tqdm` (Progress bars)
* **Development Environment:** Jupyter Notebook, VS Code (or preferred IDE)

## 🚀 Setup and Running the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your_username/bank-customer-churn-prediction.git](https://github.com/your_username/bank-customer-churn-prediction.git)
    cd bank-customer-churn-prediction
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    Python 3.10 or 3.11 is recommended for compatibility with TensorFlow and other libraries.
    ```bash
    # Example: Creating a virtual environment named venv_tf with Python 3.10 (using py launcher on Windows)
    py -3.10 -m venv venv_tf
    .\venv_tf\Scripts\activate

    # Or on macOS/Linux:
    # python3.10 -m venv venv_tf
    # source venv_tf/bin/activate
    ```

3.  **Install Required Libraries:**
    With your virtual environment activated:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train and Save the Model Pipeline (if `churn_model_pipeline.joblib` is not present):**
    This script trains the model using CSV files from the `data/` folder and creates `churn_model_pipeline.joblib`.
    ```bash
    python save_pipeline.py
    ```
    *(Note: For this script to run, `custom_transformers.py` and the `data` folder must be correctly configured.)*

5.  **Run the Streamlit Application:**
    ```bash
    streamlit run app.py
    ```
    The application will open in your default web browser.

## 📊 Streamlit Interface Features

* **Main Page / Single Prediction Tab:**
    * Interactive form to input customer details.
    * Real-time churn probability and risk level prediction based on the inputs.
    * Simple insights and recommendations based on the prediction outcome.
* **General Insights Tab:**
    * EDA graphs from the training data (e.g., relationship of geography, gender, number of products with churn).
* **Model & Project Info Tab:**
    * Summary information about the model used, key performance metrics, and the project process.

## 🔮 Future Enhancements

* Integration of more models (e.g., the best performing CatBoost or TensorFlow model) into the Streamlit interface, offering users a choice of model.
* Integration of explainability methods like SHAP or LIME for predictions.
* Ability to make batch predictions by uploading customer data (e.g., via CSV).
* Further enrichment of the user interface and an increase in visualizations.

## 🎬 Demo Video

[![Bank Customer Churn Prediction App Demo]](https://youtu.be/rMTl9fgsOFQ)

*Watch a 1:30 minute demonstration of the Streamlit application showcasing its features and prediction capabilities. Click the image above or [this link](https://youtu.be/rMTl9fgsOFQ) to view the video on YouTube.*

## 🙏 Contributing

If you would like to contribute to this project, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details (if present).
