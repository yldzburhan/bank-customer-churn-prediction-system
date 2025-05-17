# Bank Customer Churn Prediction

This project implements a machine learning pipeline to predict bank customer churn probability using various features like credit score, balance, and customer demographics.

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run Jupyter Notebook:
```bash
jupyter notebook
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Project Structure

- `custom_transformers.py`: Custom sklearn transformers for feature engineering
- `app.py`: Streamlit web application for model inference
- `save_pipeline.py`: Script to train and save the model pipeline
- `requirements.txt`: Project dependencies
- `notebooks/`: Jupyter notebooks for model development and analysis

## Features

- Credit score analysis
- Customer demographics processing
- Balance and product usage patterns
- Text feature vectorization (TF-IDF + SVD)
- Custom feature engineering
- Interactive web interface for predictions

## Model Pipeline

The prediction pipeline includes:
1. Data preprocessing
2. Feature engineering
3. Text vectorization
4. Categorical encoding
5. XGBoost classifier

## Usage

1. Start the Streamlit app
2. Enter customer information in the form
3. Click "Predict" to get churn probability
4. Review insights and recommendations

## Development

To modify the model:
1. Update transformers in `custom_transformers.py`
2. Modify the pipeline in `save_pipeline.py`
3. Retrain the model by running `save_pipeline.py`
4. Test changes through the Streamlit interface

## Requirements

- Python 3.8+
- See `requirements.txt` for full list of dependencies 