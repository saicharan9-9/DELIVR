# DELIVR

DELIVR is a machine learning web application that predicts food delivery time based on distance, weather, traffic, time of day, vehicle type, preparation time, and courier experience.

The application uses a trained Random Forest Regression model integrated with a scikit-learn preprocessing pipeline and provides an interactive user interface built with Streamlit.

## Features

- Food delivery time prediction
- Random Forest Regression
- Scikit-learn preprocessing pipeline
- OneHotEncoder for categorical features
- ColumnTransformer for feature preprocessing
- Interactive Streamlit interface
- Custom dark glassmorphism UI
- Input validation
- Real-time prediction
- Model evaluation using R², MAE, and RMSE
- Streamlit Community Cloud deployment support

## Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- NumPy
- HTML
- CSS
- JavaScript
- Pickle

## Input Features

The model uses seven input features.

| Feature | Type | Description |
|---|---|---|
| `Distance_km` | Numerical | Distance between the restaurant and customer |
| `Weather` | Categorical | Weather condition during delivery |
| `Traffic_Level` | Categorical | Traffic condition on the route |
| `Time_of_Day` | Categorical | Time period of the order |
| `Vehicle_Type` | Categorical | Vehicle used for delivery |
| `Preparation_Time_min` | Numerical | Food preparation and packaging time |
| `Courier_Experience_yrs` | Numerical | Courier's delivery experience |

### Weather

- Clear
- Foggy
- Rainy
- Snowy
- Windy

### Traffic Level

- Low
- Medium
- High

### Time of Day

- Morning
- Afternoon
- Evening
- Night

### Vehicle Type

- Bike
- Car
- Scooter

## Machine Learning

DELIVR uses a Random Forest Regressor for predicting delivery time.

The model is integrated into a scikit-learn Pipeline. Categorical features are processed using OneHotEncoder and ColumnTransformer before being passed to the Random Forest model.

### Prediction Pipeline

```text
Input Data
    |
    v
Feature Preprocessing
    |
    v
OneHotEncoder + ColumnTransformer
    |
    v
Random Forest Regressor
    |
    v
Predicted Delivery Time
```

The complete trained pipeline is stored as a pickle file and loaded directly by the Streamlit application.

## Model Performance

The trained model achieved approximately:

| Metric | Score |
|---|---:|
| R² Score | 0.77 |
| MAE | 6.78 minutes |

Model performance can vary depending on the training and testing split.

## Dataset

The project uses the `Food_Delivery_Times.csv` dataset.

The dataset contains information related to:

- Delivery distance
- Weather conditions
- Traffic level
- Time of day
- Vehicle type
- Food preparation time
- Courier experience
- Delivery time

The target variable is:

```text
Delivery_Time_min
```

## Project Structure

```text
DELIVR/
│
├── app.py
├── food_delivery_model.pkl
├── requirements.txt
├── Food_Delivery_Times.csv
├── Food_delivery_prediction.ipynb
├── README.md
│
└── assets/
    ├── style.css
    └── script.js
```

### File Description

| File | Description |
|---|---|
| `app.py` | Streamlit application and prediction interface |
| `food_delivery_model.pkl` | Trained scikit-learn machine learning pipeline |
| `requirements.txt` | Python dependencies |
| `Food_Delivery_Times.csv` | Dataset used for model training |
| `Food_delivery_prediction.ipynb` | Model training and evaluation notebook |
| `assets/style.css` | Custom application styling |
| `assets/script.js` | Frontend interactions |
| `README.md` | Project documentation |

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/saicharan9-9/DELIVR.git
cd DELIVR
```

### 2. Create a Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate the environment:

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Locally

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

## Deployment

DELIVR can be deployed using Streamlit Community Cloud.

### Deployment Steps

1. Push the project repository to GitHub.
2. Sign in to Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select the `DELIVR` repository.
5. Select the `main` branch.
6. Set `app.py` as the main file.
7. Click **Deploy**.

The `requirements.txt` file contains the dependencies required to run the application.

## Requirements

The project uses the following main dependencies:

```text
streamlit
pandas
numpy
scikit-learn==1.9.0
```

The scikit-learn version is pinned to `1.9.0` to match the environment used when the trained model pipeline was serialized.

## Model Usage

The application loads the trained pipeline:

```text
food_delivery_model.pkl
```

Since preprocessing is included inside the pipeline, the application can pass the raw user input directly to the model without manually encoding categorical features.

## Development

The machine learning workflow is documented in:

```text
Food_delivery_prediction.ipynb
```

The notebook contains the data preparation, preprocessing, model training, evaluation, and experimentation used to develop the prediction model.

## License

This project is intended for educational and project demonstration purposes.
