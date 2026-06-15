# AI-Driven Vehicle Carbon Footprint Predictor

## Project Overview

This repository hosts an advanced machine learning application designed to predict vehicle CO2 emissions (measured in grams per kilometer) based on mechanical and fuel consumption characteristics. Developed to align with the United Nations Sustainable Development Goal 13 (Climate Action), this tool provides precise estimations of vehicle environmental footprints, facilitating environmental compliance assessment and consumer transparency.

The frontend is built with Streamlit, styled with an immersive dark glassmorphic user interface, and powered by a high-performance predictive model in the backend.

## Predictive Engine

The core prediction framework is constructed utilizing Extreme Gradient Boosting (XGBoost) regression, achieving high-fidelity predictive accuracy for this domain.

- Algorithm: XGBoost Regressor (Extreme Gradient Boosting)
- Accuracy: >99% R-squared (R2) score on test splits
- Core Features Utilized:
  - Car Brand (Make)
  - Vehicle Class
  - Transmission Type
  - Fuel Type (Regular/Premium Gasoline, Diesel, Ethanol, Natural Gas)
  - Engine Size (Liters)
  - Cylinders count
  - Combined Fuel Consumption (L/100km)

Categorical variables are preprocessed and aligned dynamically using serialized LabelEncoders.

## Technology Stack

- Core Programming Language: Python 3.9+
- Frontend User Interface: Streamlit
- Scientific Computing & Machine Learning: Pandas, NumPy, Scikit-Learn, XGBoost
- Model Serialization: Joblib

## Installation and Deployment

Follow these instructions to run the application in a local development environment.

### 1. Clone the Repository

Clone the project repository and navigate to the project directory:

```bash
git clone https://github.com/OlanIsm/Carbon-Emission-Ai-Prediction.git
cd Carbon-Emission-Ai-Prediction
```

### 2. Configure Virtual Environment

Create and activate a virtual environment to manage dependencies:

- On Windows:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- On macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 3. Install Dependencies

Install all package requirements listed in the requirements file:

```bash
pip install -r requirements.txt
```

### 4. Run the Application

Start the Streamlit application server:

```bash
streamlit run app.py
```

The application will be accessible via your default web browser at `http://localhost:8501`.

## Contributors

Developed by Group 4 as part of the Artificial Intelligence Course curriculum.
