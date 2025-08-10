from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Model and Data Loading ---

def preprocess_inputs(df, scaler=None):
    """Preprocesses the input data."""
    df = df.copy()

    # Drop profile_id if it exists
    if 'profile_id' in df.columns:
        df = df.drop('profile_id', axis=1)

    y = None
    if 'pm' in df.columns:
        y = df['pm'].copy()
        X = df.drop('pm', axis=1).copy()
    else:
        X = df.copy()

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X = pd.DataFrame(X_scaled, columns=X.columns)

    if y is not None:
        return X, y, scaler
    else:
        return X, scaler


# Creating a dummy csv file
data = {
    'u_q': np.random.rand(100) * 5 - 2.5,
    'coolant': np.random.rand(100) * 20 + 10,
    'stator_winding': np.random.rand(100) * 50 + 20,
    'u_d': np.random.rand(100) * 5 - 2.5,
    'stator_tooth': np.random.rand(100) * 30 + 15,
    'motor_speed': np.random.rand(100) * 1000,
    'i_d': np.random.rand(100) * 2 - 1,
    'i_q': np.random.rand(100) * 2 - 1,
    'pm': np.random.rand(100) * 80 + 20,
    'stator_yoke': np.random.rand(100) * 40 + 20,
    'ambient': np.random.rand(100) * 30 + 10,
    'torque': np.random.rand(100) * 10,
    'profile_id': np.random.randint(1, 10, 100)
}
dummy_df = pd.DataFrame(data)
dummy_df.to_csv("measures_v2.csv", index=False)


# Load and preprocess data
motor_temp = pd.read_csv("measures_v2.csv")
X_train, y_train, scaler = preprocess_inputs(motor_temp)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)


# --- API Endpoint ---

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for getting motor temperature predictions."""
    try:
        # Get the input data from the request
        input_data = request.get_json()
        input_df = pd.DataFrame([input_data])

        # Preprocess the input data
        X_pred, _ = preprocess_inputs(input_df, scaler=scaler)

        # Make a prediction
        prediction = model.predict(X_pred)

        # Return the prediction as a JSON response
        return jsonify({"predicted_pm_temperature": prediction[0]})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)