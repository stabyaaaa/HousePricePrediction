# Import necessary libraries
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__, static_url_path = '/static')
#this maps type of house 
type_mapping = {'Housing': 0, 'Unit': 1, 'Town': 2}

# Load the model and scaler
model_path = "model/house_price.model"
scaler_path = "model/scaler.pkl"

with open(model_path, "rb") as model_file, open(scaler_path, "rb") as scaler_file:
    house_price_model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Prediction route
@app.route('/housePricePrediction', methods=['POST'])
def predict():
    try:
        # Get input values from the user
        rooms = int(request.form['rooms'])
        type_text = request.form['type'].capitalize()  # Convert to capitalize for consistency
        bed2 = int(request.form['bed2'])
        bathrooms = int(request.form['bathrooms'])
        car = int(request.form['car'])

        # Map the text input to numerical value using the dictionary
        type_numerical = type_mapping.get(type_text, 0)  # Default to 0 if type_text is not found
        print(f"Input values: rooms={rooms}, type={type_text}, bed2={bed2}, bathrooms={bathrooms}, car={car}")

        # Create a sample dataframe
        sample = pd.DataFrame({
            "rooms": [rooms],
            "type": [type_numerical],
            "bed2": [bed2],
            "bathroom": [bathrooms],
            "car": [car]
        })

        # Scale the sample using the same scaler used for X_train and X_set
        scaled_sample = scaler.transform(sample)

        # Use the model to predict the selling price
        predicted_selling_price = house_price_model.predict(scaled_sample)

        # As the y was log-transformed during training, exponent transform the predicted value
        predicted_selling_price = np.exp(predicted_selling_price)
        
        rounded_predicted_selling_price = (round(predicted_selling_price[0]))

        return render_template('index.html', prediction=f'The predicted selling price is ${str(rounded_predicted_selling_price)}')
    except Exception as e:
        return render_template('index.html', prediction=f'Error: {str(e)}')

# Run the Flask application if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)
