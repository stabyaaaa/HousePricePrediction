# Import necessary libraries
import pickle  # For loading the trained model and scaler
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
from flask import Flask, request, render_template  # For deployement

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


# Run the Flask application if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=80)  # Start the web application with debugging enabled on port 80