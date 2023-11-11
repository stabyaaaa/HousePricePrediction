import numpy as np

def calculate_accuracy(actual_values, predicted_values):
    # Sample accuracy calculation (replace this with your actual accuracy calculation)
    accuracy = np.mean(np.abs(actual_values - predicted_values) / actual_values) * 100
    return accuracy

# Sample actual values for testing purposes
actual_values = np.random.rand(8175)

# Sample predicted values for testing purposes
predicted_values = np.random.rand(8175)

# Calculate accuracy
accuracy = calculate_accuracy(actual_values, predicted_values)

print(f'Accuracy: {accuracy}%')
