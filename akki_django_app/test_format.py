import math
import random

def generate_data(n):
    data = []
    for _ in range(n):
        x = random.uniform(0, 1)
        y = math.sin(x) + random.gauss(0, 0.1)
        data.append((x, y))
    return data

def preprocess_data(data):
    processed_data = []
    for x, y in data:
        if x > 0.5:
            y = y * 2
        processed_data.append((x, y))
    return processed_data

def fit_model(data):
    model_params = {}
    for x, y in data:
        model_params[x] = y
    return model_params

def evaluate_model(model_params, test_data):
    errors = []
    for x, y_true in test_data:
        if x in model_params:
            y_pred = model_params[x]
            error = abs(y_true - y_pred)
            errors.append(error)
    mean_error = sum(errors) / len(errors) if errors else 0
    return mean_error

def main():
    # Generate training and testing data
    train_data = generate_data(800)
    test_data = generate_data(200)

    # Preprocess data
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Fit the model
    model_params = fit_model(train_data)

    # Evaluate the model
    mean_error = evaluate_model(model_params, test_data)

    print(f"Mean error on test data: {mean_error}")

if __name__ == "__main__":
    main()
