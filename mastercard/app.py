import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the pickled model
with open('linear_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the input values from the form
        input_data = [float(x) for x in request.form.values()]
        input_df = pd.DataFrame([input_data], columns=['Open', 'High', 'Low'])

        # Make predictions
        prediction = model.predict(input_df)[0]

        # Render the prediction result
        return render_template('index.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)