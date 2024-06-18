from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from sklearn.exceptions import NotFittedError

# initializing the Flask application to variable app
app = Flask('__name__')

# routing GET method to the root path or "/" path.
# this will be the default route of the website
@app.route("/", methods=['GET'])
def home():
    # sending the index.html page as the response for the GET request
    return render_template('index.html')

# routing POST method to the path "/predict"
# the "/predict" is called when we submit the form in the index.html
@app.route("/predict", methods=['POST'])
def predict():
    variable = request.form.to_dict()  # converting the form request data to a dictionary

    # getting all the values needed for the prediction and converting them to integer value
    Age = int(variable['age'])
    Sex = int(variable['gender'])
    ChestPainType = int(variable['chestPain'])
    RestingBP = int(variable['bloodPressure'])
    Cholesterol = int(variable['cholesterol'])
    FastingBS = int(variable['bloodSugar'])
    RestingECG = int(variable['electrocardio'])
    MaxHR = int(variable['heartRate'])
    ExerciseAngina = int(variable['angine'])
    Oldpeak = float(variable['st'])
    ST_Slope = int(variable['slope'])

    # forming an input array for the prediction model
    input_array = np.array([Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope])

    try:
        # predicting the output
        predict = model.predict(input_array.reshape(1, -1))
    except NotFittedError as e:
        print("Model is not fitted yet. Please check the model loading process.")
        return jsonify({"error": str(e)})

    if predict == 1:  # checking if the prediction was 1, i.e., patient has heart risk
        # returning the output.html with the risk.svg as the image to show
        return render_template('output.html', image='risk.svg')

    # returning output.html with safe.svg as the image
    return render_template("output.html", image='safe.svg')

if __name__ == '__main__':
    # loading the Machine Learning model from the pickle file
    with open('svm_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Ensure the model is fitted
    try:
        print("Checking if the model is fitted by calling predict on dummy data...")
        model.predict(np.zeros((1, 11)))  # Check if the model can handle a dummy input
    except NotFittedError:
        print("The loaded model is not fitted. Exiting...")
        exit()

    print("Model loaded and verified as fitted. Starting the application...")
    # starting the application and listening on the port
    app.run(port=8000)
