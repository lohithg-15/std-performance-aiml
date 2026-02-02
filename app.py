from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
with open("model/student_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    study_hours = float(request.form["study_hours"])
    attendance = float(request.form["attendance"])
    sleep_hours = float(request.form["sleep_hours"])
    previous_marks = float(request.form["previous_marks"])

    input_data = np.array([[study_hours, attendance, sleep_hours, previous_marks]])
    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        result=f"Predicted Final Score: {round(prediction, 2)}"
    )

if __name__ == "__main__":
    app.run(debug=True)
