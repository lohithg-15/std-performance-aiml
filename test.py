import pickle
import numpy as np

# Load trained model
with open("model/student_model.pkl", "rb") as f:
    model = pickle.load(f)

# New student data:
# study_hours, attendance, sleep_hours, previous_marks
new_student = np.array([[6, 85, 7, 70]])

prediction = model.predict(new_student)

print("Predicted Final Score:", round(prediction[0], 2))
