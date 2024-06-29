# # Import necessary libraries
# import numpy as np
# from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, r2_score, mean_absolute_error

# # Load the dataset (Iris dataset for classification)
# from sklearn.datasets import load_iris

# data = load_iris()

# print(data)

# X = data.data
# y = data.target

# print("X = ", X)
# print("y = ", y)

# # Split the dataset into training, validation, and test sets
# X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

# print("X_temp = ", X_temp)
# print("X_test = ", X_test)
# print("X_train = ", X_train)
# print("X_val = ", X_val)
# # Initialize the model
# model = RandomForestClassifier(random_state=42)

# # Fit the model to the training data
# model.fit(X_train, y_train)
# print("Model = ", model)
# print("X_train = ", X_train)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv('Data/training_data.csv')

default_symptoms = [
"itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
"chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
"vomiting", "burning_micturition", "spotting_ urination", "fatigue", "weight_gain",
"anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
"lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
"sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion", "headache",
"yellowish_skin", "dark_urine", "nausea", "loss_of_appetite", "pain_behind_the_eyes",
"back_pain", "constipation", "abdominal_pain", "diarrhoea", "mild_fever", "yellow_urine",
"yellowing_of_eyes", "acute_liver_failure", "fluid_overload", "swelling_of_stomach",
"swelled_lymph_nodes", "malaise", "blurred_and_distorted_vision", "phlegm",
"throat_irritation", "redness_of_eyes", "sinus_pressure", "runny_nose", "congestion",
"chest_pain", "weakness_in_limbs", "fast_heart_rate", "pain_during_bowel_movements",
"pain_in_anal_region", "bloody_stool", "irritation_in_anus", "neck_pain", "dizziness",
"cramps", "bruising", "obesity", "swollen_legs", "swollen_blood_vessels",
"puffy_face_and_eyes", "enlarged_thyroid", "brittle_nails", "swollen_extremeties",
"excessive_hunger", "extra_marital_contacts", "drying_and_tingling_lips", "slurred_speech",
"knee_pain", "hip_joint_pain", "muscle_weakness", "stiff_neck", "swelling_joints",
"movement_stiffness", "spinning_movements", "loss_of_balance", "unsteadiness",
"weakness_of_one_body_side", "loss_of_smell", "bladder_discomfort", "foul_smell_of urine",
"continuous_feel_of_urine", "passage_of_gases", "internal_itching", "toxic_look_(typhos)",
"depression", "irritability", "muscle_pain", "altered_sensorium", "red_spots_over_body",
"belly_pain", "abnormal_menstruation", "dischromic _patches", "watering_from_eyes",
"increased_appetite", "polyuria", "family_history", "mucoid_sputum", "rusty_sputum",
"lack_of_concentration", "visual_disturbances", "receiving_blood_transfusion",
"receiving_unsterile_injections", "coma", "stomach_bleeding", "distention_of_abdomen",
"history_of_alcohol_consumption", "fluid_overload", "blood_in_sputum", "prominent_veins_on_calf",
"palpitations", "painful_walking", "pus_filled_pimples", "blackheads", "scurring",
"skin_peeling", "silver_like_dusting", "small_dents_in_nails", "inflammatory_nails",
"blister", "red_sore_around_nose", "yellow_crust_ooze", "prognosis"
]

X = data[default_symptoms]
y = data['Disease']

#Testing the model
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Retrain the model with the best hyperparameters on the combined training and validation set
model = RandomForestClassifier(**best_params, random_state=42)
model.fit(np.vstack((X_train, X_val)), np.hstack((y_train, y_val)))

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-Validation Accuracy Scores: ", [score.round(4) for score in cv_scores])
print(f"Mean Cross-Validation Accuracy: {cv_scores.mean():.4f}")
