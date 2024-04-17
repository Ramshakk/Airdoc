import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

disease_data = pd.read_csv("C:/Users/React/OneDrive/Desktop/airrdog/uploads/disease.csv")

X_disease = disease_data.drop(columns=["prognosis"])
y_disease = disease_data["prognosis"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_disease)

X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(X_disease, y_encoded, test_size=0.2, random_state=42)

disease_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_disease.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

disease_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

disease_model.fit(X_train_disease, y_train_disease, epochs=1000, validation_data=(X_test_disease, y_test_disease))

disease_model.save("predict_disease.tflite")

# Convert the disease prediction TensorFlow model to TensorFlow Lite
disease_converter = tf.lite.TFLiteConverter.from_keras_model(disease_model)
disease_tflite_model = disease_converter.convert()

# Save the disease prediction TensorFlow Lite model to a file
with open("C:/Users/React/Downloads/predict_disease.tflite", "wb") as f:
    f.write(disease_tflite_model)

# Load the disease prediction TensorFlow Lite model
disease_interpreter = tf.lite.Interpreter(model_path="C:/Users/React/Downloads/predict_disease.tflite")
disease_interpreter.allocate_tensors()

# Function to make disease predictions using TensorFlow Lite
def predict_disease_tflite(symptoms):
    all_symptoms = X_disease.columns.tolist()
    input_details = disease_interpreter.get_input_details()
    output_details = disease_interpreter.get_output_details()

    # Create an empty array for the user input
    user_input = np.zeros(input_details[0]['shape'], dtype=np.float32)

    # Populate the user input based on the symptoms
    for symptom in symptoms:
        if symptom in all_symptoms:
            # Set the corresponding feature to 1
            feature_index = all_symptoms.index(symptom)
            user_input[0, feature_index] = 1.0

    disease_interpreter.set_tensor(input_details[0]['index'], user_input)
    disease_interpreter.invoke()
    prediction = disease_interpreter.get_tensor(output_details[0]['index'])

    return np.argmax(prediction)


import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path='C:/Users/React/Downloads/model.tflite')
interpreter.allocate_tensors()

# Get the input and output details of the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the input shape based on the model's input details
input_shape = input_details[0]['shape']


# Define the symptom and disease dictionaries as before
symptoms = {
    "itching": 0,
    "skin rash": 0,
    "nodal skin eruptions": 0,
    "continuous sneezing": 0,
    "shivering": 0,
    "chills": 0,
    "joint pain": 0,
    "stomach pain": 0,
    "acidity": 0,
    "ulcers on tongue": 0,
    "muscle wasting": 0,
    "vomiting": 0,
    "burning micturition": 0,
    "spotting urination": 0,
    "fatigue": 0,
    "weight gain": 0,
    "anxiety": 0,
    "cold hands and feets": 0,
    "mood swings": 0,
    "weight loss": 0,
    "restlessness": 0,
    "lethargy": 0,
    "patches in throat": 0,
    "irregular sugar level": 0,
    "cough": 0,
    "high fever": 0,
    "sunken eyes": 0,
    "breathlessness": 0,
    "sweating": 0,
    "dehydration": 0,
    "indigestion": 0,
    "headache": 0,
    "yellowish skin": 0,
    "dark urine": 0,
    "nausea": 0,
    "loss of appetite": 0,
    "pain behind the eyes": 0,
    "back pain": 0,
    "constipation": 0,
    "abdominal pain": 0,
    "diarrhoea": 0,
    "mild fever": 0,
    "yellow urine": 0,
    "yellowing of eyes": 0,
    "acute liver failure": 0,
    "fluid overload": 0,
    "swelling of stomach": 0,
    "swelled lymph nodes": 0,
    "malaise": 0,
    "blurred and distorted_vision": 0,
    "phlegm": 0,
    "throat irritation": 0,
    "redness of eyes": 0,
    "sinus pressure": 0,
    "runny nose": 0,
    "congestion": 0,
    "chest pain": 0,
    "weakness in limbs": 0,
    "fast heart rate": 0,
    "pain during bowel movements": 0,
    "pain in anal region": 0,
    "bloody stool": 0,
    "irritation in anus": 0,
    "neck pain": 0,
    "dizziness": 0,
    "cramps": 0,
    "bruising": 0,
    "obesity": 0,
    "swollen legs": 0,
    "swollen blood vessels": 0,
    "puffy face and eyes": 0,
    "enlarged thyroid": 0,
    "brittle nails": 0,
    "swollen extremeties": 0,
    "excessive hunger": 0,
    "extra marital contacts": 0,
    "drying and tingling lips": 0,
    "slurred speech": 0,
    "knee pain": 0,
    "hip joint pain": 0,
    "muscle weakness": 0,
    "stiff neck": 0,
    "swelling joints": 0,
    "movement stiffness": 0,
    "spinning movements": 0,
    "loss of balance": 0,
    "unsteadiness": 0,
    "weakness of one body side": 0,
    "loss of smell": 0,
    "bladder discomfort": 0,
    "foul smell of urine": 0,
    "continuous feel of urine": 0,
    "passage of gases": 0,
    "internal itching": 0,
    "toxic look (typhos)": 0,
    "depression": 0,
    "irritability": 0,
    "muscle pain": 0,
    "altered sensorium": 0,
    "red spots over body": 0,
    "belly pain": 0,
    "abnormal menstruation": 0,
    "dischromic patches": 0,
    "watering from eyes": 0,
    "increased appetite": 0,
    "polyuria": 0,
    "family history": 0,
    "mucoid sputum": 0,
    "rusty sputum": 0,
    "lack of concentration": 0,
    "visual disturbances": 0,
    "receiving blood transfusion": 0,
    "receiving unsterile injections": 0,
    "coma": 0,
    "stomach bleeding": 0,
    "distention of abdomen": 0,
    "history of alcohol consumption": 0,
    "fluid overload": 0,
    "blood in sputum": 0,
    "prominent veins on calf": 0,
    "palpitations": 0,
    "painful walking": 0,
    "pus filled pimples": 0,
    "blackheads": 0,
    "scurring": 0,
    "skin peeling": 0,
    "silver like dusting": 0,
    "small dents in nails": 0,
    "inflammatory nails": 0,
    "blister": 0,
    "red sore around nose": 0,
    "yellow crust ooze": 0,
    "prognosis": 0
}

diseases = {
    'Fungal infection': [],
    'Allergy': [],
    'GERD': [],
    'Chronic cholestasis': [],
    'Drug Reaction': [],
    'Peptic ulcer diseae': [],
    'AIDS': [],
    'Diabetes ': [],
    'Gastroenteritis': [],
    'Bronchial Asthma': [],
    'Hypertension ': [],
    'Migraine': [],
    'Cervical spondylosis': [],
    'Paralysis (brain hemorrhage)': [],
    'Jaundice': [],
    'Malaria': [],
    'Chicken pox': [],
    'Dengue': [],
    'Typhoid': [],
    'hepatitis A': [],
    'Hepatitis B': [],
    'Hepatitis C': [],
    'Hepatitis D': [],
    'Hepatitis E': [],
    'Alcoholic hepatitis': [],
    'Tuberculosis': [],
    'Common Cold': [],
    'Pneumonia': [],
    'Dimorphic hemmorhoids(piles)': [],
    'Heart attack': [],
    'Varicose veins': [],
    'Hypothyroidism': [],
    'Hyperthyroidism': [],
    'Hypoglycemia': [],
    'Osteoarthristis': [],
    'Arthritis': [],
    '(vertigo) Paroymsal  Positional Vertigo': [],
    'Acne': [],
    'Urinary tract infection': [],
    'Psoriasis': [],
    'Impetigo)': []
}


import numpy as np

def predict_disease(input_symptoms):
    # Create a copy of the symptoms dictionary
    symptoms_copy = symptoms.copy()

    # Set user-input symptoms to 1
    for symptom in input_symptoms:
        symptoms_copy[symptom] = 1

    # Convert the symptoms to a NumPy array
    input_data = np.array(list(symptoms_copy.values()), dtype=np.float32)

    # Ensure input_data has the expected shape (same as the model's input shape)
    expected_input_shape = (1, len(symptoms))
    if input_data.shape != expected_input_shape:
        # If the shape doesn't match, reshape the input_data to match the model's expected input shape
        input_data = input_data.reshape(expected_input_shape)

    # Set the input tensor of the model
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run the inference
    interpreter.invoke()

    # Get the output tensor and find the disease with the highest probability
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_disease_index = np.argmax(output_data)
    predicted_disease = list(diseases.keys())[predicted_disease_index]

    return predicted_disease

# User input (provide a list of symptoms)
user_input = ["vomiting", "chest pain", "breathlessness", "sweating"]

# Predict the disease based on user input

predicted_disease = predict_disease(user_input)

# Convert the output to a string
predicted_disease_str = str(predicted_disease)

print("Predicted Disease:", predicted_disease_str)



