import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the disease dataset
disease_data = pd.read_csv("C:/Users/React/OneDrive/Desktop/airrdog/uploads/disease.csv")


# Split data into features (X) and target (y)
X_disease = disease_data.drop(columns=["prognosis"])
y_disease = disease_data["prognosis"]

# Label encode the target variable (assuming "prognosis" is a string)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_disease)

# Split the data into training and testing sets for disease prediction
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(X_disease, y_encoded, test_size=0.2, random_state=42)

# Initialize and train the disease prediction TensorFlow model (e.g., a neural network)
disease_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_disease.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(np.unique(y_encoded)), activation='softmax')
])

disease_model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

disease_model.fit(X_train_disease, y_train_disease, epochs=10, validation_data=(X_test_disease, y_test_disease))

# Save the disease prediction TensorFlow model
disease_model.save("predict_disease_tensorflow.h5")

# Convert the disease prediction TensorFlow model to TensorFlow Lite
disease_converter = tf.lite.TFLiteConverter.from_keras_model(disease_model)
disease_tflite_model = disease_converter.convert()

# Save the disease prediction TensorFlow Lite model to a file
with open("C:/Users/React/OneDrive/Desktop/airrdog/model/predict_disease.tflite", "wb") as f:
    f.write(disease_tflite_model)

# Load the disease prediction TensorFlow Lite model
disease_interpreter = tf.lite.Interpreter(model_path="C:/Users/React/OneDrive/Desktop/airrdog/model/predict_disease.tflite")
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

########################################################################## FLASK APP ######################################################


from flask import Flask, send_from_directory
app = Flask(__name__)
@app.route('/uploads/predict_disease.tflite')
def download_file():
    try:
        return send_from_directory('uploads', 'predict_disease.tflite', as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404
if __name__ == '__main__':
    app.run(port=8082)