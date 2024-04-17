import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer

# Load data

data = {
    'Disease': [
        'Varicose veins', 'Jaundice', 'Psoriasis', '(vertigo) Paroymsal Positional Vertigo',
        'Hepatitis B', 'Osteoarthritis', 'Dimorphic hemorrhoids(piles)', 'Impetigo',
        'Malaria', 'Common Cold', 'Hypothyroidism', 'Arthritis', 'Hepatitis D',
        'Drug Reaction', 'Cervical spondylosis', 'Gastroenteritis', 'Hypoglycemia',
        'Tuberculosis', 'Alcoholic hepatitis', 'Fungal infection', 'Allergy',
        'Bronchial Asthma', 'Chronic cholestasis', 'Hyperthyroidism', 'Migraine',
        'Acne', 'Hypertension', 'Heart attack', 'Paralysis (brain hemorrhage)',
        'Typhoid', 'GERD', 'AIDS', 'Peptic ulcer disease', 'Hepatitis C',
        'Urinary tract infection', 'Dengue', 'Hepatitis E', 'Diabetes',
        'Chicken pox', 'hepatitis A', 'Pneumonia'
    ],
    'Emergency Medicine': [
        'Pain relievers', 'Supportive care', 'Topical corticosteroids', 'Meclizine',
        'Supportive care', 'NSAIDs', 'Pain relievers', 'Antibiotics',
        'ACTs', 'Paracetamol', 'Levothyroxine', 'NSAIDs', 'Supportive care',
        'Antihistamines', 'Pain relievers', 'Rehydration', 'Glucose',
        'Antibiotics', 'Supportive care', 'Antifungal', 'Antihistamines',
        'Bronchodilators', 'Ursodeoxycholic acid', 'Beta blockers', 'Pain relievers',
        'Topical treatments', 'Antihypertensives', 'Aspirin', 'Supportive care',
        'Antibiotics', 'Antacids', 'Antiretroviral therapy', 'Antacids', 'Supportive care',
        'Antibiotics', 'Supportive care', 'Supportive care', 'Insulin',
        'Antivirals', 'Supportive care', 'Antibiotics'
    ]
}

df = pd.DataFrame(data)
df

tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Disease'])
X = tokenizer.texts_to_sequences(df['Disease'])
X = tf.keras.preprocessing.sequence.pad_sequences(X)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Emergency Medicine'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=X.shape[1]))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Assuming you have previously defined the tokenizer as in your first code block

# Your second code block
interpreter = tf.lite.Interpreter(model_path="C:/Users/React/OneDrive/Desktop/airrdog/model/disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sample_diseases = ["Pneumonia"]

# Convert the disease name to a sequence using the loaded tokenizer
X_sample = tokenizer.texts_to_sequences(sample_diseases)

# Ensure the shape matches the expected input shape
expected_shape = input_details[0]['shape']

X_sample = pad_sequences(X_sample, maxlen=expected_shape[1], padding='post')

X_sample = X_sample.astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], X_sample)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_label = np.argmax(output_data, axis=1)
predicted_medicine = label_encoder.inverse_transform(predicted_label)

for disease, medicine in zip(sample_diseases, predicted_medicine):
    print(f"For {disease}, the suggested emergency medicine is: {medicine}")

######################################################################### FLASK APP ###############################################

from flask import Flask, send_from_directory
app = Flask(__name__)
@app.route('/uploads/disease_model.tflite')
def download_file():
    try:
        return send_from_directory('uploads', 'disease_model.tflite', as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404
if __name__ == '__main__':
    app.run(port=8082)
