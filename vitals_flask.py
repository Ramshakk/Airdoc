import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
df = pd.read_csv('C:/Users/React/OneDrive/Desktop/airrdog/New folder/emergency_data.csv')

# Preprocess the data
features = df.drop(columns=["Emergency_Level"])
labels = df["Emergency_Level"]

# Encode the labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Normalize numerical features
numerical_features = features[['Blood_Pressure', 'Blood_Oxygen', 'Temperature', 'Pulse_Rate']]
scaler = StandardScaler()
features[['Blood_Pressure', 'Blood_Oxygen', 'Temperature', 'Pulse_Rate']] = scaler.fit_transform(numerical_features)

# One-hot encode categorical features (if any)
categorical_features = features.drop(columns=numerical_features.columns)
encoder = OneHotEncoder(sparse=False)
categorical_encoded = encoder.fit_transform(categorical_features)
categorical_feature_names = encoder.get_feature_names_out(input_features=categorical_features.columns)
categorical_encoded = pd.DataFrame(categorical_encoded, columns=categorical_feature_names)

# Concatenate numerical and one-hot encoded features
features = pd.concat([numerical_features, categorical_encoded], axis=1)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define and compile the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes for 'Emergency_Level'
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}, Validation Accuracy: {accuracy}')

# Save the trained model with a specific name
model.save('/uploads/emergency_model.tflite')

# Convert the model to TFLite format if needed
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('C:/Users/React/OneDrive/Desktop/airrdog/model/emergency_model.tflite', 'wb') as f:
    f.write(tflite_model)





########################################################################## FLASK APP ######################################################


from flask import Flask, send_from_directory
app = Flask(__name__)
@app.route('/uploads/emergency_model.tflite')
def download_file():
    try:
        return send_from_directory('uploads', 'emergency_model.tflite', as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404
if __name__ == '__main__':
    app.run(port=8082)