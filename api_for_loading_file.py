from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# Define the folder where uploaded files will be saved
UPLOAD_FOLDER = 'C:/Users/React/OneDrive/Desktop/airrdog/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        # Save the uploaded file to the UPLOAD_FOLDER
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return jsonify({"message": "File uploaded successfully"})

if __name__ == '__main__':
    app.run(debug=False)
