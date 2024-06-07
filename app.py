# This is a server to process images sent via restful api
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import pydicom
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
from io import BytesIO
from flask_cors import CORS

Allowed_Extensions = set(["png", "jpg", "jpeg", "gif", "dcm"])


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Allowed_Extensions


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/upload", methods=["POST"])
def process_image():
    model_path = "modelv1.h5"
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})
        if file and allowed_file(file.filename):
            if file.filename.lower().endswith(".dcm"):
                dcm_file = pydicom.dcmread(file)
                # Extract pixel data from DICOM file
                print("did this here run")
                pixel_data = dcm_file.pixel_array
                # Convert pixel data to PIL Image
                img = Image.fromarray(pixel_data)
            else:
                img = Image.open(file)
            img = img.convert("RGB")
            img = img.resize((150, 150))
            img_tensor = tf.keras.preprocessing.image.img_to_array(img)
            img_tensor = np.expand_dims(img_tensor, axis=0)
            img_tensor /= 255.0
            model = tf.keras.models.load_model(model_path)
            # Predict the class of the image
            prediction = model.predict(img_tensor)
            binary_prediction = (prediction > 0.5).astype(int)

            last_conv_layer = model.get_layer(
                "block5_conv2"
            )  # Adjust to your model architecture
            heatmap_model = tf.keras.models.Model(
                [model.inputs], [last_conv_layer.output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_output, predictions = heatmap_model(img_tensor)
                class_idx = tf.argmax(predictions[0])
                heatmap = tape.gradient(predictions, conv_output)[0]
                heatmap = tf.reduce_mean(heatmap, axis=-1)

            # Normalize the heatmap
            heatmap = tf.maximum(heatmap, 0)
            heatmap /= tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()

            # Resize the heatmap to match the image size
            heatmap = cv2.resize(heatmap, (img.width, img.height))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(
                cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0
            )
            heatmap_filename = os.path.join(app.config["UPLOAD_FOLDER"], "heatmap.jpg")
            overlay_filename = os.path.join(app.config["UPLOAD_FOLDER"], "overlay.jpg")

            cv2.imwrite(heatmap_filename, heatmap)
            cv2.imwrite(overlay_filename, superimposed_img)

            return jsonify(
                {
                    "pneumonia": bool(binary_prediction[0][0]),
                }
            )
    return jsonify({"error": "Invalid request"})

def apply_window_level(pixel_data, window, level):
    # Apply windowing and leveling to pixel data
    windowed_data = pixel_data.copy()
    window_min = level - (window / 2)
    window_max = level + (window / 2)
    windowed_data[windowed_data < window_min] = window_min
    windowed_data[windowed_data > window_max] = window_max

    # Normalize to 8-bit range
    normalized_data = ((windowed_data - window_min) / (window_max - window_min) * 255).astype('uint8')

    return normalized_data

@app.route('/convert', methods=['POST'])
def convert_dicom_to_jpg():
    try:
        # Check if the request contains a file
        if 'dicom_file' not in request.files:
            return jsonify({'error': 'No DICOM file provided'})

        dicom_file = request.files['dicom_file']

        # Check if the file has a valid extension (optional)
        if not dicom_file.filename.endswith('.dcm'):
            return jsonify({'error': 'Invalid file format. Please provide a DICOM file'})

        # Read the DICOM file
        dicom_data = pydicom.dcmread(dicom_file)

        # Extract pixel data
        
        pixel_data = dicom_data.pixel_array
         # Set window and level values (adjust these as needed)
        window = 4095
        level = 2047.5

        # normalized_data
        
        # normalized_data = apply_window_level(pixel_data, window, level)

        # Convert pixel data to PIL Image
        img = Image.fromarray(pixel_data)
        # img = Image.fromarray(normalized_data)

        # Convert PIL Image to bytes (JPEG format)
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Send the JPEG image back as a file or bytes
        return send_file(img_bytes, mimetype='image/png', as_attachment=True, download_name='converted_image.png')

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')
