from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import os
import uuid
from tensorflow.keras.applications.efficientnet import preprocess_input
from heatmap import make_gradcam_heatmap, overlay_heatmap, clear_old_heatmaps
from report import generate_pdf_report, clear_old_reports
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Maximum upload size (5MB)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Allowed image types
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Model paths
CANCER_MODEL_PATH = "skin_cancer_best.keras"
GATEKEEPER_MODEL_PATH = "lesion_gatekeeper.keras"

# Load both models once when server starts
cancer_model = tf.keras.models.load_model(CANCER_MODEL_PATH)
gatekeeper_model = tf.keras.models.load_model(GATEKEEPER_MODEL_PATH)

IMG_SIZE = (224, 224)

# Thresholds
LESION_THRESHOLD = 0.70
CANCER_THRESHOLD = 0.50


# ------------------------------
# Check file extension
# ------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ------------------------------
# Image preprocessing
# ------------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    img_array = np.array(image).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ------------------------------
# Home page
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


# ------------------------------
# Prediction endpoint
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload JPG or PNG images."})

    try:
        static_folder = os.path.join(app.root_path, "static")
        os.makedirs(static_folder, exist_ok=True)

        # save uploaded original image
        original_filename = f"upload_{uuid.uuid4().hex}.jpg"
        original_image_path = os.path.join(static_folder, original_filename)

        # 1. Open and preprocess image
        image = Image.open(file.stream).convert("RGB")
        image.save(original_image_path)

        processed_image = preprocess_image(image)

        # 2. Gatekeeper
        gate_score = float(gatekeeper_model.predict(processed_image, verbose=0)[0][0])
        lesion_score = 1 - gate_score   # if class order was ['lesion', 'non_lesion']

        if lesion_score < LESION_THRESHOLD:
            return jsonify({
                "prediction": "Unsupported image",
                "confidence": round((1 - lesion_score) * 100, 2),
                "note": "Please upload a clear skin lesion image."
            })

        # 3. Cancer prediction
        cancer_score = float(cancer_model.predict(processed_image, verbose=0)[0][0])

        if cancer_score >= CANCER_THRESHOLD:
            label = "Malignant (Cancerous)"
            confidence = cancer_score
        else:
            label = "Benign (Non-Cancerous)"
            confidence = 1 - cancer_score

        # 4. Only AFTER label exists -> generate heatmap/report
        clear_old_heatmaps(static_folder)
        clear_old_reports(static_folder)

        heatmap = make_gradcam_heatmap(
            processed_image,
            cancer_model,
            last_conv_layer_name="top_conv"
        )

        heatmap_img = overlay_heatmap(image, heatmap)

        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
        heatmap_path = os.path.join(static_folder, heatmap_filename)
        cv2.imwrite(heatmap_path, heatmap_img)

        report_filename, report_path = generate_pdf_report(
            static_folder=static_folder,
            original_image_path=original_image_path,
            heatmap_path=heatmap_path,
            prediction=label,
            confidence=round(confidence * 100, 2),
            lesion_score=round(lesion_score * 100, 2)
        )

        # 5. Return final response
        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "lesion_score": round(lesion_score * 100, 2),
            "heatmap": f"/static/{heatmap_filename}",
            "report": f"/static/{report_filename}"
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    




# ------------------------------
# Run server
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)