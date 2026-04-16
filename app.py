from flask import Flask, render_template, request, jsonify
import tensorflow as tf          # Import TensorFlow for loading and using trained deep learning models
import numpy as np               # NumPy is used for image array handling and numerical operations
from PIL import Image            # PIL (Python Imaging Library) is used to open and process uploaded images
import cv2                       # OpenCV is used for saving heatmap images
import os                        # os is used for file paths and folder creation
import uuid                      # uuid is used to generate unique filenames for uploaded images, heatmaps, and reports
from tensorflow.keras.applications.efficientnet import preprocess_input  # EfficientNet-specific preprocessing function
from heatmap import make_gradcam_heatmap, overlay_heatmap, clear_old_heatmaps
from report import generate_pdf_report, clear_old_reports
from werkzeug.utils import secure_filename



# Create Flask application instance
app = Flask(__name__)

# Set maximum allowed upload size to 5MB
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

# Define which image file extensions are allowed
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# File paths of the trained models
CANCER_MODEL_PATH = "skin_cancer_best.keras"
GATEKEEPER_MODEL_PATH = "lesion_gatekeeper.keras"

# Load both models only once when the server starts
cancer_model = tf.keras.models.load_model(CANCER_MODEL_PATH)
gatekeeper_model = tf.keras.models.load_model(GATEKEEPER_MODEL_PATH)

# Input image size expected by the models
IMG_SIZE = (224, 224)

# Threshold used by gatekeeper model
# If lesion score is lower than this, image is considered unsupported
LESION_THRESHOLD = 0.70

# Threshold used by cancer model
# If score is >= 0.50 -> malignant, otherwise benign
CANCER_THRESHOLD = 0.50


# --------------------------------------------------
# Check whether uploaded file has an allowed extension
# --------------------------------------------------
def allowed_file(filename):
    """
    Returns True if the uploaded file has a valid extension
    such as .png, .jpg, or .jpeg.
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# --------------------------------------------------
# Preprocess image before feeding into the models
# --------------------------------------------------
def preprocess_image(image):
    """
    Converts uploaded image into model-ready format.

    Steps:
    1. Convert to RGB
    2. Resize to 224x224
    3. Convert to NumPy array
    4. Apply EfficientNet preprocessing
    5. Add batch dimension
    """

    image = image.convert("RGB")

    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# --------------------------------------------------
# Route: Home page
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Route: About page
# --------------------------------------------------
@app.route("/about")
def about():
    return render_template("about.html")


# --------------------------------------------------
# Route: Contact page
# --------------------------------------------------
@app.route("/contact")
def contact():
    return render_template("contact.html")


# --------------------------------------------------
# Route: Disclaimer / Privacy page
# --------------------------------------------------
@app.route("/disclaimer")
def disclaimer():
    return render_template("disclaimer.html")


# --------------------------------------------------
# Route: Feedback page
# --------------------------------------------------
@app.route("/feedback")
def feedback():
    return render_template("feedback.html")


# --------------------------------------------------
# Route: Prediction endpoint
# This route receives the uploaded image, validates it,
# runs gatekeeper + cancer model, generates heatmap and report,
# then returns the final result as JSON
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    # Check if the request actually contains a file
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    # Get the uploaded file
    file = request.files["file"]

    # Check if filename is empty
    if file.filename == "":
        return jsonify({"error": "Empty filename"})

    # Validate file extension
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Please upload JPG or PNG images."})

    try:
        # Create /static folder if it does not exist
        # This folder is used to store uploaded images, heatmaps, and reports
        static_folder = os.path.join(app.root_path, "static")
        os.makedirs(static_folder, exist_ok=True)

        # Generate unique filename for uploaded original image
        original_filename = f"upload_{uuid.uuid4().hex}.jpg"
        original_image_path = os.path.join(static_folder, original_filename)

        # ------------------------------
        # Step 1: Open and preprocess image
        # ------------------------------

        # Open uploaded image and convert to RGB
        image = Image.open(file.stream).convert("RGB")

        # Save original image into static folder
        image.save(original_image_path)

        # Preprocess image for model prediction
        processed_image = preprocess_image(image)

        # ------------------------------
        # Step 2: Gatekeeper prediction
        # ------------------------------

        # Gatekeeper model predicts whether this is a valid lesion image
        gate_score = float(gatekeeper_model.predict(processed_image, verbose=0)[0][0])

        # The gatekeeper was trained with output representing "non-lesion"
        # so lesion score is calculated as 1 - gate_score
        lesion_score = 1 - gate_score

        # If lesion score is too low, reject the image
        if lesion_score < LESION_THRESHOLD:
            return jsonify({
                "prediction": "Unsupported image",
                "confidence": round((1 - lesion_score) * 100, 2),
                "note": "Please upload a clear skin lesion image."
            })

        # ------------------------------
        # Step 3: Cancer model prediction
        # ------------------------------

        # Cancer model predicts malignant probability
        cancer_score = float(cancer_model.predict(processed_image, verbose=0)[0][0])

        # Convert probability into final label + confidence
        if cancer_score >= CANCER_THRESHOLD:
            label = "Malignant (Cancerous)"
            confidence = cancer_score
        else:
            label = "Benign (Non-Cancerous)"
            confidence = 1 - cancer_score

        # ------------------------------
        # Step 4: Generate heatmap + report
        # Only do this after valid label is created
        # ------------------------------

        # Remove previous heatmaps and reports
        clear_old_heatmaps(static_folder)
        clear_old_reports(static_folder)

        # Create Grad-CAM heatmap from the cancer model
        heatmap = make_gradcam_heatmap(
            processed_image,
            cancer_model,
            last_conv_layer_name="top_conv"
        )

        # Overlay heatmap on original image
        heatmap_img = overlay_heatmap(image, heatmap)

        # Generate unique filename for heatmap image
        heatmap_filename = f"heatmap_{uuid.uuid4().hex}.jpg"
        heatmap_path = os.path.join(static_folder, heatmap_filename)

        # Save heatmap image into static folder
        cv2.imwrite(heatmap_path, heatmap_img)

        # Generate PDF report with original image, heatmap, and results
        report_filename, report_path = generate_pdf_report(
            static_folder=static_folder,
            original_image_path=original_image_path,
            heatmap_path=heatmap_path,
            prediction=label,
            confidence=round(confidence * 100, 2),
            lesion_score=round(lesion_score * 100, 2)
        )

        # ------------------------------
        # Step 5: Return final response to frontend
        # ------------------------------
        return jsonify({
            "prediction": label,
            "confidence": round(confidence * 100, 2),
            "lesion_score": round(lesion_score * 100, 2),
            "heatmap": f"/static/{heatmap_filename}",
            "report": f"/static/{report_filename}"
        })

    # If any error happens, send error message back to frontend
    except Exception as e:
        return jsonify({"error": str(e)})


# --------------------------------------------------
# Run Flask server
# host="0.0.0.0" -> makes it accessible from other devices/network
# port=5001 -> app runs on port 5001
# --------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)