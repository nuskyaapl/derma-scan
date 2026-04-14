from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
import os
import uuid


def clear_old_reports(static_folder):
    if not os.path.exists(static_folder):
        return

    for file in os.listdir(static_folder):
        if file.startswith("report_") and file.endswith(".pdf"):
            try:
                os.remove(os.path.join(static_folder, file))
            except Exception:
                pass


def generate_pdf_report(
    static_folder,
    original_image_path,
    heatmap_path,
    prediction,
    confidence,
    lesion_score=None
):
    report_filename = f"report_{uuid.uuid4().hex}.pdf"
    report_path = os.path.join(static_folder, report_filename)

    c = canvas.Canvas(report_path, pagesize=A4)
    width, height = A4

    # Background
    c.setFillColor(colors.HexColor("#f4f7fb"))
    c.rect(0, 0, width, height, fill=1, stroke=0)

    # Main white card
    card_x = 35
    card_y = 40
    card_w = width - 70
    card_h = height - 80

    c.setFillColor(colors.white)
    c.roundRect(card_x, card_y, card_w, card_h, 18, fill=1, stroke=0)

    y = height - 80

    # Header
    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width / 2, y, "Dermascan AI")

    y -= 22
    c.setFillColor(colors.HexColor("#6b7280"))
    c.setFont("Helvetica", 11)
    c.drawCentredString(width / 2, y, "AI-powered skin lesion analysis and skin cancer detection report")

    # Result panel
    y -= 40
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(50, y - 95, 240, 100, 12, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 14)
    c.drawString(65, y - 15, "Prediction Result")

    c.setFont("Helvetica", 12)
    c.drawString(65, y - 40, f"Prediction: {prediction}")
    c.drawString(65, y - 60, f"Confidence: {confidence:.2f}%")

    if lesion_score is not None:
        c.drawString(65, y - 80, f"Lesion Validation Score: {lesion_score:.2f}%")

    # Images section
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(305, y - 200, 230, 205, 12, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(320, y - 18, "Visual Analysis")

    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#374151"))
    c.drawString(320, y - 35, "Uploaded Image")
    c.drawString(425, y - 35, "Grad-CAM Heatmap")

    if os.path.exists(original_image_path):
        c.drawImage(
            ImageReader(original_image_path),
            320, y - 150,
            width=90, height=90,
            preserveAspectRatio=True, mask='auto'
        )

    if os.path.exists(heatmap_path):
        c.drawImage(
            ImageReader(heatmap_path),
            425, y - 150,
            width=90, height=90,
            preserveAspectRatio=True, mask='auto'
        )

    # Explanation box
    y -= 240
    c.setFillColor(colors.HexColor("#f8fafc"))
    c.roundRect(50, y - 80, 485, 85, 12, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#111827"))
    c.setFont("Helvetica-Bold", 12)
    c.drawString(65, y - 18, "Recommendations")

    # Recommendations based on prediction
    c.setFont("Helvetica", 10)
    c.setFillColor(colors.HexColor("#374151"))

    if "malignant" in prediction.lower():
        c.drawString(65, y - 38, "• The model indicates a high likelihood of a malignant lesion.")
        c.drawString(65, y - 53, "• It is strongly recommended to consult a dermatologist immediately.")
        c.drawString(65, y - 68, "• Avoid self-diagnosis and seek professional medical advice.")
        
    elif "benign" in prediction.lower():
        c.drawString(65, y - 38, "• The lesion appears to be benign based on the model prediction.")
        c.drawString(65, y - 53, "• Regular monitoring is advised for any visible changes.")
        c.drawString(65, y - 68, "• Consult a doctor if the lesion changes in size, shape, or color.")

    else:
        c.drawString(65, y - 38, "• The uploaded image is not recognized as a valid skin lesion.")
        c.drawString(65, y - 53, "• Please upload a clear and focused image of a skin lesion.")
        c.drawString(65, y - 68, "• Ensure proper lighting and avoid blurred images.")

    # Disclaimer box
    y -= 115
    c.setFillColor(colors.HexColor("#fffbeb"))
    c.roundRect(50, y - 55, 485, 60, 12, fill=1, stroke=0)

    c.setFillColor(colors.HexColor("#92400e"))
    c.setFont("Helvetica-Bold", 11)
    c.drawString(65, y - 18, "Disclaimer")

    c.setFont("Helvetica", 9)
    c.drawString(65, y - 35, "DermaScan AI is a decision-support prototype developed for research")
    c.drawString(65, y - 48, "and educational purposes. It must not be used as a medical diagnosis tool.")

    # Footer
    c.setFillColor(colors.HexColor("#64748b"))
    c.setFont("Helvetica", 9)
    c.drawString(50, 25, "Dermascan AI • Final Year Project Prototype")

    c.save()

    return report_filename, report_path