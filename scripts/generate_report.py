import os
import sys
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, os.path.abspath("."))


def build_pdf_report(
    inputs: dict,
    predictions: dict,
    recommendation: str,
    crack_image_path: str = None,
    shap_image_path: str = None,
    output_path: str = "reports/building_report.pdf",
):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontSize=22,
        textColor=colors.HexColor("#1A237E"),
        spaceAfter=8,
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#5C6BC0"),
        alignment=1,
        spaceAfter=20,
    )
    section_header_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1A237E"),
        spaceBefore=16,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=10,
        leading=16,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "Small",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
    )

    story = [
        Paragraph("AI STRUCTURAL HEALTH MONITORING REPORT", title_style),
        Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}", subtitle_style),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A237E"), spaceAfter=16),
        Paragraph("Structural Assessment Summary", section_header_style),
    ]

    summary_data = [
        ["Parameter", "Value"],
        ["Building Age", f"{inputs.get('building_age', 'N/A')} years"],
        ["Corrosion Level", f"{inputs.get('corrosion_level', 'N/A')} (0-1 scale)"],
        ["Crack Width", f"{inputs.get('crack_width', 'N/A')} mm"],
        ["Crack Density", f"{inputs.get('crack_density', 'N/A')} cracks/m2"],
        ["Moisture Content", f"{inputs.get('moisture_content', 'N/A')}%"],
        ["Compressive Strength", f"{inputs.get('compressive_strength', 'N/A')} MPa"],
        ["Load Stress", f"{inputs.get('load_stress', 'N/A')} MPa"],
        ["ML Predictions", ""],
        ["Structural Health Index (SHI)", f"{predictions.get('shi', 'N/A')} / 100"],
        ["Risk Level", f"{predictions.get('risk_level', 'N/A')}"],
        ["Remaining Useful Life (RUL)", f"{predictions.get('rul', 'N/A')} years"],
    ]

    table = Table(summary_data, colWidths=[8 * cm, 8 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A237E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 11),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 10),
                ("BACKGROUND", (0, 8), (-1, 8), colors.HexColor("#E8EAF6")),
                ("FONTNAME", (0, 8), (-1, 8), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend([table, Spacer(1, 0.5 * cm)])

    if crack_image_path and os.path.exists(crack_image_path):
        story.append(Paragraph("Crack Detection Image", section_header_style))
        story.append(RLImage(crack_image_path, width=14 * cm, height=9 * cm))
        story.append(Paragraph("Figure: YOLOv8 crack detection output on sample image.", small_style))
        story.append(Spacer(1, 0.3 * cm))

    if shap_image_path and os.path.exists(shap_image_path):
        story.append(Paragraph("Explainable AI - Feature Importance", section_header_style))
        story.append(RLImage(shap_image_path, width=14 * cm, height=9 * cm))
        story.append(Paragraph("Figure: SHAP feature importance for SHI prediction.", small_style))
        story.append(Spacer(1, 0.3 * cm))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#5C6BC0"), spaceAfter=8))
    story.append(Paragraph("AI Inspector Recommendation", section_header_style))
    for line in recommendation.splitlines():
        stripped = line.strip()
        if stripped:
            story.append(Paragraph(stripped, body_style))

    story.extend(
        [
            Spacer(1, 1 * cm),
            HRFlowable(width="100%", thickness=1, color=colors.lightgrey),
            Paragraph(
                "This report is generated by an AI system and should be reviewed by a qualified structural engineer.",
                small_style,
            ),
        ]
    )

    doc.build(story)
    print(f"PDF report saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    from scripts.agent import analyze_structure

    sample_input = {
        "building_age": 45.0,
        "corrosion_level": 0.75,
        "crack_width": 6.2,
        "crack_density": 12.5,
        "moisture_content": 10.2,
        "compressive_strength": 24.0,
        "temperature": 32.0,
        "humidity": 65.0,
        "load_stress": 25.0,
    }

    predictions, recommendation = analyze_structure(sample_input)
    build_pdf_report(
        inputs=sample_input,
        predictions=predictions,
        recommendation=recommendation,
        crack_image_path="runs/detect/predict/test.jpg",
        shap_image_path="outputs/shap/shi_shap_bar.png",
    )
