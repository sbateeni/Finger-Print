import os
import cv2
import numpy as np
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from config import *

def generate_report(original_image1, original_image2, processed_image1, processed_image2,
                   minutiae1, minutiae2, match_result, score, timestamp):
    """
    Generate a detailed PDF report of the fingerprint matching analysis.
    
    Args:
        original_image1 (str): Path to first original image
        original_image2 (str): Path to second original image
        processed_image1 (numpy.ndarray): First processed image
        processed_image2 (numpy.ndarray): Second processed image
        minutiae1 (list): Minutiae points from first fingerprint
        minutiae2 (list): Minutiae points from second fingerprint
        match_result (dict): Matching results
        score (float): Overall similarity score
        timestamp (str): Timestamp for the analysis
        
    Returns:
        str: Path to the generated PDF report
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Define report path
    report_path = os.path.join(OUTPUT_FOLDER, f'report_{timestamp}.pdf')
    
    # Create document
    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Create style
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=12,
        leading=16,
        alignment=1  # Center alignment
    )
    
    # Create story (content)
    story = []
    
    # Add title
    title = Paragraph('Fingerprint Analysis Report', title_style)
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Add date and time
    date_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    date_para = Paragraph(f'Analysis Date: {date_str}', normal_style)
    story.append(date_para)
    story.append(Spacer(1, 20))
    
    # Add original images
    story.append(Paragraph('Original Fingerprints', normal_style))
    story.append(Spacer(1, 10))
    
    # Save and add images to report
    img1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_original.png')
    img2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_original.png')
    cv2.imwrite(img1_path, cv2.imread(original_image1))
    cv2.imwrite(img2_path, cv2.imread(original_image2))
    
    img_table = Table([
        [Image(img1_path, width=200, height=200),
         Image(img2_path, width=200, height=200)]
    ])
    story.append(img_table)
    story.append(Spacer(1, 20))
    
    # Add processed images
    story.append(Paragraph('Processed Fingerprints', normal_style))
    story.append(Spacer(1, 10))
    
    proc1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_processed.png')
    proc2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_processed.png')
    cv2.imwrite(proc1_path, processed_image1)
    cv2.imwrite(proc2_path, processed_image2)
    
    proc_table = Table([
        [Image(proc1_path, width=200, height=200),
         Image(proc2_path, width=200, height=200)]
    ])
    story.append(proc_table)
    story.append(Spacer(1, 20))
    
    # Add minutiae visualization
    story.append(Paragraph('Extracted Minutiae Points', normal_style))
    story.append(Spacer(1, 10))
    
    # Create minutiae visualizations
    from .minutiae_extraction import visualize_minutiae
    min1_img = visualize_minutiae(processed_image1, minutiae1)
    min2_img = visualize_minutiae(processed_image2, minutiae2)
    
    min1_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_1_minutiae.png')
    min2_path = os.path.join(PROCESSED_FOLDER, f'{timestamp}_2_minutiae.png')
    cv2.imwrite(min1_path, min1_img)
    cv2.imwrite(min2_path, min2_img)
    
    min_table = Table([
        [Image(min1_path, width=200, height=200),
         Image(min2_path, width=200, height=200)]
    ])
    story.append(min_table)
    story.append(Spacer(1, 20))
    
    # Add matching visualization
    story.append(Paragraph('Matching Points', normal_style))
    story.append(Spacer(1, 10))
    
    from .matcher import visualize_matches
    match_img = visualize_matches(processed_image1, processed_image2, match_result['matched_minutiae'])
    match_path = os.path.join(RESULTS_FOLDER, f'{timestamp}_match_visualization.png')
    cv2.imwrite(match_path, match_img)
    
    story.append(Image(match_path, width=400, height=200))
    story.append(Spacer(1, 20))
    
    # Add analysis results
    story.append(Paragraph('Analysis Results', normal_style))
    story.append(Spacer(1, 10))
    
    from .scoring import get_score_details, analyze_match_quality
    score_details = get_score_details(match_result)
    analysis = analyze_match_quality(match_result)
    
    # Create results table
    results_data = [
        ['Metric', 'Value'],
        ['Final Score', f"{score_details['total_score']:.2f}%"],
        ['Confidence Level', score_details['confidence']],
        ['Matched Points Count', str(score_details['matched_count'])],
        ['Minutiae Match Score', f"{score_details['minutiae_score']:.2f}%"],
        ['Orientation Match Score', f"{score_details['orientation_score']:.2f}%"],
        ['Ridge Density Match Score', f"{score_details['density_score']:.2f}%"]
    ]
    
    results_table = Table(results_data)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 20))
    
    # Add quality analysis
    story.append(Paragraph('Match Quality Analysis', normal_style))
    story.append(Spacer(1, 10))
    
    story.append(Paragraph(f"Quality Level: {analysis['quality_level']}", normal_style))
    
    if analysis['issues']:
        story.append(Paragraph('Detected Issues:', normal_style))
        for issue in analysis['issues']:
            story.append(Paragraph(f"• {issue}", normal_style))
    
    if analysis['recommendations']:
        story.append(Paragraph('Recommendations:', normal_style))
        for rec in analysis['recommendations']:
            story.append(Paragraph(f"• {rec}", normal_style))
    
    # Build PDF
    doc.build(story)
    
    return report_path 