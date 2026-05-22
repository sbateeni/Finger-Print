import base64
import html
import os
from datetime import datetime

import cv2

from config import APP_VERSION, OUTPUT_DIR, SOFTWARE_NAME


def _png_data_uri(img) -> str:
    if img is None:
        return ""
    try:
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return ""
        return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
    except Exception:
        return ""


def _pipeline_gallery_html(branch: dict, title: str) -> str:
    if not branch:
        return ""
    parts = [f'<h3 style="margin-top:1.5rem;border-bottom:1px solid #ccc;padding-bottom:6px;">{html.escape(title)}</h3>']
    stats = []
    if branch.get("white_pre") is not None:
        stats.append(f"Binary white pixels: {branch['white_pre']}")
    if branch.get("white_ridges") is not None:
        stats.append(f"Gabor white pixels: {branch['white_ridges']}")
    if branch.get("white_skel") is not None:
        stats.append(f"Skeleton white pixels: {branch['white_skel']}")
    if branch.get("n_min") is not None:
        stats.append(f"Filtered minutiae: {branch['n_min']}")
    if stats:
        parts.append("<p><strong>Stats:</strong> " + " | ".join(stats) + "</p>")

    parts.append('<div class="gallery">')
    items = [
        ("Preprocessed Binary", "processed"),
        ("Quality Heatmap", "quality_map"),
        ("Singular Points", "singular_vis"),
        ("Ridge Enhancement (Gabor)", "ridges"),
        ("Skeleton", "skeleton"),
        ("Minutiae Visualization", "minutiae_vis"),
    ]
    for label, key in items:
        uri = _png_data_uri(branch.get(key))
        if not uri:
            continue
        parts.append(
            '<div class="gallery-item"><p class="cap">'
            + html.escape(label)
            + '</p><img src="'
            + uri
            + '" alt="" role="presentation" loading="lazy"/></div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def generate_report(original_img, partial_img, match_result, output_dir=OUTPUT_DIR, audit=None, pipeline=None):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        case_ref_raw = (audit or {}).get("case_reference") or "Case"
        safe_ref = "".join(c for c in str(case_ref_raw) if c.isalnum() or c in (" ", "_", "-")).strip()
        case_folder_name = f"{safe_ref}_{timestamp}" if safe_ref else f"Case_{timestamp}"
        case_dir = os.path.join(output_dir, case_folder_name)
        os.makedirs(case_dir, exist_ok=True)

        report_path = os.path.join(case_dir, "forensic_report.html")
        original_fn = "original_skeleton.png"
        partial_fn = "partial_skeleton.png"
        cv2.imwrite(os.path.join(case_dir, original_fn), original_img)
        cv2.imwrite(os.path.join(case_dir, partial_fn), partial_img)

        op = html.escape((audit or {}).get("operator_name") or "-")
        case_ref = html.escape((audit or {}).get("case_reference") or "-")
        sha_o = html.escape((audit or {}).get("sha256_original") or "-")
        sha_p = html.escape((audit or {}).get("sha256_partial") or "-")
        status = str(match_result.get("status") or "")
        tier_en_map = {
            "HIGH MATCH": "Tier A - High statistical similarity under configured thresholds",
            "MEDIUM MATCH": "Tier B - Medium statistical similarity under configured thresholds",
            "LOW MATCH": "Tier C - Low statistical similarity under configured thresholds",
            "NO MATCH": "Tier D - Below configured similarity threshold",
            "INCONCLUSIVE": "Inconclusive - insufficient quality/evidence for reliable decision",
            "ERROR": "Error - matching failed",
        }
        tier_text = html.escape(tier_en_map.get(status, status or "Unknown"))
        note_en = (
            "This system reports statistical metrics for fingerprint representation comparison "
            "under recorded parameters. It must not be used as sole identity proof and requires "
            "qualified expert review and lab SOP."
        )
        methodology_en = (
            "Grayscale conversion, size normalization, CLAHE, optional denoising, Otsu/adaptive binarization, "
            "multi-orientation Gabor ridge enhancement, thinning, minutiae extraction (ending/bifurcation) with "
            "filtering, and one-to-one constrained matching by distance/angle/type. For partial-print verification, "
            "the query minutiae are searched with translation and small rotation inside the reference frame."
        )
        mp = int(match_result.get("matched_points") or 0)
        tp = int(match_result.get("total_partial") or 0)
        score_explanation = (
            f"Score explanation: {mp} / {tp} = matched pairs divided by total extracted query minutiae. "
            "This is not area coverage."
            if tp > 0
            else ""
        )
        score_explanation_html = (
            f'<p class="small" style="margin-top:-4px;">{html.escape(score_explanation)}</p>'
            if score_explanation
            else ""
        )

        params_html = ""
        if audit and audit.get("form_params"):
            for k, v in audit["form_params"].items():
                params_html += f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"

        alignment_block = ""
        if match_result.get("alignment"):
            al = match_result.get("alignment") or {}
            baseline_m = int(match_result.get("baseline_matched") or 0)
            final_m = int(match_result.get("matched_points") or 0)
            gain_m = int(match_result.get("alignment_gain_matches") or (final_m - baseline_m))
            gain_s = float(match_result.get("alignment_gain_score") or 0.0)
            alignment_block = (
                '<div class="result-box" style="margin-top:12px;"><h3>Partial-to-Reference Alignment</h3><p>'
                + html.escape(
                    f"Approximate alignment: dx={al.get('dx', 0)} dy={al.get('dy', 0)} "
                    f"rot={float(al.get('rot_deg', 0.0)):.1f} deg. "
                    f"Baseline matches: {baseline_m}, final matches: {final_m}, "
                    f"gain: {gain_m}, score delta: {gain_s:+.2f}."
                )
                + '</p><p class="small">Green: matched reference points | Orange: matched partial points after alignment.</p></div>'
            )

        pipeline_html = ""
        if pipeline:
            pipeline_html += _pipeline_gallery_html(pipeline.get("reference") or {}, "Reference Processing Pipeline")
            pipeline_html += _pipeline_gallery_html(pipeline.get("query") or {}, "Query Processing Pipeline")
            uri_mv = _png_data_uri(pipeline.get("matches_vis"))
            if uri_mv:
                pipeline_html += (
                    '<h3 style="margin-top:1.5rem;border-bottom:1px solid #ccc;padding-bottom:6px;">Aligned Match Visualization</h3>'
                    '<div class="gallery"><div class="gallery-item full">'
                    '<p class="cap">Match Overlay</p>'
                    f'<img src="{uri_mv}" alt="" role="presentation" loading="lazy"/></div></div>'
                )

        orb_vis_html = ""
        orb_uri = match_result.get("orb_visualization")
        if orb_uri:
            orb_vis_html = (
                '<h3 style="margin-top:1.5rem;border-bottom:1px solid #ccc;padding-bottom:6px;">ORB Visual Verification</h3>'
                '<div class="gallery"><div class="gallery-item full">'
                '<p class="cap">ORB Feature Matches</p>'
                f'<img src="{html.escape(str(orb_uri))}" alt="" role="presentation" loading="lazy"/></div></div>'
            )

        fused_html = ""
        if match_result.get("fused_score") is not None:
            fused_html = f"<p>Fused Score: <span class='success'>{float(match_result.get('fused_score', 0)):.2f}%</span></p>"
        if match_result.get("combined_verdict"):
            fused_html += (
                f"<p>Combined verdict: <span class='success'>"
                f"{html.escape(str(match_result.get('combined_verdict')))}</span></p>"
            )
        if match_result.get("decision_mode"):
            fused_html += (
                f"<p>Decision mode: <span class='highlight'>"
                f"{html.escape(str(match_result.get('decision_mode')))}</span></p>"
            )
        fusion_components = match_result.get("fusion_components") or {}
        fusion_comp_html = ""
        if fusion_components:
            fusion_comp_html = (
                "<p>Fusion Components - "
                f"Minutiae: <span class='success'>{float(fusion_components.get('minutiae_score', 0)):.2f}%</span> | "
                f"MCC: <span class='success'>{float(fusion_components.get('mcc_score', 0)):.2f}%</span> | "
                f"ORB: <span class='success'>{float(fusion_components.get('orb_score', 0)):.2f}%</span></p>"
            )

        sk_ref_uri = _png_data_uri(original_img)
        sk_cmp_uri = _png_data_uri(partial_img)
        archive_note = (
            f"<p class='small'>Archive PNG files: <code>{html.escape(original_fn)}</code>, "
            f"<code>{html.escape(partial_fn)}</code> in report folder.</p>"
        )

        report_content = f"""<!doctype html>
<html lang="en" dir="ltr">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Forensic Report - Fingerprint Comparison</title>
  <style>
    body {{ font-family: Tahoma, "Segoe UI", Arial, sans-serif; margin: 20px; background: #eef1f5; color:#222; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #fff; padding: 24px; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); }}
    .disclaimer {{ background: #fff8e6; border: 1px solid #e6c200; padding: 14px; border-radius: 8px; margin-bottom: 20px; font-size: 0.95rem; }}
    .audit {{ background: #f4f7fb; border: 1px solid #c5d4e8; padding: 14px; border-radius: 8px; margin: 16px 0; font-size: 0.9rem; word-break: break-all; }}
    .result-box {{ background: #f8f9fa; border-radius: 8px; padding: 16px; margin: 12px 0; }}
    .highlight {{ color: #0b5cab; font-weight: bold; }}
    .success {{ color: #0d7a3e; font-weight: bold; }}
    table.params {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
    table.params td, table.params th {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
    table.params th {{ background: #e9ecef; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; margin: 12px 0; }}
    .gallery-item img {{ max-width: 100%; height: auto; border-radius: 6px; border: 1px solid #ddd; }}
    .gallery-item.full {{ grid-column: 1 / -1; }}
    .cap {{ font-size: 0.85rem; color: #444; margin: 0 0 6px; font-weight: 600; }}
    .tier {{ font-size: 1.15rem; font-weight: 700; margin-top: 10px; color: #1a1a2e; }}
    .small {{ font-size: 0.85rem; color: #666; }}
  </style>
</head>
<body>
  <div class="container">
    <div style="text-align:center;margin-bottom:20px;">
      <h1 style="margin:0;">Forensic Report - Fingerprint Comparison</h1>
      <p style="color:#555;">{html.escape(SOFTWARE_NAME)} - Version {html.escape(str(APP_VERSION))}</p>
      <p>Report generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>

    <div class="disclaimer">
      <strong>Lab disclaimer:</strong> This report contains statistical software outputs and must not be used as sole identity evidence.
    </div>

    <div class="audit">
      <h3 style="margin-top:0;">Audit Trail</h3>
      <p><strong>Case Reference:</strong> {case_ref}</p>
      <p><strong>Operator:</strong> {op}</p>
      <p><strong>SHA-256 (Reference file):</strong><br/>{sha_o}</p>
      <p><strong>SHA-256 (Query file):</strong><br/>{sha_p}</p>
    </div>

    <div class="result-box">
      <h3>Methodology Summary</h3>
      <p>{html.escape(methodology_en)}</p>
    </div>

    <h3>Recorded Run Parameters</h3>
    <table class="params">
      <tr><th>Parameter</th><th>Value</th></tr>
      {params_html if params_html else "<tr><td colspan='2'>-</td></tr>"}
    </table>

    <h3>Pipeline Images</h3>
    {pipeline_html}
    {orb_vis_html}

    <h3 style="margin-top:1.25rem;">Final Skeletons Used for Matching</h3>
    <div class="gallery">
      <div class="gallery-item">
        <p class="cap">Reference Skeleton</p>
        <img src="{sk_ref_uri}" alt="" role="presentation" loading="lazy"/>
      </div>
      <div class="gallery-item">
        <p class="cap">Query Skeleton</p>
        <img src="{sk_cmp_uri}" alt="" role="presentation" loading="lazy"/>
      </div>
    </div>
    {archive_note}
    {alignment_block}

    <div class="result-box">
      <h2>Statistical Comparison Results</h2>
      <p>Reference minutiae count: <span class="highlight">{match_result.get("total_original", 0)}</span></p>
      <p>Query minutiae count: <span class="highlight">{match_result.get("total_partial", 0)}</span></p>
      <p>One-to-one matched pairs: <span class="success">{match_result.get("matched_points", 0)}</span></p>
      <p>Match score (query-based): <span class="success">{float(match_result.get("match_score", 0)):.2f}%</span></p>
      {fused_html}
      {fusion_comp_html}
      {score_explanation_html}
      <p>Dice score: <span class="success">{float(match_result.get("dice_score", 0)):.2f}%</span></p>
      <p class="tier">Forensic tier: {tier_text}</p>
      <p style="margin-top:12px;font-size:0.95rem;color:#444;">{html.escape(note_en)}</p>
      <p style="font-size:0.85rem;color:#666;">Internal status: {html.escape(str(match_result.get("status", "")))}</p>
    </div>
  </div>
</body>
</html>
"""

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        return report_path
    except Exception as e:
        print(f"Error generating report: {str(e)}")
        return None
