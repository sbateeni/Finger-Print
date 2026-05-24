import base64
import html
import os
from datetime import datetime

import cv2

from config import APP_VERSION, OUTPUT_DIR, SOFTWARE_NAME
from utils.report_i18n import report_lang, s, tier_text, verdict_text

STAGE_KEYS = [
    ("stage_processed", "processed"),
    ("stage_quality", "quality_map"),
    ("stage_singular", "singular_vis"),
    ("stage_ridges", "ridges"),
    ("stage_skeleton", "skeleton"),
    ("stage_minutiae", "minutiae_vis"),
]


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


def _pipeline_gallery_html(branch: dict, title: str, lang: str) -> str:
    if not branch:
        return ""
    lg = report_lang(lang)
    parts = [
        f'<h3 class="section-h">{html.escape(title)}</h3>',
    ]
    stats = []
    if branch.get("white_pre") is not None:
        stats.append(f"{s(lg, 'stats_binary')}: {branch['white_pre']}")
    if branch.get("white_ridges") is not None:
        stats.append(f"{s(lg, 'stats_gabor')}: {branch['white_ridges']}")
    if branch.get("white_skel") is not None:
        stats.append(f"{s(lg, 'stats_skel')}: {branch['white_skel']}")
    if branch.get("n_min") is not None:
        stats.append(f"{s(lg, 'stats_min')}: {branch['n_min']}")
    if stats:
        parts.append("<p class='stats-line'>" + " | ".join(stats) + "</p>")

    parts.append('<div class="gallery">')
    for label_key, key in STAGE_KEYS:
        uri = _png_data_uri(branch.get(key))
        if not uri:
            continue
        parts.append(
            '<div class="gallery-item"><p class="cap">'
            + html.escape(s(lg, label_key))
            + '</p><img src="'
            + uri
            + '" alt="" role="presentation" loading="lazy"/></div>'
        )
    parts.append("</div>")
    return "\n".join(parts)


def _report_css(lang: str) -> str:
    lg = report_lang(lang)
    font = (
        "'Noto Sans Arabic', 'Segoe UI', Tahoma, Arial, sans-serif"
        if lg == "ar"
        else "'Segoe UI', Tahoma, Arial, sans-serif"
    )
    return f"""
    @page {{ margin: 14mm; }}
    body {{ font-family: {font}; margin: 0; background: #e8ecf2; color: #1a2332; line-height: 1.55; }}
    .container {{ max-width: 980px; margin: 24px auto; background: #fff; padding: 28px 32px; border-radius: 12px;
      box-shadow: 0 4px 24px rgba(15,23,42,0.08); border: 1px solid #d8e0ea; }}
    .hero {{ text-align: center; padding-bottom: 18px; border-bottom: 2px solid #0b5cab; margin-bottom: 22px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 1.65rem; color: #0b3d6e; }}
    .hero .sub {{ color: #5a6b7d; margin: 0; }}
    .disclaimer {{ background: linear-gradient(135deg,#fff9e6,#fff3cc); border: 1px solid #e6c200;
      padding: 16px 18px; border-radius: 10px; margin-bottom: 20px; font-size: 0.95rem; }}
    .audit {{ background: #f4f8fc; border: 1px solid #c5d4e8; padding: 16px 18px; border-radius: 10px;
      margin: 18px 0; font-size: 0.9rem; word-break: break-all; }}
    .result-box {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px 20px; margin: 14px 0; }}
    .section-h {{ margin-top: 1.6rem; border-bottom: 2px solid #0b5cab; padding-bottom: 8px; color: #0b3d6e; font-size: 1.1rem; }}
    .highlight {{ color: #0b5cab; font-weight: 700; }}
    .success {{ color: #0d7a3e; font-weight: 700; }}
    table.params {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 0.9rem; }}
    table.params td, table.params th {{ border: 1px solid #cbd5e1; padding: 9px 11px; }}
    table.params th {{ background: #e9eef5; font-weight: 600; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 14px; margin: 12px 0; }}
    .gallery-item img {{ max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #cbd5e1; }}
    .gallery-item.full {{ grid-column: 1 / -1; }}
    .cap {{ font-size: 0.82rem; color: #475569; margin: 0 0 6px; font-weight: 600; }}
    .tier {{ font-size: 1.2rem; font-weight: 800; margin-top: 12px; color: #0b3d6e; padding: 10px 14px;
      background: #eef6ff; border-radius: 8px; border-left: 4px solid #0b5cab; }}
    [dir="rtl"] .tier {{ border-left: none; border-right: 4px solid #0b5cab; }}
    .small {{ font-size: 0.85rem; color: #64748b; }}
    .verdict-box {{ font-size: 1.05rem; font-weight: 700; color: #0d7a3e; margin: 8px 0; }}
    .stats-line {{ font-size: 0.88rem; color: #475569; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; font-size: 0.85em; }}
    """


def generate_report(
    original_img,
    partial_img,
    match_result,
    output_dir=OUTPUT_DIR,
    audit=None,
    pipeline=None,
    lang: str = "ar",
):
    try:
        lg = report_lang((audit or {}).get("report_lang") or lang)
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
        tier_lbl = html.escape(tier_text(lg, status))
        verdict_lbl = html.escape(verdict_text(lg, match_result))

        fp = (audit or {}).get("form_params") or {}
        mode_raw = str(fp.get("analysis_mode") or "deep").lower()
        mode_lbl = s(lg, "deep") if mode_raw in ("deep", "1", "true", "on", "wide") else s(lg, "fast")

        mp = int(match_result.get("matched_points") or 0)
        tp = int(match_result.get("total_partial") or 0)
        if lg == "ar":
            score_explanation = (
                f"شرح النسبة: {mp} ÷ {tp} = أزواج متطابقة ÷ إجمالي نقاط المقارنة (ليست تغطية مساحة)."
                if tp > 0
                else ""
            )
        else:
            score_explanation = (
                f"Score: {mp} / {tp} = matched pairs ÷ total query minutiae (not area coverage)."
                if tp > 0
                else ""
            )
        score_explanation_html = (
            f'<p class="small">{html.escape(score_explanation)}</p>' if score_explanation else ""
        )

        params_html = ""
        if fp:
            for k, v in fp.items():
                if k == "report_lang":
                    continue
                params_html += f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>"

        alignment_block = ""
        if match_result.get("alignment"):
            al = match_result.get("alignment") or {}
            baseline_m = int(match_result.get("baseline_matched") or 0)
            final_m = int(match_result.get("matched_points") or 0)
            gain_m = int(match_result.get("alignment_gain_matches") or (final_m - baseline_m))
            gain_s = float(match_result.get("alignment_gain_score") or 0.0)
            if lg == "ar":
                al_txt = (
                    f"محاذاة تقريبية: dx={al.get('dx', 0)} dy={al.get('dy', 0)} "
                    f"دوران={float(al.get('rot_deg', 0.0)):.1f}°. "
                    f"تطابق أساسي: {baseline_m}، نهائي: {final_m}، تحسّن: {gain_m}، Δنسبة: {gain_s:+.2f}."
                )
            else:
                al_txt = (
                    f"Alignment: dx={al.get('dx', 0)} dy={al.get('dy', 0)} "
                    f"rot={float(al.get('rot_deg', 0.0)):.1f}°. "
                    f"Baseline matches: {baseline_m}, final: {final_m}, gain: {gain_m}, score Δ: {gain_s:+.2f}."
                )
            alignment_block = (
                f'<div class="result-box"><h3>{html.escape(s(lg, "alignment_title"))}</h3>'
                f"<p>{html.escape(al_txt)}</p>"
                f'<p class="small">{html.escape(s(lg, "alignment_hint"))}</p></div>'
            )

        pipeline_html = ""
        if pipeline:
            pipeline_html += _pipeline_gallery_html(
                pipeline.get("reference") or {}, s(lg, "pipeline_ref"), lg
            )
            pipeline_html += _pipeline_gallery_html(
                pipeline.get("query") or {}, s(lg, "pipeline_query"), lg
            )
            uri_mv = _png_data_uri(pipeline.get("matches_vis"))
            if uri_mv:
                pipeline_html += (
                    f'<h3 class="section-h">{html.escape(s(lg, "match_overlay"))}</h3>'
                    f'<div class="gallery"><div class="gallery-item full">'
                    f'<img src="{uri_mv}" alt="" role="presentation" loading="lazy"/></div></div>'
                )

        orb_vis_html = ""
        orb_uri = match_result.get("orb_visualization")
        if orb_uri:
            orb_vis_html = (
                f'<h3 class="section-h">{html.escape(s(lg, "orb_title"))}</h3>'
                f'<div class="gallery"><div class="gallery-item full">'
                f'<img src="{html.escape(str(orb_uri))}" alt="" role="presentation" loading="lazy"/>'
                f"</div></div>"
            )

        fused_html = ""
        if match_result.get("fused_score") is not None:
            fused_html = (
                f"<p>{html.escape(s(lg, 'fused_score'))}: "
                f"<span class='success'>{float(match_result.get('fused_score', 0)):.2f}%</span></p>"
            )
        if match_result.get("combined_verdict") or match_result.get("status"):
            fused_html += f'<p class="verdict-box">{html.escape(s(lg, "combined_verdict"))}: {verdict_lbl}</p>'
        if match_result.get("decision_mode"):
            fused_html += (
                f"<p>{html.escape(s(lg, 'decision_mode'))}: "
                f"<span class='highlight'>{html.escape(str(match_result.get('decision_mode')))}</span></p>"
            )
        fusion_components = match_result.get("fusion_components") or {}
        fusion_comp_html = ""
        if fusion_components:
            fusion_comp_html = (
                f"<p>{html.escape(s(lg, 'fusion_components'))} — "
                f"{html.escape(s(lg, 'fusion_min'))}: "
                f"<span class='success'>{float(fusion_components.get('minutiae_score', 0)):.2f}%</span> | "
                f"{html.escape(s(lg, 'fusion_mcc'))}: "
                f"<span class='success'>{float(fusion_components.get('mcc_score', 0)):.2f}%</span> | "
                f"{html.escape(s(lg, 'fusion_orb'))}: "
                f"<span class='success'>{float(fusion_components.get('orb_score', 0)):.2f}%</span></p>"
            )

        sk_ref_uri = _png_data_uri(original_img)
        sk_cmp_uri = _png_data_uri(partial_img)
        archive_note = (
            f"<p class='small'>{html.escape(s(lg, 'archive_note'))}: "
            f"<code>{html.escape(original_fn)}</code>, <code>{html.escape(partial_fn)}</code></p>"
        )

        font_link = ""
        if lg == "ar":
            font_link = (
                '<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Arabic:wght@400;600;700&display=swap" rel="stylesheet"/>'
            )

        report_content = f"""<!doctype html>
<html lang="{s(lg, 'html_lang')}" dir="{s(lg, 'dir')}">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(s(lg, 'title'))}</title>
  {font_link}
  <style>{_report_css(lg)}</style>
</head>
<body>
  <div class="container">
    <div class="hero">
      <h1>{html.escape(s(lg, 'title'))}</h1>
      <p class="sub">{html.escape(s(lg, 'subtitle'))}</p>
      <p class="small">{html.escape(s(lg, 'generated_at'))}: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
      <p class="small">{html.escape(SOFTWARE_NAME)} · v{html.escape(str(APP_VERSION))}</p>
    </div>

    <div class="disclaimer">
      <strong>{html.escape(s(lg, 'disclaimer_title'))}:</strong> {html.escape(s(lg, 'disclaimer'))}
    </div>

    <div class="audit">
      <h3 style="margin-top:0;">{html.escape(s(lg, 'audit_title'))}</h3>
      <p><strong>{html.escape(s(lg, 'case_ref'))}:</strong> {case_ref}</p>
      <p><strong>{html.escape(s(lg, 'operator'))}:</strong> {op}</p>
      <p><strong>{html.escape(s(lg, 'analysis_mode'))}:</strong> {html.escape(mode_lbl)}</p>
      <p><strong>{html.escape(s(lg, 'sha_ref'))}:</strong><br/>{sha_o}</p>
      <p><strong>{html.escape(s(lg, 'sha_query'))}:</strong><br/>{sha_p}</p>
    </div>

    <div class="result-box">
      <h3>{html.escape(s(lg, 'methodology_title'))}</h3>
      <p>{html.escape(s(lg, 'methodology'))}</p>
    </div>

    <h3 class="section-h">{html.escape(s(lg, 'params_title'))}</h3>
    <table class="params">
      <tr><th>{html.escape(s(lg, 'param_col'))}</th><th>{html.escape(s(lg, 'value_col'))}</th></tr>
      {params_html if params_html else "<tr><td colspan='2'>—</td></tr>"}
    </table>

    <h3 class="section-h">{html.escape(s(lg, 'pipeline_title'))}</h3>
    {pipeline_html}
    {orb_vis_html}

    <h3 class="section-h">{html.escape(s(lg, 'skeletons_title'))}</h3>
    <div class="gallery">
      <div class="gallery-item">
        <p class="cap">{html.escape(s(lg, 'sk_ref'))}</p>
        <img src="{sk_ref_uri}" alt="" role="presentation" loading="lazy"/>
      </div>
      <div class="gallery-item">
        <p class="cap">{html.escape(s(lg, 'sk_query'))}</p>
        <img src="{sk_cmp_uri}" alt="" role="presentation" loading="lazy"/>
      </div>
    </div>
    {archive_note}
    {alignment_block}

    <div class="result-box">
      <h2>{html.escape(s(lg, 'results_title'))}</h2>
      <p>{html.escape(s(lg, 'n_ref'))}: <span class="highlight">{match_result.get('total_original', 0)}</span></p>
      <p>{html.escape(s(lg, 'n_query'))}: <span class="highlight">{match_result.get('total_partial', 0)}</span></p>
      <p>{html.escape(s(lg, 'n_matched'))}: <span class="success">{match_result.get('matched_points', 0)}</span></p>
      <p>{html.escape(s(lg, 'match_score'))}: <span class="success">{float(match_result.get('match_score', 0)):.2f}%</span></p>
      {fused_html}
      {fusion_comp_html}
      {score_explanation_html}
      <p>{html.escape(s(lg, 'dice'))}: <span class="success">{float(match_result.get('dice_score', 0)):.2f}%</span></p>
      <p class="tier">{html.escape(s(lg, 'tier'))}: {tier_lbl}</p>
      <p class="small">{html.escape(s(lg, 'internal_status'))}: {html.escape(status)}</p>
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
