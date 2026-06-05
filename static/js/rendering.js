(function (window) {
  "use strict";

  var FP_APP = window.FP_APP || {};

  var STAGE_KEYS = {
    processed: "stage_processed",
    quality_map: "stage_quality",
    singular_vis: "stage_singular",
    ridges: "stage_ridges",
    skeleton: "stage_skeleton",
    minutiae_vis: "stage_minutiae",
    alignment: "stage_alignment_ref",
    side_by_side: "stage_side_by_side",
    minutiae_pairs: "stage_minutiae_pairs",
    alignment_ref: "stage_alignment_ref",
    orb_vis: "stage_orb",
  };

  FP_APP.updateLiveImage = function (ev) {
    var branch = ev.branch;
    var col = branch === "reference" ? FP_APP.$("live-col-ref") :
              branch === "partial" ? FP_APP.$("live-col-partial") :
              (branch === "match" || branch === "orb") ? FP_APP.$("live-match-gallery") : null;
    
    if (!col || !ev.src) return;

    var isMatch = branch === "match" || branch === "orb";
    var fig = document.createElement("figure");
    fig.className = isMatch
      ? "live-fig live-fig--match" + (ev.featured ? " live-fig--featured" : "")
      : "live-fig live-fig--pipe";

    var cap = FP_APP.t(STAGE_KEYS[ev.stage] || ev.stage, ev.stage);
    var meta = [];
    if (ev.white != null) meta.push("بيض: " + ev.white);
    if (ev.n_min != null) meta.push(FP_APP.t("stage_minutiae") + ": " + ev.n_min);
    if (ev.quality_score != null) meta.push(FP_APP.t("quality_score_label") + ": " + ev.quality_score);
    if (ev.n_match != null) meta.push(FP_APP.t("match_lines_count") + ": " + ev.n_match);
    if (meta.length) cap += " — " + meta.join(" · ");

    fig.innerHTML = '<img src="' + ev.src + '" alt="" loading="lazy"><figcaption>' + FP_APP.esc(cap) + '</figcaption>';
    col.appendChild(fig);
    FP_APP.setColStatus(branch, "");
  };

  FP_APP.updateManualEditLinks = function (ids) {
    var container = FP_APP.$("manual-edit-container");
    if (!container || !ids) return;

    // Ensure the container itself is visible
    container.style.display = "block";

    var html = '<div class="card" id="manual-edit-card" style="margin-top: 1rem; background: var(--glass); border: 1px dashed var(--accent); padding: 1rem;">';
    html += '<h4>' + FP_APP.esc(FP_APP.t("manual_edit_btn")) + '</h4>';
    html += '<p class="small" style="margin-bottom: 1rem;">' + FP_APP.esc(FP_APP.t("manual_edit_hint")) + '</p>';
    html += '<div style="display: flex; gap: 0.5rem;">';
    
    var bothIds = '&original_id=' + (ids.db_original_id || '') + '&partial_id=' + (ids.db_partial_id || '');
    if (ids.db_original_id) {
      html += '<a href="/editor?fingerprint_id=' + ids.db_original_id + bothIds + '&lang=' + FP_APP.TRANS.lang_code + '" target="_blank" class="btn small" style="flex: 1; font-size: 0.85rem; text-decoration: none; text-align: center; background: var(--success);">' + FP_APP.esc(FP_APP.t("original_fp")) + '</a>';
    }
    if (ids.db_partial_id) {
      html += '<a href="/editor?fingerprint_id=' + ids.db_partial_id + bothIds + '&lang=' + FP_APP.TRANS.lang_code + '" target="_blank" class="btn small" style="flex: 1; font-size: 0.85rem; text-decoration: none; text-align: center; background: var(--success);">' + FP_APP.esc(FP_APP.t("partial_fp")) + '</a>';
    }
    html += '</div>';

    if (ids.db_original_id && ids.db_partial_id) {
      html += '<div style="margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem;">';
      html += '<button type="button" class="btn" style="width: 100%; background: var(--accent); color: white;" onclick="FP_APP.reCompareSaved(' + ids.db_original_id + ', ' + ids.db_partial_id + ')">' + FP_APP.esc(FP_APP.t("re_compare_btn")) + '</button>';
      html += '<p class="small" style="margin-top: 0.5rem; text-align: center; color: var(--muted);">' + FP_APP.esc(FP_APP.t("re_compare_hint")) + '</p>';
      html += '</div>';
    }
    html += '</div>';
    
    container.innerHTML = html;

    // Update the sidebar permanent button with both fingerprint IDs
    var sidebarBtn = FP_APP.$("sidebar-editor-btn");
    if (sidebarBtn) {
      var fpId = ids.db_original_id || ids.db_partial_id;
      if (fpId) {
        sidebarBtn.href = '/editor?fingerprint_id=' + fpId
          + '&original_id=' + (ids.db_original_id || '')
          + '&partial_id=' + (ids.db_partial_id || '')
          + '&lang=' + FP_APP.TRANS.lang_code;
        sidebarBtn.style.border = '2px solid var(--success)';
        sidebarBtn.style.color = 'var(--success)';
      }
    }
  };

  FP_APP.renderDone = function (data) {
    var m = data.match;
    if (!m) return;
    var wrap = FP_APP.$("live-match-wrap");
    if (!wrap) return;

    var html = '<div class="card">';
    if (m.orb_visualization) {
      html += '<figure style="margin:0 0 1rem"><figcaption>ORB Visual Verification</figcaption><img src="' + m.orb_visualization + '" alt="ORB" style="width:100%;border-radius:8px;border:1px solid var(--border)" /></figure>';
    }

    html += '<div class="results-summary">';
    
    // Verdict & Scores
    if (m.combined_verdict) {
      html += '<div class="decision ' + (m.combined_color || "no") + '">' + FP_APP.esc(FP_APP.t("combined_verdict")) + ": " + FP_APP.esc(m.combined_verdict) + '</div>';
      if (m.fused_score != null) html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("fused_score")) + '</span><strong>' + Number(m.fused_score).toFixed(2) + '%</strong></div>';
      if (m.fusion_components) {
        html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("fusion_min")) + '</span><strong>' + Number(m.fusion_components.minutiae_score || 0).toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("fusion_mcc")) + '</span><strong>' + Number(m.fusion_components.mcc_score || 0).toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("fusion_orb")) + '</span><strong>' + Number(m.fusion_components.orb_score || 0).toFixed(2) + '%</strong></div>';
        if (m.fusion_components.landmark_score != null) {
          html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("fusion_landmarks")) + '</span><strong>' + Number(m.fusion_components.landmark_score).toFixed(2) + '%</strong></div>';
        }
      }
      html += '<hr style="margin: 1rem 0; opacity: 0.1;">';
    }

    // Stats
    html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("points_original")) + '</span><strong>' + FP_APP.esc(m.total_original) + "</strong></div>";
    html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("points_partial")) + '</span><strong>' + FP_APP.esc(m.total_partial) + "</strong></div>";
    html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("matched_points")) + '</span><strong>' + FP_APP.esc(m.matched_points) + "</strong></div>";
    html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("similarity_ratio")) + '</span><strong>' + (typeof m.match_score === "number" ? m.match_score.toFixed(2) : FP_APP.esc(m.match_score)) + "%</strong></div>";
    
    if (m.orb_confidence) {
        html += '<div class="row"><span>ORB Matches</span><strong>' + FP_APP.esc(m.orb_matches) + '</strong></div>';
        html += '<div class="row"><span>ORB Confidence</span><strong>' + FP_APP.esc(m.orb_confidence) + '</strong></div>';
    }
    
    if (m.mcc_score != null) {
        html += '<div class="row"><span>MCC Similarity</span><strong>' + m.mcc_score.toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>MCC Pairs</span><strong>' + FP_APP.esc(m.mcc_matches) + '</strong></div>';
    }

    if (m.quality_score != null) html += '<div class="row"><span>Forensic Quality Score</span><strong>' + m.quality_score.toFixed(1) + '/100</strong></div>';
    if (m.score_explanation_ar) html += '<p class="forensic-note" style="margin-top:0.35rem;font-size:0.86rem;">' + FP_APP.esc(m.score_explanation_ar) + "</p>";
    
    html += '<div class="row"><span>' + FP_APP.esc(FP_APP.t("dice_coefficient")) + '</span><strong>' + (typeof m.dice_score === "number" ? m.dice_score.toFixed(2) : FP_APP.esc(m.dice_score)) + "%</strong></div>";
    
    if (m.baseline_matched != null) {
      html += '<div class="row"><span>تطابقات قبل المحاذاة</span><strong>' + FP_APP.esc(m.baseline_matched) + "</strong></div>";
      html += '<div class="row"><span>نسبة قبل المحاذاة</span><strong>' + (typeof m.baseline_match_score === "number" ? m.baseline_match_score.toFixed(2) : FP_APP.esc(m.baseline_match_score)) + "%</strong></div>";
    }
    if (m.alignment_summary_ar) html += '<p class="forensic-note" style="margin-top:0.5rem;">' + FP_APP.esc(m.alignment_summary_ar) + "</p>";
    
    html += '<div class="decision ' + (m.status ? m.status.toLowerCase() : "no") + '">' + FP_APP.esc(FP_APP.t("technical_classification")) + ": " + FP_APP.esc(m.forensic_tier_ar || m.status) + "</div>";
    if (m.forensic_standard_note) html += '<p class="forensic-note">' + FP_APP.esc(m.forensic_standard_note) + "</p>";

    // Report Download Links
    if (data.report_download) {
      var uiLang = (FP_APP.$("fp-analyze-form")?.getAttribute("data-lang")) || "ar";
      var rp = encodeURIComponent(data.report_download);
      var langQ = "lang=" + encodeURIComponent(uiLang);
      html += '<div style="display:flex;gap:1rem;margin-top:1.5rem;flex-wrap:wrap;">';
      if (uiLang === "ar") {
        html += '<a class="dl" style="flex:1;background:var(--accent);color:white;" href="/download-report/' + rp + '?download=1&' + langQ + '" target="_blank">' + FP_APP.esc(FP_APP.t("download_report_file")) + "</a>";
        html += '<a class="dl" style="flex:1" href="/download-report/' + rp + '?' + langQ + '" target="_blank">' + FP_APP.esc(FP_APP.t("download_report")) + "</a>";
      } else {
        html += '<a class="dl" style="flex:1" href="/download-report/' + rp + '?' + langQ + '" target="_blank">' + FP_APP.esc(FP_APP.t("download_report")) + "</a>";
        html += '<a class="dl" style="flex:1;background:var(--accent);color:white;" href="/download-report/' + rp + '?format=pdf&download=1&' + langQ + '" target="_blank">' + FP_APP.esc(FP_APP.t("download_pdf")) + "</a>";
      }
      html += "</div>";
    }
    html += "</div></div>";
    wrap.innerHTML = html;
  };

  FP_APP.resetLiveColumns = function () {
    ["ref", "partial", "match"].forEach(function (b) {
      var g = FP_APP.$("live-" + b + "-gallery");
      if (g) g.innerHTML = "";
    });
    var w = FP_APP.$("live-match-wrap");
    if (w) w.innerHTML = "";
    var log = FP_APP.$("live-log");
    if (log) log.textContent = "";
    // Note: manual-edit-container is kept persistent intentionally
  };

  window.FP_APP = FP_APP;
})(window);
