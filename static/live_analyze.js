(function () {
  "use strict";

  var FP_TRANS = {};
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

  function loadTrans() {
    var el = document.getElementById("fp-trans-json");
    if (!el || !el.textContent) return;
    try {
      FP_TRANS = JSON.parse(el.textContent);
    } catch (e) {
      console.warn("fp i18n", e);
    }
  }

  function t(key, fallback) {
    if (FP_TRANS && FP_TRANS[key] != null) return FP_TRANS[key];
    return fallback != null ? fallback : key;
  }

  function stageCaption(stage) {
    var k = STAGE_KEYS[stage];
    return k ? t(k, stage) : stage;
  }

  function $(id) {
    return document.getElementById(id);
  }

  function esc(s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  }

  function appendLog(line) {
    var el = $("live-log");
    if (!el) return;
    el.textContent += (el.textContent ? "\n" : "") + line;
    el.scrollTop = el.scrollHeight;
  }

  function clearAlerts() {
    var a = $("live-alerts");
    if (a) a.innerHTML = "";
  }

  function alertBox(className, html) {
    var a = $("live-alerts");
    if (!a) return;
    var d = document.createElement("div");
    d.className = "alert " + className;
    d.innerHTML = html;
    a.appendChild(d);
  }

  function colForBranch(branch) {
    if (branch === "reference") return $("live-col-ref");
    if (branch === "partial") return $("live-col-partial");
    if (branch === "match" || branch === "orb") return $("live-match-gallery");
    return null;
  }

  function setColStatus(branch, text) {
    var id =
      branch === "reference"
        ? "live-status-ref"
        : branch === "partial"
          ? "live-status-partial"
          : branch === "match" || branch === "orb"
            ? "live-status-match"
            : null;
    var el = id ? $(id) : null;
    if (el) el.textContent = text;
  }

  function buildCaption(stage, meta) {
    var cap = stageCaption(stage);
    var parts = [];
    if (meta && meta.white != null) parts.push("بيض: " + meta.white);
    if (meta && meta.n_min != null) parts.push(t("stage_minutiae", "Minutiae") + ": " + meta.n_min);
    if (meta && meta.quality_score != null) {
      parts.push(t("quality_score_label", "Quality") + ": " + meta.quality_score);
    }
    if (meta && meta.n_match != null) {
      parts.push(t("match_lines_count", "Pairs") + ": " + meta.n_match);
    }
    if (meta && meta.extra) parts.push(meta.extra);
    if (parts.length) cap += " — " + parts.join(" · ");
    return cap;
  }

  function appendImage(branch, stage, src, meta) {
    var col = colForBranch(branch);
    if (!col || !src) return;
    meta = meta || {};
    var isMatch = branch === "match" || branch === "orb";
    var fig = document.createElement("figure");
    fig.className = isMatch
      ? "live-fig live-fig--match" + (meta.featured ? " live-fig--featured" : "")
      : "live-fig live-fig--pipe";
    var img = document.createElement("img");
    img.src = src;
    img.alt = "";
    img.loading = "lazy";
    var fc = document.createElement("figcaption");
    fc.textContent = buildCaption(stage, meta);
    fig.appendChild(img);
    fig.appendChild(fc);
    col.appendChild(fig);
    if (branch === "reference") setColStatus("reference", "");
    if (branch === "partial") setColStatus("partial", "");
    if (isMatch) setColStatus("match", "");
  }

  function decisionClass(status) {
    var st = status || "";
    if (st.indexOf("HIGH") >= 0) return "high";
    if (st.indexOf("MEDIUM") >= 0) return "medium";
    if (st.indexOf("LOW") >= 0) return "low";
    return "no";
  }

  function renderDone(data) {
    var m = data.match;
    if (!m) return;
    var wrap = $("live-match-wrap");
    if (!wrap) return;
    var st = m.status || "";
    var tier = m.forensic_tier_ar || st;
    var note = m.forensic_standard_note || "";
    var alignNote = m.alignment_summary_ar || "";
    var audit = data.audit || {};
    var shaO = audit.sha256_original || "";
    var shaP = audit.sha256_partial || "";
    var shortSha = function (h) {
      return h.length > 24 ? h.slice(0, 24) + "…" : h;
    };

    var html = "";
    html += '<div class="card">';
    if (m.orb_visualization) {
      html += '<figure style="margin:0 0 1rem"><figcaption>ORB Visual Verification</figcaption><img src="' + m.orb_visualization + '" alt="ORB" style="width:100%;border-radius:8px;border:1px solid var(--border)" /></figure>';
    }
    html += '<div class="results-summary">';
    
    if (m.combined_verdict) {
      html += '<div class="decision ' + (m.combined_color || "no") + '">' + esc(t("combined_verdict")) + ": " + esc(m.combined_verdict) + '</div>';
      if (m.fused_score != null) {
        html += '<div class="row"><span>' + esc(t("fused_score")) + '</span><strong>' + Number(m.fused_score).toFixed(2) + '%</strong></div>';
      }
      if (m.fusion_components) {
        html += '<div class="row"><span>' + esc(t("fusion_min")) + '</span><strong>' + Number(m.fusion_components.minutiae_score || 0).toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>' + esc(t("fusion_mcc")) + '</span><strong>' + Number(m.fusion_components.mcc_score || 0).toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>' + esc(t("fusion_orb")) + '</span><strong>' + Number(m.fusion_components.orb_score || 0).toFixed(2) + '%</strong></div>';
      }
      html += '<hr style="margin: 1rem 0; opacity: 0.1;">';
    }

    html +=
      '<div class="row"><span>' + esc(t("points_original")) + '</span><strong>' +
      esc(m.total_original) +
      "</strong></div>";
    html +=
      '<div class="row"><span>' + esc(t("points_partial")) + '</span><strong>' +
      esc(m.total_partial) +
      "</strong></div>";
    html +=
      '<div class="row"><span>' + esc(t("matched_points")) + '</span><strong>' +
      esc(m.matched_points) +
      "</strong></div>";
    html +=
      '<div class="row"><span>' + esc(t("similarity_ratio")) + '</span><strong>' +
      (typeof m.match_score === "number" ? m.match_score.toFixed(2) : esc(m.match_score)) +
      "%</strong></div>";
    
    if (m.orb_confidence) {
        html += '<div class="row"><span>ORB Matches</span><strong>' + esc(m.orb_matches) + '</strong></div>';
        html += '<div class="row"><span>ORB Confidence</span><strong>' + esc(m.orb_confidence) + '</strong></div>';
    }
    
    if (m.mcc_score != null) {
        html += '<div class="row"><span>MCC Similarity</span><strong>' + m.mcc_score.toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>MCC Pairs</span><strong>' + esc(m.mcc_matches) + '</strong></div>';
    }

    if (m.quality_score != null) {
        html += '<div class="row"><span>Forensic Quality Score</span><strong>' + m.quality_score.toFixed(1) + '/100</strong></div>';
    }

    if (m.score_explanation_ar) {
      html +=
        '<p class="forensic-note" style="margin-top:0.35rem;font-size:0.86rem;">' +
        esc(m.score_explanation_ar) +
        "</p>";
    }
    html +=
      '<div class="row"><span>' + esc(t("dice_coefficient")) + '</span><strong>' +
      (typeof m.dice_score === "number" ? m.dice_score.toFixed(2) : esc(m.dice_score)) +
      "%</strong></div>";
    if (m.baseline_matched != null) {
      html +=
        '<div class="row"><span>تطابقات قبل المحاذاة</span><strong>' +
        esc(m.baseline_matched) +
        "</strong></div>";
      html +=
        '<div class="row"><span>نسبة قبل المحاذاة</span><strong>' +
        (typeof m.baseline_match_score === "number"
          ? m.baseline_match_score.toFixed(2)
          : esc(m.baseline_match_score)) +
        "%</strong></div>";
    }
    if (alignNote) {
      html +=
        '<p class="forensic-note" style="margin-top:0.5rem;">' +
        esc(alignNote) +
        "</p>";
    }
    html +=
      '<div class="decision ' +
      decisionClass(st) +
      '">' + esc(t("technical_classification")) + ": " +
      esc(tier) +
      "</div>";
    if (note) {
      html += '<p class="forensic-note">' + esc(note) + "</p>";
    }
    if (shaO || shaP) {
      html += '<div class="audit-inline">';
      if (shaO) {
        html +=
          "<div><strong>SHA-256 مرجعية:</strong> <code>" +
          esc(shortSha(shaO)) +
          "</code></div>";
      }
      if (shaP) {
        html +=
          "<div><strong>SHA-256 مقارنة:</strong> <code>" +
          esc(shortSha(shaP)) +
          "</code></div>";
      }
      html += "</div>";
    }
    html += "</div>";
    if (data.ref_grid) {
      html +=
        '<p class="forensic-note" style="margin-top:0.5rem;font-size:0.88rem;">' +
        esc(t("ref_grid_selected")) +
        " (مرجعية): " +
        esc(data.ref_grid.ref_region_label || "") +
        "<br/>" +
        esc(t("ref_grid_selected")) +
        " (مقارنة): " +
        esc(data.ref_grid.partial_region_label || "") +
        "</p>";
    }
    if (data.report_download) {
      var formEl = $("fp-analyze-form");
      var uiLang = (formEl && formEl.getAttribute("data-lang")) || "ar";
      var rp = encodeURIComponent(data.report_download);
      var langQ = "&lang=" + encodeURIComponent(uiLang);
      html += '<div style="display:flex;gap:1rem;margin-top:1.5rem;flex-wrap:wrap;">';
      if (uiLang === "ar") {
        html +=
          '<a class="dl" style="flex:1;background:var(--accent);color:white;" href="/download-report/' +
          rp +
          "?download=1" +
          langQ +
          '" target="_blank" rel="noopener noreferrer">' +
          esc(t("download_report_file")) +
          "</a>";
        html +=
          '<a class="dl" style="flex:1" href="/download-report/' +
          rp +
          langQ +
          '" target="_blank" rel="noopener noreferrer">' +
          esc(t("download_report")) +
          "</a>";
        html +=
          '<p class="small" style="width:100%;margin:0.35rem 0 0;color:var(--muted);">' +
          esc(t("download_pdf_unavailable_ar")) +
          "</p>";
      } else {
        html +=
          '<a class="dl" style="flex:1" href="/download-report/' +
          rp +
          langQ +
          '" target="_blank" rel="noopener noreferrer">' +
          esc(t("download_report")) +
          "</a>";
        html +=
          '<a class="dl" style="flex:1;background:var(--accent);color:white;" href="/download-report/' +
          rp +
          "?format=pdf&download=1" +
          langQ +
          '" target="_blank" rel="noopener noreferrer">' +
          esc(t("download_pdf")) +
          "</a>";
      }
      html += "</div>";
    }
    html += "</div>";
    wrap.innerHTML = html;
  }

  function resetLiveColumns() {
    ["live-col-ref", "live-col-partial"].forEach(function (id) {
      var el = $(id);
      if (!el) return;
      var h = el.querySelector("h3");
      var st = el.querySelector(".live-col-status");
      el.querySelectorAll(".live-fig").forEach(function (n) {
        n.remove();
      });
      if (!h) {
        h = document.createElement("h3");
        el.insertBefore(h, el.firstChild);
      }
      h.textContent =
        id === "live-col-ref"
          ? t("original_fp") + " " + t("live_suffix")
          : t("partial_fp") + " " + t("live_suffix");
      if (!st) {
        st = document.createElement("p");
        st.className = "live-col-status small";
        el.insertBefore(st, h.nextSibling);
      }
      st.textContent = t("live_processing", "Processing…");
    });
    var gallery = $("live-match-gallery");
    if (gallery) gallery.innerHTML = "";
    var stMatch = $("live-status-match");
    if (stMatch) stMatch.textContent = t("live_processing", "Processing…");
    var w = $("live-match-wrap");
    if (w) w.innerHTML = "";
    var log = $("live-log");
    if (log) log.textContent = "";
  }

  var activeRun = {
    running: false,
    abort: null,
    reader: null,
    generation: 0,
  };

  function finishRun(wasCancelled, runGeneration) {
    if (runGeneration != null && runGeneration !== activeRun.generation) {
      return;
    }
    activeRun.running = false;
    activeRun.abort = null;
    activeRun.reader = null;
    var btn = $("fp-submit-btn");
    var panel = $("live-panel");
    if (btn) {
      btn.type = "submit";
      btn.classList.remove("btn--stop");
      btn.disabled = false;
      btn.textContent = btn.dataset.label || t("analyze_btn");
      btn.removeAttribute("aria-pressed");
    }
    if (panel) panel.setAttribute("aria-busy", "false");
    if (wasCancelled) {
      appendLog(t("cancelled_log", "— Stopped —"));
      alertBox("warn", t("analysis_stopped", "Analysis stopped."));
      ["reference", "partial", "match"].forEach(function (b) {
        setColStatus(b, "");
      });
    }
  }

  function setRunActive(active) {
    var btn = $("fp-submit-btn");
    if (!btn) return;
    if (!btn.dataset.label) {
      btn.dataset.label = btn.textContent || t("analyze_btn");
    }
    if (active) {
      btn.type = "button";
      btn.classList.add("btn--stop");
      btn.disabled = false;
      btn.textContent = t("stop_analysis", "Stop");
      btn.setAttribute("aria-pressed", "true");
    } else {
      btn.type = "submit";
      btn.classList.remove("btn--stop");
      btn.textContent = btn.dataset.label || t("analyze_btn");
      btn.removeAttribute("aria-pressed");
    }
  }

  function stopActiveRun() {
    if (!activeRun.running) return;
    activeRun.generation += 1;
    activeRun.running = false;
    if (activeRun.abort) {
      try {
        activeRun.abort.abort();
      } catch (e) {}
    }
    if (activeRun.reader) {
      activeRun.reader.cancel().catch(function () {});
    }
    finishRun(true);
  }

  function parseSseBuffer(buffer, onEvent) {
    var parts = buffer.split("\n\n");
    var rest = parts.pop() || "";
    for (var i = 0; i < parts.length; i++) {
      var block = parts[i];
      var lines = block.split("\n");
      for (var j = 0; j < lines.length; j++) {
        var line = lines[j];
        if (line.indexOf("data: ") === 0) {
          try {
            onEvent(JSON.parse(line.slice(6)));
          } catch (e) {
            console.warn("SSE parse", e);
          }
        }
      }
    }
    return rest;
  }

  async function runStream(form) {
    if (activeRun.running) return;

    var runGeneration = ++activeRun.generation;

    var panel = $("live-panel");
    var serverBlock = $("server-results");
    if (panel) {
      panel.hidden = false;
      panel.setAttribute("aria-busy", "true");
    }
    if (serverBlock) serverBlock.hidden = true;
    clearAlerts();
    resetLiveColumns();

    activeRun.running = true;
    var abort = new AbortController();
    activeRun.abort = abort;
    setRunActive(true);

    var fd = new FormData(form);
    var uiLang = form.getAttribute("data-lang") || "ar";
    var res;
    var cancelled = false;
    try {
      res = await fetch("/analyze-stream?lang=" + encodeURIComponent(uiLang), {
        method: "POST",
        body: fd,
        headers: { Accept: "text/event-stream" },
        signal: abort.signal,
      });
    } catch (e) {
      if (e && e.name === "AbortError") {
        cancelled = true;
      } else {
        alertBox("err", t("server_error") + ": " + esc(e.message));
      }
      finishRun(cancelled, runGeneration);
      return;
    }

    if (!activeRun.running || runGeneration !== activeRun.generation) {
      finishRun(true, runGeneration);
      return;
    }

    if (!res.ok || !res.body) {
      alertBox("err", t("unexpected_response") + " (" + res.status + ").");
      finishRun(false, runGeneration);
      return;
    }

    function handleEvent(ev) {
      if (!ev || !ev.type) return;
      if (ev.type === "log") {
        appendLog(ev.message || "");
        return;
      }
      if (ev.type === "hashes" && ev.same_file_warning) {
        alertBox("warn", t("same_file_sse"));
        return;
      }
      if (ev.type === "image") {
        appendImage(ev.branch, ev.stage, ev.src, {
          white: ev.white,
          n_min: ev.n_min,
          quality_score: ev.quality_score,
          n_match: ev.n_match,
          featured: ev.featured,
        });
        return;
      }
      if (ev.type === "fatal") {
        alertBox("err", esc(ev.message || "خطأ"));
        if (ev.forensic_quality_warning) {
          alertBox("warn", t("quality_sse"));
        }
        return;
      }
      if (ev.type === "done") {
        if (ev.forensic_quality_warning) {
          alertBox("warn", t("quality_warning"));
        }
        renderDone(ev);
        appendLog(t("done_log"));
        return;
      }
    }

    var reader = res.body.getReader();
    activeRun.reader = reader;
    var dec = new TextDecoder();
    var buf = "";

    try {
      while (activeRun.running) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += dec.decode(chunk.value, { stream: true });
        buf = parseSseBuffer(buf, handleEvent);
      }
      if (activeRun.running) {
        buf = parseSseBuffer(buf + "\n\n", handleEvent);
      } else {
        cancelled = true;
      }
    } catch (e) {
      if (e && e.name === "AbortError") {
        cancelled = true;
      } else if (!activeRun.running) {
        cancelled = true;
      } else {
        alertBox("err", t("server_error") + ": " + esc(e.message));
      }
    } finally {
      try {
        reader.releaseLock();
      } catch (e2) {}
    }

    finishRun(cancelled);
  }

  function init() {
    loadTrans();
    var form = $("fp-analyze-form");
    if (!form) return;

    function bindPreview(fileInputId, imgId, emptyId, zoomId, zoomValId, shiftXId, shiftXValId, shiftYId, shiftYValId, resetBtnId, hostId) {
      var fileInput = $(fileInputId);
      var img = $(imgId);
      var empty = $(emptyId);
      var host = hostId ? $(hostId) : null;
      var zoom = $(zoomId);
      var zoomVal = $(zoomValId);
      var shiftX = shiftXId ? $(shiftXId) : null;
      var shiftXVal = shiftXValId ? $(shiftXValId) : null;
      var shiftY = shiftYId ? $(shiftYId) : null;
      var shiftYVal = shiftYValId ? $(shiftYValId) : null;
      var resetBtn = resetBtnId ? $(resetBtnId) : null;
      if (!fileInput || !img || !empty || !zoom || !zoomVal) return;

      function applyTransform() {
        var scale = Math.max(50, Math.min(250, Number(zoom.value) || 100));
        var tx = shiftX ? Math.max(-300, Math.min(300, Number(shiftX.value) || 0)) : 0;
        var ty = shiftY ? Math.max(-300, Math.min(300, Number(shiftY.value) || 0)) : 0;
        zoomVal.textContent = String(scale) + "%";
        if (shiftXVal) shiftXVal.textContent = String(tx) + " px";
        if (shiftYVal) shiftYVal.textContent = String(ty) + " px";
        var tr =
          "translate(" + tx + "px, " + ty + "px) scale(" + (scale / 100).toFixed(2) + ")";
        img.style.transform = tr;
        img.style.transformOrigin = "center center";
        if (host) {
          host.dispatchEvent(new CustomEvent("fp-preview-transform", { bubbles: false }));
        }
      }

      zoom.addEventListener("input", applyTransform);
      if (shiftX) shiftX.addEventListener("input", applyTransform);
      if (shiftY) shiftY.addEventListener("input", applyTransform);
      if (resetBtn) {
        resetBtn.addEventListener("click", function () {
          zoom.value = "100";
          if (shiftX) shiftX.value = "0";
          if (shiftY) shiftY.value = "0";
          applyTransform();
        });
      }
      applyTransform();

      fileInput.addEventListener("change", function () {
        var file = fileInput.files && fileInput.files[0];
        if (!file) {
          img.hidden = true;
          img.removeAttribute("src");
          empty.hidden = false;
          return;
        }

        var reader = new FileReader();
        reader.onload = function (ev) {
          img.src = ev.target && ev.target.result ? String(ev.target.result) : "";
          img.hidden = false;
          empty.hidden = true;
          applyTransform();
        };
        reader.onerror = function () {
          img.hidden = true;
          empty.hidden = false;
        };
        reader.readAsDataURL(file);
      });
    }

    if (typeof window.initRegionSelector === "function") {
      window.initRegionSelector(
        {
          hostId: "original-preview-host",
          imgId: "original-preview-img",
          modeId: "ref_region_mode",
          divisionsId: "ref_grid_divisions",
          cellsId: "ref_grid_cells",
          regionId: "ref_region",
          labelId: "ref-grid-selected-label",
          gridOverlayId: "ref-grid-overlay",
          rectLayerId: "ref-rect-layer",
          selectionBoxId: "ref-selection-box",
          divisionsWrapId: "ref-grid-divisions-wrap",
        },
        t
      );
      window.initRegionSelector(
        {
          hostId: "partial-preview-host",
          imgId: "partial-preview-img",
          modeId: "partial_region_mode",
          divisionsId: "partial_grid_divisions",
          cellsId: "partial_grid_cells",
          regionId: "partial_region",
          labelId: "partial-grid-selected-label",
          gridOverlayId: "partial-grid-overlay",
          rectLayerId: "partial-rect-layer",
          selectionBoxId: "partial-selection-box",
          divisionsWrapId: "partial-grid-divisions-wrap",
        },
        t
      );
    }

    bindPreview(
      "original",
      "original-preview-img",
      "original-preview-empty",
      "original-zoom",
      "original-zoom-val",
      null,
      null,
      null,
      null,
      null,
      "original-preview-host"
    );
    bindPreview(
      "partial",
      "partial-preview-img",
      "partial-preview-empty",
      "partial-zoom",
      "partial-zoom-val",
      "partial-shift-x",
      "partial-shift-x-val",
      "partial-shift-y",
      "partial-shift-y-val",
      "partial-reset-transform",
      "partial-preview-host"
    );

    var autoSweepBtn = $("partial-auto-sweep");
    var autoSweepWideBtn = $("partial-auto-sweep-wide");
    var sweepStatus = $("partial-sweep-status");

    async function runSweep(mode) {
        var orig = $("original");
        var part = $("partial");
        if (!orig || !part || !orig.files || !part.files || !orig.files[0] || !part.files[0]) {
          if (sweepStatus) sweepStatus.textContent = t("sweep_need_files");
          return;
        }
        if (autoSweepBtn) autoSweepBtn.disabled = true;
        if (autoSweepWideBtn) autoSweepWideBtn.disabled = true;
        if (sweepStatus) {
          sweepStatus.textContent = mode === "wide"
            ? t("sweep_running_wide")
            : t("sweep_running_quick");
        }
        try {
          var fd = new FormData(form);
          fd.set("sweep_mode", mode);
          var res = await fetch("/analyze-sweep", { method: "POST", body: fd });
          var data = await res.json();
          if (!res.ok || !data.ok || !data.best) {
            if (sweepStatus) sweepStatus.textContent = (data && data.message) ? data.message : t("sweep_failed");
            return;
          }

          var pz = $("partial-zoom");
          var sx = $("partial-shift-x");
          var sy = $("partial-shift-y");
          if (pz) {
            pz.value = String(data.best.partial_zoom);
            pz.dispatchEvent(new Event("input", { bubbles: true }));
          }
          if (sx) {
            sx.value = String(data.best.partial_shift_x);
            sx.dispatchEvent(new Event("input", { bubbles: true }));
          }
          if (sy) {
            sy.value = String(data.best.partial_shift_y);
            sy.dispatchEvent(new Event("input", { bubbles: true }));
          }

          if (sweepStatus) {
            sweepStatus.textContent =
              t("sweep_done") + ": Zoom " +
              data.best.partial_zoom +
              "%, X " +
              data.best.partial_shift_x +
              ", Y " +
              data.best.partial_shift_y +
              " (" + (data.mode || mode) + ", score " +
              data.best.objective_score +
              ").";
          }
        } catch (err) {
          if (sweepStatus) sweepStatus.textContent = t("sweep_error") + ": " + (err && err.message ? err.message : String(err));
        } finally {
          if (autoSweepBtn) autoSweepBtn.disabled = false;
          if (autoSweepWideBtn) autoSweepWideBtn.disabled = false;
        }
    }

    if (autoSweepBtn) {
      autoSweepBtn.addEventListener("click", function () {
        runSweep("quick");
      });
    }
    if (autoSweepWideBtn) {
      autoSweepWideBtn.addEventListener("click", function () {
        runSweep("wide");
      });
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      if (activeRun.running) {
        stopActiveRun();
        return;
      }
      runStream(form);
    });

    var submitBtn = $("fp-submit-btn");
    if (submitBtn) {
      submitBtn.addEventListener("click", function (e) {
        if (activeRun.running) {
          e.preventDefault();
          stopActiveRun();
        }
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
