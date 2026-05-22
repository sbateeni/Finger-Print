(function () {
  "use strict";

  var STAGE_CAPTION = {
    processed: "ثنائية بعد المعالجة",
    quality_map: "خريطة الجودة (Heatmap)",
    singular_vis: "النقاط المفردة (مراكز ودلتات)",
    ridges: "Gabor — التموجات",
    skeleton: "Skeleton — الهيكلة",
    minutiae_vis: "النقاط الدقيقة (تصور)",
    alignment: "المطابقة على هيكل المرجعية",
    orb_vis: "التحقق البصري (ORB)",
  };

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
    if (branch === "match") return $("live-match-col");
    return null;
  }

  function appendImage(branch, stage, src, extraCaption) {
    var col = colForBranch(branch);
    if (!col || !src) return;
    var cap = STAGE_CAPTION[stage] || stage;
    if (extraCaption) cap += " — " + extraCaption;
    var fig = document.createElement("figure");
    var img = document.createElement("img");
    img.src = src;
    img.alt = "";
    img.style.maxWidth = "100%";
    img.style.borderRadius = "8px";
    img.style.border = "1px solid var(--border)";
    var fc = document.createElement("figcaption");
    fc.textContent = cap;
    fig.appendChild(img);
    fig.appendChild(fc);
    col.appendChild(fig);
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
      html += '<div class="decision ' + (m.combined_color || "no") + '">الحكم النهائي: ' + esc(m.combined_verdict) + '</div>';
      if (m.fused_score != null) {
        html += '<div class="row"><span>Fused Score</span><strong>' + Number(m.fused_score).toFixed(2) + '%</strong></div>';
      }
      if (m.fusion_components) {
        html += '<div class="row"><span>Fusion - Minutiae</span><strong>' + Number(m.fusion_components.minutiae_score || 0).toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>Fusion - MCC</span><strong>' + Number(m.fusion_components.mcc_score || 0).toFixed(2) + '%</strong></div>';
        html += '<div class="row"><span>Fusion - ORB</span><strong>' + Number(m.fusion_components.orb_score || 0).toFixed(2) + '%</strong></div>';
      }
      html += '<hr style="margin: 1rem 0; opacity: 0.1;">';
    }

    html +=
      '<div class="row"><span>نقاط الأصلية</span><strong>' +
      esc(m.total_original) +
      "</strong></div>";
    html +=
      '<div class="row"><span>نقاط الجزئية</span><strong>' +
      esc(m.total_partial) +
      "</strong></div>";
    html +=
      '<div class="row"><span>تطابقات (واحد‑لواحد)</span><strong>' +
      esc(m.matched_points) +
      "</strong></div>";
    html +=
      '<div class="row"><span>نسبة التشابه (حسب الجزئية)</span><strong>' +
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
      '<div class="row"><span>معامل Dice</span><strong>' +
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
      '">التصنيف الفني: ' +
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
    if (data.report_download) {
      html += '<div style="display:flex;gap:1rem;margin-top:1.5rem;">';
      html +=
        '<a class="dl" style="flex:1" href="/download-report/' +
        encodeURIComponent(data.report_download) +
        '" target="_blank" rel="noopener noreferrer">معاينة التقرير (تبويب جديد)</a>';
      html +=
        '<a class="dl" style="flex:1;background:var(--accent);color:white;" href="/download-report/' +
        encodeURIComponent(data.report_download) +
        '?format=pdf&download=1" target="_blank" rel="noopener noreferrer">تحميل التقرير PDF</a>';
      html += "</div>";
    }
    html += "</div>";
    wrap.innerHTML = html;
  }

  function resetLiveColumns() {
    var cols = [
      { id: "live-col-ref", title: "مسار الأصلية (مباشر)" },
      { id: "live-col-partial", title: "مسار الجزئية (مباشر)" },
      { id: "live-match-col", title: "المطابقة (مباشر)" },
    ];
    cols.forEach(function (c) {
      var el = $(c.id);
      if (!el) return;
      el.innerHTML = "";
      var h = document.createElement("h3");
      h.textContent = c.title;
      el.appendChild(h);
    });
    var w = $("live-match-wrap");
    if (w) w.innerHTML = "";
    var log = $("live-log");
    if (log) log.textContent = "";
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
    var panel = $("live-panel");
    var serverBlock = $("server-results");
    var btn = $("fp-submit-btn");
    if (panel) {
      panel.hidden = false;
      panel.setAttribute("aria-busy", "true");
    }
    if (serverBlock) serverBlock.hidden = true;
    if (btn) {
      btn.disabled = true;
      btn.dataset.label = btn.dataset.label || btn.textContent;
      btn.textContent = "جاري المعالجة (مباشر)…";
    }
    clearAlerts();
    resetLiveColumns();

    var fd = new FormData(form);
    var res;
    try {
      res = await fetch("/analyze-stream", {
        method: "POST",
        body: fd,
        headers: { Accept: "text/event-stream" },
      });
    } catch (e) {
      alertBox("err", "تعذر الاتصال بالخادم: " + esc(e.message));
      if (btn) {
        btn.disabled = false;
        btn.textContent = btn.dataset.label || "تحليل ومقارنة";
      }
      if (panel) panel.setAttribute("aria-busy", "false");
      return;
    }

    if (!res.ok || !res.body) {
      alertBox("err", "رد غير متوقع من الخادم (" + res.status + ").");
      if (btn) {
        btn.disabled = false;
        btn.textContent = btn.dataset.label || "تحليل ومقارنة";
      }
      if (panel) panel.setAttribute("aria-busy", "false");
      return;
    }

    function handleEvent(ev) {
      if (!ev || !ev.type) return;
      if (ev.type === "log") {
        appendLog(ev.message || "");
        return;
      }
      if (ev.type === "hashes" && ev.same_file_warning) {
        alertBox(
          "warn",
          "تنبيه: الملفان متطابقان بايتًا — النتيجة ستكون تطابقًا شبه كامل ولا تصلح كاختبار حقيقي."
        );
        return;
      }
      if (ev.type === "image") {
        var extra = "";
        if (ev.white != null) extra = "بيض: " + ev.white;
        if (ev.n_min != null) extra = "النقاط الدقيقة: " + ev.n_min;
        appendImage(ev.branch, ev.stage, ev.src, extra);
        return;
      }
      if (ev.type === "fatal") {
        alertBox("err", esc(ev.message || "خطأ"));
        if (ev.forensic_quality_warning) {
          alertBox(
            "warn",
            "تنبيه جودة: راجع جودة التصوير — عدد النقاط الدقيقة قد يكون منخفضًا."
          );
        }
        return;
      }
      if (ev.type === "done") {
        if (ev.forensic_quality_warning) {
          alertBox(
            "warn",
            "تنبيه جودة: عدد النقاط الدقيقة في إحدى الصورتين أقل من الحد التنبيهي — راجع جودة التصوير والإضاءة."
          );
        }
        renderDone(ev);
        appendLog("اكتمل — يمكن معاينة التقرير من الرابط أعلاه.");
        return;
      }
    }

    var reader = res.body.getReader();
    var dec = new TextDecoder();
    var buf = "";

    try {
      while (true) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += dec.decode(chunk.value, { stream: true });
        buf = parseSseBuffer(buf, handleEvent);
      }
      buf = parseSseBuffer(buf + "\n\n", handleEvent);
    } catch (e) {
      alertBox("err", "انقطع التدفق: " + esc(e.message));
    }

    if (btn) {
      btn.disabled = false;
      btn.textContent = btn.dataset.label || "تحليل ومقارنة";
    }
    if (panel) panel.setAttribute("aria-busy", "false");
  }

  function init() {
    var form = $("fp-analyze-form");
    if (!form) return;

    function bindPreview(fileInputId, imgId, emptyId, zoomId, zoomValId, shiftXId, shiftXValId, shiftYId, shiftYValId, resetBtnId) {
      var fileInput = $(fileInputId);
      var img = $(imgId);
      var empty = $(emptyId);
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
        img.style.transform = "translate(" + tx + "px, " + ty + "px) scale(" + (scale / 100).toFixed(2) + ")";
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
      null
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
      "partial-reset-transform"
    );

    var autoSweepBtn = $("partial-auto-sweep");
    var autoSweepWideBtn = $("partial-auto-sweep-wide");
    var sweepStatus = $("partial-sweep-status");

    async function runSweep(mode) {
        var orig = $("original");
        var part = $("partial");
        if (!orig || !part || !orig.files || !part.files || !orig.files[0] || !part.files[0]) {
          if (sweepStatus) sweepStatus.textContent = "يرجى رفع الصورتين أولاً.";
          return;
        }
        if (autoSweepBtn) autoSweepBtn.disabled = true;
        if (autoSweepWideBtn) autoSweepWideBtn.disabled = true;
        if (sweepStatus) {
          sweepStatus.textContent = mode === "wide"
            ? "جاري البحث الواسع (قد يستغرق وقتًا أطول)..."
            : "جاري البحث عن أفضل Zoom/Shift...";
        }
        try {
          var fd = new FormData(form);
          fd.set("sweep_mode", mode);
          var res = await fetch("/analyze-sweep", { method: "POST", body: fd });
          var data = await res.json();
          if (!res.ok || !data.ok || !data.best) {
            if (sweepStatus) sweepStatus.textContent = (data && data.message) ? data.message : "فشل Auto-sweep.";
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
              "تم اختيار أفضل إعداد: Zoom " +
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
          if (sweepStatus) sweepStatus.textContent = "خطأ أثناء Auto-sweep: " + (err && err.message ? err.message : String(err));
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
      runStream(form);
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
