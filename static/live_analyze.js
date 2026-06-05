(function (window) {
  "use strict";

  var FP_APP = window.FP_APP || {};

  // --- Re-compare functionality ---
  FP_APP.reCompareSaved = function(idO, idP) {
    if (!confirm(FP_APP.t("re_compare_btn") + "?")) return;
    FP_APP.appendLog("Re-calculating match with manual edits...");
    fetch('/re-compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ original_id: idO, partial_id: idP, lang: FP_APP.TRANS.lang_code })
    })
    .then(r => r.json())
    .then(data => {
        if (data.status === 'success') {
            alert("Match re-calculated successfully!");
            FP_APP.renderDone(data);
        } else alert("Error: " + (data.detail || "Unknown error"));
    })
    .catch(e => alert("Failed to connect to server"));
  };

  // --- Initialization ---
  function init() {
    FP_APP.loadTrans();
    var form = FP_APP.$("fp-analyze-form");
    if (!form) return;

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      if (FP_APP.activeRun.running) FP_APP.stopActiveRun(); 
      else FP_APP.runStream(form);
    });

    // Re-bind previews (simplified for refactor)
    if (typeof window.initRegionSelector === "function") {
        var t_wrapper = FP_APP.t;
        // Original
        window.initRegionSelector({
          hostId: "original-preview-host", imgId: "original-preview-img", modeId: "ref_region_mode",
          divisionsId: "ref_grid_divisions", cellsId: "ref_grid_cells", regionId: "ref_region",
          labelId: "ref-grid-selected-label", gridOverlayId: "ref-grid-overlay",
          rectLayerId: "ref-rect-layer", selectionBoxId: "ref-selection-box", divisionsWrapId: "ref-grid-divisions-wrap"
        }, t_wrapper);
        // Partial
        window.initRegionSelector({
          hostId: "partial-preview-host", imgId: "partial-preview-img", modeId: "partial_region_mode",
          divisionsId: "partial_grid_divisions", cellsId: "partial_grid_cells", regionId: "partial_region",
          labelId: "partial-grid-selected-label", gridOverlayId: "partial-grid-overlay",
          rectLayerId: "partial-rect-layer", selectionBoxId: "partial-selection-box", divisionsWrapId: "partial-grid-divisions-wrap"
        }, t_wrapper);
    }
    
    // Bind generic preview logic (zoom/shift)
    function bindP(prefix) {
        var zoom = FP_APP.$(prefix + "-zoom"), img = FP_APP.$(prefix + "-preview-img"), val = FP_APP.$(prefix + "-zoom-val");
        var sx = FP_APP.$(prefix + "-shift-x"), sy = FP_APP.$(prefix + "-shift-y");
        var sxV = FP_APP.$(prefix + "-shift-x-val"), syV = FP_APP.$(prefix + "-shift-y-val");
        if (!zoom || !img) return;
        
        var update = function() {
            var z = zoom.value, x = sx ? sx.value : 0, y = sy ? sy.value : 0;
            if (val) val.textContent = z + "%";
            if (sxV) sxV.textContent = x + " px";
            if (syV) syV.textContent = y + " px";
            img.style.transform = "translate(" + x + "px," + y + "px) scale(" + (z/100) + ")";
        };
        
        [zoom, sx, sy].forEach(el => el && el.addEventListener("input", update));
        
        FP_APP.$(prefix).addEventListener("change", function() {
            var f = this.files[0];
            if (f) { 
                var r = new FileReader(); 
                r.onload = function(e) { 
                    img.src = e.target.result; 
                    img.hidden = false; 
                    FP_APP.$(prefix + "-preview-empty").hidden = true; 
                    update(); 
                };
                r.readAsDataURL(f);
            }
        });
        
        FP_APP.$(prefix + "-reset-transform")?.addEventListener("click", function() {
            zoom.value = 100; if (sx) sx.value = 0; if (sy) sy.value = 0; update();
        });
    }
    bindP("original"); bindP("partial");

    // Auto-sweep buttons
    async function runSw(mode) {
        var btn = FP_APP.$("partial-auto-sweep" + (mode === "wide" ? "-wide" : ""));
        var st = FP_APP.$("partial-sweep-status");
        if (!btn) return;
        btn.disabled = true;
        if (st) st.textContent = "...";
        try {
            var fd = new FormData(form); fd.set("sweep_mode", mode);
            var r = await fetch("/analyze-sweep", { method: "POST", body: fd });
            var d = await r.json();
            if (d.ok && d.best) {
                FP_APP.$("partial-zoom").value = d.best.partial_zoom;
                FP_APP.$("partial-shift-x").value = d.best.partial_shift_x;
                FP_APP.$("partial-shift-y").value = d.best.partial_shift_y;
                FP_APP.$("partial-zoom").dispatchEvent(new Event("input"));
                if (st) st.textContent = "Done";
            }
        } catch (e) {
            console.error("Sweep Error:", e);
        } finally { 
            btn.disabled = false; 
        }
    }
    FP_APP.$("partial-auto-sweep")?.addEventListener("click", () => runSw("quick"));
    FP_APP.$("partial-auto-sweep-wide")?.addEventListener("click", () => runSw("wide"));
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
  
  window.FP_APP = FP_APP;
})(window);
