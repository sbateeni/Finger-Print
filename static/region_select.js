/**
 * Grid + drag-rectangle selection on fingerprint previews.
 * Regions are 0–1 in original image pixels. Preview zoom applies to the image only.
 */
(function (global) {
  "use strict";

  function $(id) {
    return document.getElementById(id);
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function formatRegion(x, y, w, h) {
    return (
      clamp(x, 0, 1).toFixed(4) +
      "," +
      clamp(y, 0, 1).toFixed(4) +
      "," +
      clamp(w, 0.02, 1).toFixed(4) +
      "," +
      clamp(h, 0.02, 1).toFixed(4)
    );
  }

  function parseCells(s) {
    if (!s) return [];
    return s
      .split(",")
      .map(function (x) {
        return parseInt(x, 10);
      })
      .filter(function (n) {
        return !isNaN(n);
      });
  }

  function unionCellsNorm(divisions, cells) {
    var cols = divisions === 6 ? 3 : 2;
    var rows = 2;
    var xs = [],
      ys = [],
      x2 = [],
      y2 = [];
    cells.forEach(function (c) {
      if (c < 0 || c >= divisions) return;
      var col = c % cols;
      var row = Math.floor(c / cols);
      xs.push(col / cols);
      ys.push(row / rows);
      x2.push((col + 1) / cols);
      y2.push((row + 1) / rows);
    });
    if (!xs.length) return { x: 0, y: 0, w: 1, h: 1 };
    var x = Math.min.apply(null, xs);
    var y = Math.min.apply(null, ys);
    return { x: x, y: y, w: Math.max.apply(null, x2) - x, h: Math.max.apply(null, y2) - y };
  }

  function initRegionSelector(cfg, t) {
    var host = $(cfg.hostId);
    var img = $(cfg.imgId);
    var modeEl = $(cfg.modeId);
    var divisionsEl = $(cfg.divisionsId);
    var cellsInput = $(cfg.cellsId);
    var regionInput = $(cfg.regionId);
    var labelEl = $(cfg.labelId);
    var gridOverlay = $(cfg.gridOverlayId);
    var rectLayer = $(cfg.rectLayerId);
    var selectionBox = $(cfg.selectionBoxId);
    var divisionsWrap = cfg.divisionsWrapId ? $(cfg.divisionsWrapId) : null;
    if (!host || !img || !modeEl || !regionInput) return;

    var selectedCells = new Set(parseCells(cellsInput && cellsInput.value));
    var drag = null;

    function isFull(r) {
      return r.w >= 0.999 && r.h >= 0.999 && r.x <= 0.001 && r.y <= 0.001;
    }

    /** Image box in host-local pixels (tracks preview zoom on img only). */
    function imageBoxInHost() {
      var hr = host.getBoundingClientRect();
      var ir = img.getBoundingClientRect();
      return {
        left: ir.left - hr.left,
        top: ir.top - hr.top,
        width: ir.width || 1,
        height: ir.height || 1,
      };
    }

    function placeBox(el, x, y, w, h) {
      if (!el) return;
      el.style.left = x + "px";
      el.style.top = y + "px";
      el.style.width = w + "px";
      el.style.height = h + "px";
      el.style.right = "auto";
      el.style.bottom = "auto";
    }

    function syncOverlayToImage(el) {
      if (!el) return;
      var b = imageBoxInHost();
      placeBox(el, b.left, b.top, b.width, b.height);
    }

    function updateSelectionBox(r) {
      if (!selectionBox) return;
      if (!r || isFull(r) || img.hidden) {
        selectionBox.hidden = true;
        return;
      }
      var b = imageBoxInHost();
      selectionBox.hidden = false;
      placeBox(
        selectionBox,
        b.left + r.x * b.width,
        b.top + r.y * b.height,
        r.w * b.width,
        r.h * b.height
      );
    }

    function readStoredRegion() {
      var cur = regionInput.value.split(",");
      if (cur.length !== 4) return null;
      return { x: +cur[0], y: +cur[1], w: +cur[2], h: +cur[3] };
    }

    function refreshOverlays() {
      var mode = modeEl.value;
      if (mode === "rect" && rectLayer) syncOverlayToImage(rectLayer);
      if (mode === "grid" && gridOverlay && !gridOverlay.hidden) syncOverlayToImage(gridOverlay);
      var r = readStoredRegion();
      if (r && !isFull(r)) updateSelectionBox(r);
    }

    function commitRegion(r) {
      regionInput.value = formatRegion(r.x, r.y, r.w, r.h);
      updateSelectionBox(r);
      if (labelEl) {
        if (isFull(r)) {
          labelEl.textContent = (t("ref_grid_selected") || "Selected") + ": " + (t("ref_grid_full") || "Full");
        } else {
          var pct = Math.round(r.w * r.h * 100);
          labelEl.textContent = (t("ref_grid_selected") || "Selected") + ": " + pct + "%";
        }
      }
    }

    function syncCellsInput() {
      if (!cellsInput) return;
      cellsInput.value = Array.from(selectedCells)
        .sort(function (a, b) {
          return a - b;
        })
        .join(",");
    }

    function buildGrid() {
      if (!gridOverlay) return;
      gridOverlay.innerHTML = "";
      var mode = modeEl.value;
      var d = Number(divisionsEl && divisionsEl.value) || 1;
      if (mode !== "grid" || d <= 1 || img.hidden) {
        gridOverlay.hidden = true;
        gridOverlay.classList.remove("is-active");
        return;
      }
      syncOverlayToImage(gridOverlay);
      gridOverlay.hidden = false;
      gridOverlay.classList.add("is-active");
      var cols = d === 6 ? 3 : 2;
      gridOverlay.style.gridTemplateColumns = "repeat(" + cols + ", 1fr)";
      gridOverlay.style.gridTemplateRows = "repeat(2, 1fr)";
      if (!selectedCells.size) selectedCells.add(0);
      for (var i = 0; i < d; i++) {
        var btn = document.createElement("button");
        btn.type = "button";
        btn.className = "ref-grid-cell" + (selectedCells.has(i) ? " is-selected" : "");
        btn.dataset.cell = String(i);
        var cap = document.createElement("span");
        cap.textContent = String(i + 1);
        btn.appendChild(cap);
        btn.addEventListener("click", function (ev) {
          ev.preventDefault();
          var c = parseInt(this.dataset.cell, 10);
          if (ev.ctrlKey || ev.metaKey || ev.shiftKey) {
            if (selectedCells.has(c)) selectedCells.delete(c);
            else selectedCells.add(c);
          } else {
            selectedCells.clear();
            selectedCells.add(c);
          }
          if (!selectedCells.size) selectedCells.add(0);
          syncCellsInput();
          var r = unionCellsNorm(d, Array.from(selectedCells));
          commitRegion(r);
          buildGrid();
        });
        gridOverlay.appendChild(btn);
      }
      var r = unionCellsNorm(d, Array.from(selectedCells));
      commitRegion(r);
    }

    function hideGrid() {
      if (gridOverlay) {
        gridOverlay.hidden = true;
        gridOverlay.classList.remove("is-active");
        gridOverlay.innerHTML = "";
      }
    }

    function setModeUi() {
      var mode = modeEl.value;
      if (divisionsWrap) {
        divisionsWrap.hidden = mode !== "grid";
      }
      if (rectLayer) {
        rectLayer.style.pointerEvents = mode === "rect" ? "auto" : "none";
        rectLayer.classList.toggle("is-active", mode === "rect");
        if (mode === "rect") syncOverlayToImage(rectLayer);
      }
      if (host) {
        host.classList.toggle("region-mode-rect", mode === "rect");
      }
      if (mode === "full") {
        selectedCells.clear();
        syncCellsInput();
        hideGrid();
        commitRegion({ x: 0, y: 0, w: 1, h: 1 });
      } else if (mode === "grid") {
        if (selectionBox) selectionBox.hidden = true;
        buildGrid();
      } else {
        hideGrid();
        var cur = readStoredRegion();
        if (cur && !isFull(cur)) {
          commitRegion(cur);
        } else if (labelEl) {
          labelEl.textContent =
            (t("ref_grid_selected") || "Selected") + ": " + (t("region_drag_hint") || "Drag on image");
        }
      }
    }

    function pointerToNorm(clientX, clientY) {
      var ir = img.getBoundingClientRect();
      if (!ir.width || !ir.height) return { x: 0, y: 0 };
      return {
        x: clamp((clientX - ir.left) / ir.width, 0, 1),
        y: clamp((clientY - ir.top) / ir.height, 0, 1),
      };
    }

    function onRectDown(ev) {
      if (modeEl.value !== "rect" || img.hidden) return;
      ev.preventDefault();
      var p = pointerToNorm(ev.clientX, ev.clientY);
      drag = { x0: p.x, y0: p.y, x1: p.x, y1: p.y };
      if (rectLayer) rectLayer.setPointerCapture(ev.pointerId);
    }

    function onRectMove(ev) {
      if (!drag) return;
      var p = pointerToNorm(ev.clientX, ev.clientY);
      drag.x1 = p.x;
      drag.y1 = p.y;
      var x = Math.min(drag.x0, drag.x1);
      var y = Math.min(drag.y0, drag.y1);
      var w = Math.abs(drag.x1 - drag.x0);
      var h = Math.abs(drag.y1 - drag.y0);
      updateSelectionBox({ x: x, y: y, w: w, h: h });
    }

    function onRectUp(ev) {
      if (!drag) return;
      var p = pointerToNorm(ev.clientX, ev.clientY);
      drag.x1 = p.x;
      drag.y1 = p.y;
      var x = Math.min(drag.x0, drag.x1);
      var y = Math.min(drag.y0, drag.y1);
      var w = Math.max(0.02, Math.abs(drag.x1 - drag.x0));
      var h = Math.max(0.02, Math.abs(drag.y1 - drag.y0));
      drag = null;
      if (rectLayer) {
        try {
          rectLayer.releasePointerCapture(ev.pointerId);
        } catch (e) {}
      }
      commitRegion({ x: x, y: y, w: w, h: h });
    }

    modeEl.addEventListener("change", setModeUi);
    if (divisionsEl) divisionsEl.addEventListener("change", buildGrid);
    if (img) {
      img.addEventListener("load", function () {
        setModeUi();
        refreshOverlays();
      });
    }
    host.addEventListener("fp-preview-transform", refreshOverlays);
    host.addEventListener("scroll", refreshOverlays, true);

    if (rectLayer) {
      rectLayer.addEventListener("pointerdown", onRectDown);
      rectLayer.addEventListener("pointermove", onRectMove);
      rectLayer.addEventListener("pointerup", onRectUp);
      rectLayer.addEventListener("pointercancel", onRectUp);
    }

    setModeUi();
  }

  global.initRegionSelector = initRegionSelector;
})(window);
