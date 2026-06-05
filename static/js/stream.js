(function (window) {
  "use strict";

  var FP_APP = window.FP_APP || {};

  FP_APP.activeRun = {
    running: false,
    abort: null,
    reader: null,
    generation: 0,
    db_ids: null,
  };

  FP_APP.handleSseEvent = function (ev) {
    if (!ev || !ev.type) return;
    switch (ev.type) {
      case "log": FP_APP.appendLog(ev.message || ""); break;
      case "hashes": if (ev.same_file_warning) FP_APP.alertBox("warn", FP_APP.t("same_file_sse")); break;
      case "image": FP_APP.updateLiveImage(ev); break;
      case "db_ids": 
        FP_APP.activeRun.db_ids = ev; 
        FP_APP.updateManualEditLinks(ev); 
        break;
      case "done": 
        if (ev.forensic_quality_warning) FP_APP.alertBox("warn", FP_APP.t("quality_warning"));
        FP_APP.renderDone(ev);
        FP_APP.appendLog(FP_APP.t("done_log"));
        break;
      case "fatal": 
        FP_APP.alertBox("err", FP_APP.esc(ev.message || "Error"));
        if (ev.forensic_quality_warning) FP_APP.alertBox("warn", FP_APP.t("quality_sse"));
        break;
    }
  };

  FP_APP.runStream = async function (form) {
    if (FP_APP.activeRun.running) return;
    var runGeneration = ++FP_APP.activeRun.generation;

    var panel = FP_APP.$("live-panel");
    if (panel) { panel.hidden = false; panel.setAttribute("aria-busy", "true"); }
    var serverBlock = FP_APP.$("server-results");
    if (serverBlock) serverBlock.hidden = true;

    FP_APP.clearAlerts();
    FP_APP.resetLiveColumns();
    FP_APP.activeRun.running = true;
    FP_APP.activeRun.db_ids = null;

    var abort = new AbortController();
    FP_APP.activeRun.abort = abort;
    FP_APP.setRunActive(true);

    var fd = new FormData(form);
    var uiLang = form.getAttribute("data-lang") || "ar";
    
    try {
      var res = await fetch("/analyze-stream?lang=" + encodeURIComponent(uiLang), {
        method: "POST",
        body: fd,
        headers: { Accept: "text/event-stream" },
        signal: abort.signal,
      });

      if (!res.ok || !res.body) throw new Error(FP_APP.t("unexpected_response") + " (" + res.status + ")");

      var reader = res.body.getReader();
      FP_APP.activeRun.reader = reader;
      var dec = new TextDecoder();
      var buf = "";

      while (FP_APP.activeRun.running) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buf += dec.decode(chunk.value, { stream: true });
        buf = FP_APP.parseSseBuffer(buf, FP_APP.handleSseEvent);
      }
      if (FP_APP.activeRun.running) FP_APP.parseSseBuffer(buf + "\n\n", FP_APP.handleSseEvent);
    } catch (e) {
      console.error("Stream Error:", e);
      if (e.name !== "AbortError") FP_APP.alertBox("err", FP_APP.t("server_error") + ": " + FP_APP.esc(e.message));
    } finally {
      FP_APP.finishRun(false, runGeneration);
    }
  };

  FP_APP.finishRun = function (wasCancelled, runGeneration) {
    if (runGeneration != null && runGeneration !== FP_APP.activeRun.generation) return;
    FP_APP.activeRun.running = false;
    FP_APP.activeRun.abort = null;
    FP_APP.activeRun.reader = null;
    FP_APP.setRunActive(false);
    var panel = FP_APP.$("live-panel");
    if (panel) panel.setAttribute("aria-busy", "false");
  };

  FP_APP.setRunActive = function (active) {
    var btn = FP_APP.$("fp-submit-btn");
    if (!btn) return;
    if (!btn.dataset.label) btn.dataset.label = btn.textContent;
    if (active) {
      btn.type = "button";
      btn.classList.add("btn--stop");
      btn.textContent = FP_APP.t("stop_analysis", "Stop");
    } else {
      btn.type = "submit";
      btn.classList.remove("btn--stop");
      btn.textContent = btn.dataset.label;
    }
  };

  FP_APP.stopActiveRun = function () {
    if (!FP_APP.activeRun.running) return;
    FP_APP.activeRun.generation += 1;
    FP_APP.activeRun.running = false;
    if (FP_APP.activeRun.abort) FP_APP.activeRun.abort.abort();
    FP_APP.finishRun(true);
  };

  FP_APP.parseSseBuffer = function (buffer, onEvent) {
    var parts = buffer.split("\n\n");
    var rest = parts.pop() || "";
    for (var i = 0; i < parts.length; i++) {
      var lines = parts[i].split("\n");
      for (var j = 0; j < lines.length; j++) {
        if (lines[j].indexOf("data: ") === 0) {
          try { onEvent(JSON.parse(lines[j].slice(6))); } catch (e) {}
        }
      }
    }
    return rest;
  };

  window.FP_APP = FP_APP;
})(window);
