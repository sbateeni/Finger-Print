(function (window) {
  "use strict";

  var FP_APP = window.FP_APP || {};
  FP_APP.TRANS = {};

  // --- Core Utilities ---
  FP_APP.$ = function (id) { return document.getElementById(id); };
  
  FP_APP.esc = function (s) {
    var d = document.createElement("div");
    d.textContent = s == null ? "" : String(s);
    return d.innerHTML;
  };

  FP_APP.t = function (key, fallback) {
    if (FP_APP.TRANS && FP_APP.TRANS[key] != null) return FP_APP.TRANS[key];
    return fallback != null ? fallback : key;
  };

  FP_APP.loadTrans = function () {
    var el = FP_APP.$("fp-trans-json");
    if (!el || !el.textContent) return;
    try { 
      FP_APP.TRANS = JSON.parse(el.textContent); 
    } catch (e) { 
      console.warn("fp i18n", e); 
    }
  };

  // --- UI Feedback ---
  FP_APP.appendLog = function (line) {
    var el = FP_APP.$("live-log");
    if (!el) return;
    el.textContent += (el.textContent ? "\n" : "") + line;
    el.scrollTop = el.scrollHeight;
  };

  FP_APP.alertBox = function (className, html) {
    var a = FP_APP.$("live-alerts");
    if (!a) return;
    var d = document.createElement("div");
    d.className = "alert " + className;
    d.innerHTML = html;
    a.appendChild(d);
  };

  FP_APP.clearAlerts = function () {
    var a = FP_APP.$("live-alerts");
    if (a) a.innerHTML = "";
  };

  FP_APP.setColStatus = function (branch, text) {
    var map = {
      reference: "live-status-ref",
      partial: "live-status-partial",
      match: "live-status-match",
      orb: "live-status-match"
    };
    var el = FP_APP.$(map[branch]);
    if (el) el.textContent = text;
  };

  window.FP_APP = FP_APP;
})(window);
