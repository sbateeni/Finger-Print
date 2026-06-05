(function () {
    "use strict";
    var App = window.EditorApp;
    if (!App) return;
    var state = App.state;
    var LANDMARK_COLORS = App.LANDMARK_COLORS;
    var LANDMARK_SYMBOLS = App.LANDMARK_SYMBOLS;
    var LANDMARK_NAMES_AR = App.LANDMARK_NAMES_AR;

    App.getMatchEndpointPos = function (pair, sideName, canvasRect, wrapperRect) {
        var s = state[sideName];
        var pt = sideName === 'original' ? pair.original : pair.partial;
        var cx = s.canvas.width / 2;
        var cy = s.canvas.height / 2;
        return {
            x: canvasRect.left - wrapperRect.left + (pt.x * s.scale - cx) * s.zoom + cx + s.offsetX,
            y: canvasRect.top - wrapperRect.top + (pt.y * s.scale - cy) * s.zoom + cy + s.offsetY
        };
    };

    App.drawMatchOverlayLines = function () {
        var overlay = document.getElementById('match-overlay');
        if (!overlay) return;
        if (!state.matchData || !state.matchData.matched_details) return;

        var wrapper = document.getElementById('canvas-wrapper');
        var wr = wrapper.getBoundingClientRect();

        overlay.style.display = '';
        overlay.width = wrapper.clientWidth;
        overlay.height = wrapper.clientHeight;
        overlay.style.width = wrapper.clientWidth + 'px';
        overlay.style.height = wrapper.clientHeight + 'px';
        var ctx = overlay.getContext('2d');

        var oCanvas = document.getElementById('canvas-original');
        var pCanvas = document.getElementById('canvas-partial');
        var oR = oCanvas.getBoundingClientRect();
        var pR = pCanvas.getBoundingClientRect();

        ctx.font = 'bold 13px Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        state.matchData.matched_details.forEach(function (pair, idx) {
            var oPos = App.getMatchEndpointPos(pair, 'original', oR, wr);
            var pPos = App.getMatchEndpointPos(pair, 'partial', pR, wr);

            ctx.beginPath();
            ctx.moveTo(oPos.x, oPos.y);
            ctx.lineTo(pPos.x, pPos.y);
            ctx.strokeStyle = 'rgba(0,255,100,0.8)';
            ctx.lineWidth = 2;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(oPos.x, oPos.y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#00FF00';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(pPos.x, pPos.y, 6, 0, Math.PI * 2);
            ctx.fillStyle = '#FFC800';
            ctx.fill();
            ctx.strokeStyle = '#000';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            var num = (idx + 1).toString();
            ctx.fillStyle = '#005500';
            ctx.font = 'bold 13px Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'bottom';
            ctx.fillText(num, oPos.x, oPos.y - 9);
            ctx.fillStyle = '#885500';
            ctx.fillText(num, pPos.x, pPos.y - 9);
        });
    };

    App.hideMatchOverlay = function () {
        var overlay = document.getElementById('match-overlay');
        if (overlay) { overlay.style.display = 'none'; overlay.style.pointerEvents = 'none'; }
    };

    App.overlayToImageCoords = function (overlayX, overlayY, sideName) {
        var s = state[sideName];
        var canvas = s.canvas;
        var canvasRect = canvas.getBoundingClientRect();
        var wrapper = document.getElementById('canvas-wrapper');
        var wr = wrapper.getBoundingClientRect();
        var cx = canvas.width / 2;
        var cy = canvas.height / 2;
        var cvsX = overlayX - (canvasRect.left - wr.left);
        var cvsY = overlayY - (canvasRect.top - wr.top);
        var imgX = ((cvsX - cx - s.offsetX) / s.zoom + cx) / s.scale;
        var imgY = ((cvsY - cy - s.offsetY) / s.zoom + cy) / s.scale;
        return { x: Math.round(imgX), y: Math.round(imgY) };
    };

    App.drawBoth = function () {
        App.drawCanvas('original');
        App.drawCanvas('partial');
    };

    App.drawCanvas = function (sideName) {
        var s = state[sideName];
        if (!s.canvas || !s.ctx) return;
        if (!s.image.complete) return;

        var ctx = s.ctx;
        var canvas = s.canvas;
        var isActive = state.activeSide === sideName;

        // Size canvas to container
        var container = canvas.parentElement;
        var w = container.clientWidth;
        var h = Math.max(container.clientHeight || 500, 500);
        var imgW = s.image.naturalWidth || s.image.width;
        var imgH = s.image.naturalHeight || s.image.height;
        var sc = Math.min(w / imgW, h / imgH);
        canvas.width = Math.floor(imgW * sc);
        canvas.height = Math.floor(imgH * sc);
        canvas.style.width = canvas.width + 'px';
        canvas.style.height = canvas.height + 'px';
        s.scale = sc;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Zoom/pan transform
        ctx.save();
        var cx = canvas.width / 2;
        var cy = canvas.height / 2;
        ctx.translate(cx + s.offsetX, cy + s.offsetY);
        ctx.scale(s.zoom, s.zoom);
        ctx.translate(-cx, -cy);

        // Draw image
        ctx.drawImage(s.image, 0, 0, canvas.width, canvas.height);

        // Draw minutiae
        var invZ = 1 / s.zoom;
        s.minutiae.forEach(function (m, index) {
            var type = m.landmark_type || 'termination';
            if (state.visibleTypes[type] === false) return;
            var x = m.x * sc;
            var y = m.y * sc;
            var color = LANDMARK_COLORS[type] || '#FF0000';
            var isSelected = isActive && index === state.selectedMinutiaIndex;

            if (isSelected) {
                ctx.beginPath();
                ctx.arc(x, y, 14 * invZ, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 3 * invZ;
                ctx.stroke();
            }

            var radius = (isSelected ? 8 : 5) * invZ;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = (isActive ? 1.5 : 1) * invZ;
            ctx.stroke();

            if (isActive) {
                ctx.fillStyle = '#000000';
                ctx.font = 'bold ' + (16 * invZ) + 'px Arial, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(LANDMARK_SYMBOLS[type] || '•', x, y - 20 * invZ);

                ctx.fillStyle = isSelected ? '#FFFFFF' : color;
                ctx.font = 'bold ' + (11 * invZ) + 'px Arial, sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'top';
                ctx.fillText(LANDMARK_NAMES_AR[type] || type, x, y + radius + 4 * invZ);
            }
        });

        // Match dots
        if (state.matchData && state.matchData.matched_details) {
            var isOrig = sideName === 'original';
            state.matchData.matched_details.forEach(function (pair) {
                var pt = isOrig ? pair.original : pair.partial;
                var x = pt.x * sc;
                var y = pt.y * sc;
                ctx.beginPath();
                ctx.arc(x, y, 3 * invZ, 0, Math.PI * 2);
                ctx.fillStyle = '#00FF00';
                ctx.fill();
                ctx.strokeStyle = '#000';
                ctx.lineWidth = 0.8 * invZ;
                ctx.stroke();
            });
        }

        ctx.restore();

        // Active indicator
        ctx.fillStyle = isActive ? 'rgba(102,126,234,0.85)' : 'rgba(0,0,0,0.4)';
        ctx.font = 'bold 11px Arial, sans-serif';
        var label = isActive ? 'نشط | Active' : 'اختر | Click';
        var tw = ctx.measureText(label).width;
        ctx.fillRect(4, 4, tw + 12, 20);
        ctx.fillStyle = '#FFFFFF';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, 10, 14);
    };

    App.drawMatchLinesOnCanvases = function () {
        var overlay = document.getElementById('match-overlay');
        if (overlay) overlay.style.pointerEvents = 'auto';
        var sides = ['original', 'partial'];
        var loaded = 0;
        sides.forEach(function (sideName) {
            var s = state[sideName];
            var url = s.visualizations && s.visualizations.processed;
            if (url) {
                s.image.onload = function () {
                    App.drawCanvas(sideName);
                    loaded++;
                    if (loaded === 2) App.drawMatchOverlayLines();
                };
                s.image.src = url;
            } else {
                loaded++;
                if (loaded === 2) App.drawMatchOverlayLines();
            }
        });
    };
})();
