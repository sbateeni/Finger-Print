(function () {
    "use strict";
    var App = window.EditorApp;
    if (!App) return;
    var state = App.state;
    var elements = App.elements;

    App.setActiveSide = function (sideName) {
        state.activeSide = sideName;
        state.selectedMinutiaIndex = -1;

        var oBox = elements.boxOriginal();
        var pBox = elements.boxPartial();
        if (oBox && pBox) {
            oBox.style.borderColor = sideName === 'original' ? '#667eea' : '#ddd';
            pBox.style.borderColor = sideName === 'partial' ? '#667eea' : '#ddd';
            oBox.style.borderWidth = sideName === 'original' ? '3px' : '2px';
            pBox.style.borderWidth = sideName === 'partial' ? '3px' : '2px';
        }

        App.updateUI();
        App.drawBoth();
    };

    App.loadSide = async function (sideName) {
        var s = state[sideName];
        var fpId = sideName === 'original' ? state.originalId : state.partialId;
        s.fingerprintId = fpId;

        try {
            var r = await fetch('/api/editor/fingerprint/' + fpId);
            if (!r.ok) throw new Error('Failed to load ' + sideName);
            var d = await r.json();
            s.minutiae = d.minutiae || [];
            s.landmarks = d.landmarks || {};
            s.classification = d.classification || {};
            s.visualizations = d.visualizations || {};

            var vizUrl = s.visualizations[state.currentVizType];
            if (vizUrl) {
                s.image.onload = function () { App.drawCanvas(sideName); };
                s.image.src = vizUrl;
            } else {
                s.image.onload = function () { App.drawCanvas(sideName); };
                s.image.src = d.image_url;
            }

            if (!state.matchData) {
                try {
                    var mr = await fetch('/api/editor/match-data/' + state.originalId + '/' + state.partialId);
                    if (mr.ok) state.matchData = await mr.json();
                } catch (e) { console.error('match data load failed:', e); }
            }

            if (!App.cur().fingerprintId) {
                App.setActiveSide(sideName);
            }

            App.drawCanvas(sideName);
        } catch (err) {
            console.error('Error loading ' + sideName + ':', err);
        }
    };

    App.setupEventListeners = function () {
        var cvs = [state.original.canvas, state.partial.canvas];

        cvs.forEach(function (cv) {
            if (!cv) return;
            cv.addEventListener('contextmenu', function (e) { e.preventDefault(); });

            cv.addEventListener('mousedown', function (e) {
                var sideName = App.getCanvasSide(this);
                if (!sideName) return;

                if (e.button === 2) {
                    state.dragStartOffset.original.x = state.original.offsetX;
                    state.dragStartOffset.original.y = state.original.offsetY;
                    state.dragStartOffset.partial.x = state.partial.offsetX;
                    state.dragStartOffset.partial.y = state.partial.offsetY;
                    state.isDragging = true;
                    state.dragStartX = e.clientX;
                    state.dragStartY = e.clientY;
                    this.style.cursor = 'grabbing';
                    return;
                }

                if (e.button !== 0) return;

                App.setActiveSide(sideName);
                var s = App.cur();

                var rect = this.getBoundingClientRect();
                var px = (e.clientX - rect.left) * (s.canvas.width / rect.width);
                var py = (e.clientY - rect.top) * (s.canvas.height / rect.height);
                var cx = s.canvas.width / 2;
                var cy = s.canvas.height / 2;
                var imgX = ((px - cx - s.offsetX) / s.zoom + cx) / s.scale;
                var imgY = ((py - cy - s.offsetY) / s.zoom + cy) / s.scale;

                var currentMode = elements.modeSelect().value;

                if (currentMode === 'add') {
                    App.addMinutia(imgX, imgY);
                } else if (currentMode === 'move') {
                    // Find nearest point for moving
                    var nearestIdx = -1, minDist = 40;
                    s.minutiae.forEach(function (m, idx) {
                        var sx = (m.x * s.scale - cx) * s.zoom + cx + s.offsetX;
                        var sy = (m.y * s.scale - cy) * s.zoom + cy + s.offsetY;
                        var d = Math.sqrt((sx - px) * (sx - px) + (sy - py) * (sy - py));
                        if (d < minDist) { minDist = d; nearestIdx = idx; }
                    });
                    if (nearestIdx !== -1) {
                        state.selectedMinutiaIndex = nearestIdx;
                        state.isMovingMinutia = true;
                        state.movingMinutiaIndex = nearestIdx;
                        this.style.cursor = 'move';
                        App.drawBoth();
                        App.renderMinutiaeList();
                    }
                } else {
                    // delete / edit / view
                    var nearestIdx = -1, minDist = 40;
                    s.minutiae.forEach(function (m, idx) {
                        var sx = (m.x * s.scale - cx) * s.zoom + cx + s.offsetX;
                        var sy = (m.y * s.scale - cy) * s.zoom + cy + s.offsetY;
                        var d = Math.sqrt((sx - px) * (sx - px) + (sy - py) * (sy - py));
                        if (d < minDist) { minDist = d; nearestIdx = idx; }
                    });
                    if (nearestIdx !== -1) {
                        state.selectedMinutiaIndex = nearestIdx;
                        App.drawBoth();
                        App.renderMinutiaeList();
                        if (currentMode === 'delete') {
                            App.deleteMinutia(nearestIdx);
                        } else if (currentMode === 'edit') {
                            App.showEditTypeOverlay(nearestIdx);
                        }
                    } else {
                        console.log('No point within 40px on ' + sideName + ': px=' + px + ' py=' + py);
                    }
                }
            });

            cv.addEventListener('wheel', function (e) {
                e.preventDefault();
                App.setActiveSide(App.getCanvasSide(this));
                if (e.deltaY < 0) App.zoomIn(); else App.zoomOut();
            }, { passive: false });
        });

        // Match overlay wheel (zoom) — overlay intercepts events from canvases
        var overlay = document.getElementById('match-overlay');
        if (overlay) {
            overlay.addEventListener('wheel', function (e) {
                e.preventDefault();
                // determine which side the mouse is over
                var oRect = document.getElementById('fp-box-original').getBoundingClientRect();
                var sideName = (e.clientX >= oRect.left && e.clientX < oRect.right) ? 'original' : 'partial';
                App.setActiveSide(sideName);
                if (e.deltaY < 0) App.zoomIn(); else App.zoomOut();
            }, { passive: false });

            // Match overlay dragging
            overlay.addEventListener('mousedown', function (e) {
                if (!state.matchData || !state.matchData.matched_details) return;
                if (e.button !== 0) return;
                var wrapper = document.getElementById('canvas-wrapper');
                var wr = wrapper.getBoundingClientRect();
                var mx = e.clientX - wr.left;
                var my = e.clientY - wr.top;
                var oCanvas = document.getElementById('canvas-original');
                var pCanvas = document.getElementById('canvas-partial');
                var oR = oCanvas.getBoundingClientRect();
                var pR = pCanvas.getBoundingClientRect();
                var bestDist = 20, bestIdx = -1, bestSide = '';
                state.matchData.matched_details.forEach(function (pair, idx) {
                    var oPos = App.getMatchEndpointPos(pair, 'original', oR, wr);
                    var pPos = App.getMatchEndpointPos(pair, 'partial', pR, wr);
                    var d1 = Math.sqrt((mx - oPos.x) * (mx - oPos.x) + (my - oPos.y) * (my - oPos.y));
                    var d2 = Math.sqrt((mx - pPos.x) * (mx - pPos.x) + (my - pPos.y) * (my - pPos.y));
                    if (d1 < bestDist) { bestDist = d1; bestIdx = idx; bestSide = 'original'; }
                    if (d2 < bestDist) { bestDist = d2; bestIdx = idx; bestSide = 'partial'; }
                });
                if (bestIdx !== -1 && bestSide) {
                    state.isDraggingMatchPoint = true;
                    state.draggingMatchIndex = bestIdx;
                    state.draggingMatchSide = bestSide;
                    this.style.cursor = 'grabbing';
                }
            });
        }

        window.addEventListener('mousemove', function (e) {
            if (state.isDraggingMatchPoint) {
                var overlay = document.getElementById('match-overlay');
                var wrapper = document.getElementById('canvas-wrapper');
                var wr = wrapper.getBoundingClientRect();
                var ox = e.clientX - wr.left;
                var oy = e.clientY - wr.top;
                var pair = state.matchData.matched_details[state.draggingMatchIndex];
                var side = state.draggingMatchSide;
                var pt = side === 'original' ? pair.original : pair.partial;
                var ic = App.overlayToImageCoords(ox, oy, side);
                pt.x = ic.x;
                pt.y = ic.y;
                App.drawMatchOverlayLines();
                App.drawBoth();
                return;
            }
            if (state.isMovingMinutia) {
                var s = App.cur();
                var rect = s.canvas.getBoundingClientRect();
                var px = (e.clientX - rect.left) * (s.canvas.width / rect.width);
                var py = (e.clientY - rect.top) * (s.canvas.height / rect.height);
                var cx = s.canvas.width / 2;
                var cy = s.canvas.height / 2;
                var imgX = ((px - cx - s.offsetX) / s.zoom + cx) / s.scale;
                var imgY = ((py - cy - s.offsetY) / s.zoom + cy) / s.scale;
                var m = s.minutiae[state.movingMinutiaIndex];
                if (m) { m.x = Math.round(imgX); m.y = Math.round(imgY); App.drawBoth(); App.renderMinutiaeList(); }
                return;
            }
            if (state.isDragging) {
                App.eachSide(function (side) {
                    var s = state[side];
                    s.offsetX = state.dragStartOffset[side].x + (e.clientX - state.dragStartX);
                    s.offsetY = state.dragStartOffset[side].y + (e.clientY - state.dragStartY);
                });
                App.drawBoth();
            }
        });

        window.addEventListener('mouseup', function () {
            if (state.isDraggingMatchPoint) {
                state.isDraggingMatchPoint = false;
                state.draggingMatchIndex = -1;
                state.draggingMatchSide = '';
                var overlay = document.getElementById('match-overlay');
                if (overlay) overlay.style.cursor = 'default';
                return;
            }
            if (state.isMovingMinutia) {
                var idx = state.movingMinutiaIndex;
                state.isMovingMinutia = false;
                state.movingMinutiaIndex = -1;
                var s = App.cur();
                s.canvas.style.cursor = 'crosshair';
                if (idx !== -1 && s.minutiae[idx]) {
                    var m = s.minutiae[idx];
                    fetch('/api/editor/move-minutia', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ fingerprint_id: s.fingerprintId, minutia_index: idx, x: m.x, y: m.y })
                    }).catch(function (err) { console.error('Save move failed:', err); });
                }
                return;
            }
            if (state.isDragging) {
                state.isDragging = false;
                var s = App.cur();
                s.canvas.style.cursor = 'crosshair';
            }
        });

        if (elements.btnZoomIn()) elements.btnZoomIn().addEventListener('click', App.zoomIn);
        if (elements.btnZoomOut()) elements.btnZoomOut().addEventListener('click', App.zoomOut);
        if (elements.btnZoomReset()) elements.btnZoomReset().addEventListener('click', App.resetZoom);

        elements.modeSelect().addEventListener('change', function () {
            state.selectedMinutiaIndex = -1;
            var lmGroup = document.querySelector('.control-group:has(#landmark-select)');
            var angleGroup = document.querySelector('.control-group:has(#angle-input)');
            var isAdd = this.value === 'add';
            if (lmGroup) lmGroup.style.display = isAdd ? '' : 'none';
            if (angleGroup) angleGroup.style.display = isAdd ? '' : 'none';
            App.drawBoth();
        });
        elements.modeSelect().dispatchEvent(new Event('change'));

        elements.btnReset().addEventListener('click', function () {
            if (confirm('Reset all changes?')) {
                App.loadSide(state.activeSide);
            }
        });

        elements.btnApprove().addEventListener('click', function () { elements.approveModal().classList.add('show'); });
        elements.btnReject().addEventListener('click', function () { elements.rejectModal().classList.add('show'); });

        elements.confirmApprove().addEventListener('click', async function () {
            var s = App.cur();
            try {
                var r = await fetch('/api/editor/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fingerprint_id: s.fingerprintId, minutiae: s.minutiae, notes: elements.approveNotes().value, user_id: 1 })
                });
                if (r.ok) { alert('Fingerprint approved successfully!'); window.close(); }
                else alert('Failed to approve');
            } catch (error) { console.error('Approve failed:', error); }
        });

        elements.confirmReject().addEventListener('click', async function () {
            var s = App.cur();
            try {
                var r = await fetch('/api/editor/reject', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fingerprint_id: s.fingerprintId, reason: elements.rejectReason().value, user_id: 1 })
                });
                if (r.ok) { alert('Fingerprint rejected'); window.close(); }
                else alert('Failed to reject');
            } catch (error) { console.error('Reject failed:', error); }
        });

        window.closeModal = function (id) { document.getElementById(id).classList.remove('show'); };

        window.addEventListener('resize', function () { App.drawBoth(); });
    };

    App.init = async function () {
        var urlParams = new URLSearchParams(window.location.search);
        state.originalId = urlParams.get('original_id');
        state.partialId = urlParams.get('partial_id');
        if (!state.originalId || !state.partialId) {
            App.renderNoIdMessage(urlParams);
            return;
        }

        state.original.canvas = elements.canvasOriginal();
        state.original.ctx = state.original.canvas.getContext('2d');
        state.partial.canvas = elements.canvasPartial();
        state.partial.ctx = state.partial.canvas.getContext('2d');

        state.deletedCount = 0;

        App.setupEventListeners();
        App.setupCheckboxFilter();
        await Promise.all([App.loadSide('original'), App.loadSide('partial')]);
        App.setActiveSide('original');
        App.setupVizSelector();
        App.updateUI();
    };

    App.setupEditOverlay();

    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', App.init);
    else
        App.init();
})();
