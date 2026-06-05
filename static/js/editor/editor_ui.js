(function () {
    "use strict";
    var App = window.EditorApp;
    if (!App) return;
    var state = App.state;
    var elements = App.elements;

    App.updateUI = function () {
        var s = App.cur();
        elements.statusCount().textContent = s.minutiae.length;
        elements.statusAdded().textContent = s.minutiae.filter(function (m) { return m.manually_added; }).length;
        elements.statusDeleted().textContent = state.deletedCount;

        if (s.classification && (s.classification.finger_type || s.classification.pattern_type)) {
            var ft = s.classification.finger_type || 'unknown';
            var pattern = s.classification.pattern_type || '';
            var region = s.classification.region || 'unknown';
            var confidence = s.classification.confidence || 0;
            var ftName = App.FINGER_TYPE_NAMES_AR[ft] || ft;
            var regionName = App.REGION_NAMES_AR[region] || region;
            var patternName = pattern ? (App.PATTERN_NAMES_AR[pattern] || pattern) : '';
            var text = ftName + ' (' + ft + ')';
            if (patternName) text += ' | ' + patternName;
            text += ' | ' + regionName;
            text += ' | ثقة: ' + Math.round(confidence * 100) + '%';
            elements.statusClass().textContent = text;
        } else {
            elements.statusClass().textContent = 'غير متوفر | N/A';
        }

        App.renderMinutiaeList();
    };

    App.renderMinutiaeList = function () {
        var s = App.cur();
        var list = elements.minutiaeList();
        list.innerHTML = '';

        if (s.minutiae.length === 0) {
            list.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">لم يتم تحميل أي نقاط | No minutiae loaded</div>';
            return;
        }

        var label = state.activeSide === 'original' ? 'أصلي (Original)' : 'مقارن (Partial)';
        var html = '<div style="margin-bottom:8px;font-size:14px;color:#333;font-weight:600;">' + label + ' — ' + s.minutiae.length + ' نقطة</div>';

        // Type count summary
        var typeNames = ['termination', 'bifurcation', 'island', 'ridge', 'loop_eye', 'bridge', 'lake', 'dot'];
        var counts = {};
        typeNames.forEach(function (t) { counts[t] = 0; });
        s.minutiae.forEach(function (m) {
            var t = m.landmark_type || 'termination';
            counts[t] = (counts[t] || 0) + 1;
        });
        html += '<div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:10px;padding:6px 8px;background:#f5f5f5;border-radius:4px;font-size:11px;">';
        typeNames.forEach(function (t) {
            if (counts[t] > 0) {
                html += '<span style="padding:2px 6px;border-radius:3px;background:' + (App.LANDMARK_COLORS[t] || '#ccc') + ';color:#fff;">' + App.LANDMARK_NAMES_AR[t] + ' ' + counts[t] + '</span>';
            }
        });
        html += '</div>';

        list.innerHTML = html;

        s.minutiae.forEach(function (m, index) {
            var item = document.createElement('div');
            item.className = 'minutia-item' + (index === state.selectedMinutiaIndex ? ' selected' : '');
            var type = m.landmark_type || 'termination';
            var typeName = App.LANDMARK_NAMES_AR[type] || type;
            item.innerHTML = '<div class="minutia-info">' +
                '<div class="minutia-type" style="color:' + (App.LANDMARK_COLORS[type] || '#333') + '">' +
                (App.LANDMARK_SYMBOLS[type] || '•') + ' ' + typeName + '</div>' +
                '<div class="minutia-coords">X: ' + Math.round(m.x) + ', Y: ' + Math.round(m.y) + ' | ' + Math.round(m.angle) + '°</div></div>' +
                '<div class="minutia-actions" style="display:flex;gap:4px;">' +
                '<button class="action-btn edit-type" data-index="' + index + '" title="تعديل النوع | Edit Type" style="background:#FF9800;">✎</button>' +
                '<button class="action-btn delete" data-index="' + index + '" title="حذف | Delete">×</button></div>';

            item.addEventListener('click', function () {
                state.selectedMinutiaIndex = index;
                App.drawBoth();
                App.renderMinutiaeList();
            });

            item.querySelector('.edit-type').addEventListener('click', function (e) {
                e.stopPropagation();
                App.showEditTypeOverlay(index);
            });

            item.querySelector('.delete').addEventListener('click', function (e) {
                e.stopPropagation();
                App.deleteMinutia(index);
            });

            list.appendChild(item);
        });
    };

    App.setupVizSelector = function () {
        var sel = elements.vizSelect();
        if (!sel) return;
        var keys = ['processed', 'ridges', 'skeleton'];
        var labels = { processed: 'معالجة (Processed)', ridges: 'تموجات (Ridges)', skeleton: 'هيكل (Skeleton)' };
        if (state.matchData && state.matchData.matched_details && state.matchData.matched_details.length > 0) {
            keys.push('__match__');
            labels['__match__'] = 'مطابقة (Match) — خطوط';
        }
        sel.style.display = keys.length <= 1 ? 'none' : '';
        sel.innerHTML = '';
        keys.forEach(function (k) {
            var opt = document.createElement('option');
            opt.value = k;
            opt.textContent = labels[k] || k;
            sel.appendChild(opt);
        });
        sel.value = state.currentVizType;
        sel.onchange = function () {
            var v = sel.value;
            if (v === '__match__') {
                state.currentVizType = '__match__';
                App.drawMatchLinesOnCanvases();
            } else {
                App.hideMatchOverlay();
                state.currentVizType = v;
                var sides = ['original', 'partial'];
                sides.forEach(function (sideName) {
                    var s = state[sideName];
                    var url = s.visualizations && s.visualizations[v];
                    if (url) {
                        s.image.onload = function () { App.drawCanvas(sideName); };
                        s.image.src = url;
                    }
                });
            }
        };
    };

    App.setupCheckboxFilter = function () {
        document.querySelectorAll('.type-filter').forEach(function (cb) {
            cb.addEventListener('change', function () {
                state.visibleTypes[this.dataset.type] = this.checked;
                App.drawBoth();
            });
        });
    };

    App.updateZoomDisplay = function () {
        var el = elements.zoomLevel();
        var s = App.cur();
        if (el) el.textContent = Math.round(s.zoom * 100) + '%';
    };

    App.zoomIn = function () {
        App.eachSide(function (side) {
            var s = state[side];
            s.zoom = Math.min(s.zoom * 1.2, 10);
        });
        App.drawBoth();
        if (state.currentVizType === '__match__') App.drawMatchOverlayLines();
        App.updateZoomDisplay();
    };

    App.zoomOut = function () {
        App.eachSide(function (side) {
            var s = state[side];
            s.zoom = Math.max(s.zoom / 1.2, 0.1);
        });
        App.drawBoth();
        if (state.currentVizType === '__match__') App.drawMatchOverlayLines();
        App.updateZoomDisplay();
    };

    App.resetZoom = function () {
        App.eachSide(function (side) {
            var s = state[side];
            s.zoom = 1; s.offsetX = 0; s.offsetY = 0;
        });
        App.drawBoth();
        if (state.currentVizType === '__match__') App.drawMatchOverlayLines();
        App.updateZoomDisplay();
    };
})();
