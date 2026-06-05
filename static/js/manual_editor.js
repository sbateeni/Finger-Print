(function () {
    "use strict";

    const state = {
        activeSide: 'original',
        original: { fingerprintId: null, image: new Image(), minutiae: [], landmarks: {}, classification: {}, visualizations: {}, canvas: null, ctx: null, scale: 1, zoom: 1, offsetX: 0, offsetY: 0 },
        partial: { fingerprintId: null, image: new Image(), minutiae: [], landmarks: {}, classification: {}, visualizations: {}, canvas: null, ctx: null, scale: 1, zoom: 1, offsetX: 0, offsetY: 0 },
        matchData: null,
        currentVizType: 'processed',
        selectedMinutiaIndex: -1,
        mode: 'add',
        history: [],
        isDragging: false,
        dragStartX: 0, dragStartY: 0, dragStartOffset: { original: { x: 0, y: 0 }, partial: { x: 0, y: 0 } },
        isMovingMinutia: false, movingMinutiaIndex: -1,
        isDraggingMatchPoint: false, draggingMatchIndex: -1, draggingMatchSide: '',
        visibleTypes: { termination: true, bifurcation: true, island: true, ridge: true, loop_eye: true, bridge: true, lake: true, dot: true },
        deletedCount: 0
    };

    function cur() { return state[state.activeSide]; }

    const LANDMARK_COLORS = { termination: '#FF6B6B', bifurcation: '#4ECDC4', island: '#FFD93D', ridge: '#A8E6CF', loop_eye: '#FF8B94', bridge: '#C7CEEA', lake: '#B5EAD7', dot: '#95E1D3' };
    const LANDMARK_SYMBOLS = { termination: '◇', bifurcation: '⊢', island: '⊗', ridge: '─', loop_eye: '◯', bridge: '⌢', lake: '◈', dot: '•' };
    const LANDMARK_NAMES_AR = { termination: 'نهاية', bifurcation: 'تفرع', island: 'جزيرة', ridge: 'شرطة', loop_eye: 'عين', bridge: 'جسر', lake: 'بحيرة', dot: 'نقطة' };
    const PATTERN_NAMES_AR = { arch: 'قوس', tented_arch: 'قوس خيمي', left_loop: 'أنشوطة يسرى', right_loop: 'أنشوطة يمنى', whorl: 'دوامة' };
    const FINGER_TYPE_NAMES_AR = { thumb: 'إبهام', index: 'سبابة', middle: 'وسطى', ring: 'بنصر', pinky: 'خنصر', unknown: 'غير معروف' };
    const REGION_NAMES_AR = { fingertip: 'طرف الإصبع', palm_root: 'جذور الأصابع', sub_index: 'تحت السبابة', palm_general: 'راحة اليد', unknown: 'غير معروف' };

    const elements = {
        canvasOriginal: () => document.getElementById('canvas-original'),
        canvasPartial: () => document.getElementById('canvas-partial'),
        modeSelect: () => document.getElementById('mode-select'),
        landmarkSelect: () => document.getElementById('landmark-select'),
        angleInput: () => document.getElementById('angle-input'),
        statusCount: () => document.getElementById('status-count'),
        statusAdded: () => document.getElementById('status-added'),
        statusDeleted: () => document.getElementById('status-deleted'),
        statusClass: () => document.getElementById('status-class'),
        minutiaeList: () => document.getElementById('minutiae-list'),
        btnAddManual: () => document.getElementById('btn-add-manual'),
        btnDeleteSelected: () => document.getElementById('btn-delete-selected'),
        btnUndo: () => document.getElementById('btn-undo'),
        btnReset: () => document.getElementById('btn-reset'),
        btnApprove: () => document.getElementById('btn-approve'),
        btnReject: () => document.getElementById('btn-reject'),
        approveModal: () => document.getElementById('approve-modal'),
        rejectModal: () => document.getElementById('reject-modal'),
        confirmApprove: () => document.getElementById('confirm-approve'),
        confirmReject: () => document.getElementById('confirm-reject'),
        approveNotes: () => document.getElementById('approve-notes'),
        rejectReason: () => document.getElementById('reject-reason'),
        zoomLevel: () => document.getElementById('zoom-level'),
        btnZoomIn: () => document.getElementById('btn-zoom-in'),
        btnZoomOut: () => document.getElementById('btn-zoom-out'),
        btnZoomReset: () => document.getElementById('btn-zoom-reset'),
        editTypeSelect: () => document.getElementById('edit-type-select'),
        editConfirmBtn: () => document.getElementById('edit-confirm-btn'),
        editCancelBtn: () => document.getElementById('edit-cancel-btn'),
        editOverlay: () => document.getElementById('edit-overlay'),
        vizSelect: () => document.getElementById('viz-select'),
        boxOriginal: () => document.getElementById('fp-box-original'),
        boxPartial: () => document.getElementById('fp-box-partial'),
    };

    function renderNoIdMessage(urlParams) {
        const container = document.querySelector('.main-content');
        if (container) {
            container.innerHTML = `
                <div class="panel" style="grid-column: 1 / -1; text-align: center; padding: 3rem;">
                    <h2 style="color: #555; margin-bottom: 1rem;">${document.querySelector('header h1')?.textContent || 'Manual Editor'}</h2>
                    <div class="warning-box" style="display: inline-block; text-align: right;">
                        <strong>⚠️ لم يتم تحديد بصمة</strong>
                        <p style="margin: 0.75rem 0; font-size: 15px;">يرجى تحليل بصمة أولاً من الصفحة الرئيسية ثم الدخول إلى المحرر.</p>
                        <p style="margin: 0.75rem 0; font-size: 15px; direction: ltr;">No fingerprint selected. Please analyze a fingerprint from the main page first.</p>
                        <a href="/?lang=${urlParams.get('lang') || 'ar'}" style="display: inline-block; margin-top: 1rem; padding: 0.75rem 2rem; background: #667eea; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">
                            ← العودة إلى الصفحة الرئيسية
                        </a>
                    </div>
                </div>`;
        }
    }

    async function init() {
        const urlParams = new URLSearchParams(window.location.search);
        state.originalId = urlParams.get('original_id');
        state.partialId = urlParams.get('partial_id');
        if (!state.originalId || !state.partialId) {
            renderNoIdMessage(urlParams);
            return;
        }

        state.original.canvas = elements.canvasOriginal();
        state.original.ctx = state.original.canvas.getContext('2d');
        state.partial.canvas = elements.canvasPartial();
        state.partial.ctx = state.partial.canvas.getContext('2d');

        state.deletedCount = 0;

        setupEventListeners();
        setupCheckboxFilter();
        await Promise.all([loadSide('original'), loadSide('partial')]);
        setActiveSide('original');
        setupVizSelector();
        updateUI();
    }

    async function loadSide(sideName) {
        var s = state[sideName];
        var fpId = sideName === 'original' ? state.originalId : state.partialId;
        s.fingerprintId = fpId;

        try {
            const r = await fetch('/api/editor/fingerprint/' + fpId);
            if (!r.ok) throw new Error('Failed to load ' + sideName);
            const d = await r.json();
            s.minutiae = d.minutiae || [];
            s.landmarks = d.landmarks || {};
            s.classification = d.classification || {};
            s.visualizations = d.visualizations || {};

            var vizUrl = s.visualizations[state.currentVizType];
            if (vizUrl) {
                s.image.onload = function () { drawCanvas(sideName); };
                s.image.src = vizUrl;
            } else {
                s.image.onload = function () { drawCanvas(sideName); };
                s.image.src = d.image_url;
            }

            // Load match data once
            if (!state.matchData) {
                try {
                    var mr = await fetch('/api/editor/match-data/' + state.originalId + '/' + state.partialId);
                    if (mr.ok) state.matchData = await mr.json();
                } catch (e) { console.error('match data load failed:', e); }
            }

            // Activate original side if first load
            if (!cur().fingerprintId) {
                setActiveSide(sideName);
            }

            drawCanvas(sideName);
        } catch (err) {
            console.error('Error loading ' + sideName + ':', err);
        }
    }

    function setActiveSide(sideName) {
        state.activeSide = sideName;
        state.selectedMinutiaIndex = -1;

        // Update boxes border
        var oBox = elements.boxOriginal();
        var pBox = elements.boxPartial();
        if (oBox && pBox) {
            oBox.style.borderColor = sideName === 'original' ? '#667eea' : '#ddd';
            pBox.style.borderColor = sideName === 'partial' ? '#667eea' : '#ddd';
            oBox.style.borderWidth = sideName === 'original' ? '3px' : '2px';
            pBox.style.borderWidth = sideName === 'partial' ? '3px' : '2px';
        }

        updateUI();
        drawBoth();
    }

    function isTypeVisible(type) {
        return state.visibleTypes[type] !== false;
    }

    function setupCheckboxFilter() {
        document.querySelectorAll('.type-filter').forEach(function (cb) {
            cb.addEventListener('change', function () {
                state.visibleTypes[this.dataset.type] = this.checked;
                drawBoth();
            });
        });
    }

    function setupVizSelector() {
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
                showMatchLines();
            } else {
                hideMatchOverlay();
                state.currentVizType = v;
                var sides = ['original', 'partial'];
                sides.forEach(function (sideName) {
                    var s = state[sideName];
                    var url = s.visualizations && s.visualizations[v];
                    if (url) {
                        s.image.onload = function () { drawCanvas(sideName); };
                        s.image.src = url;
                    }
                });
            }
        };
    }

    function showMatchLines() {
        var overlay = document.getElementById('match-overlay');
        if (overlay) overlay.style.pointerEvents = 'auto';
        var sides = ['original', 'partial'];
        var loaded = 0;
        sides.forEach(function (sideName) {
            var s = state[sideName];
            var url = s.visualizations && s.visualizations.processed;
            if (url) {
                s.image.onload = function () {
                    drawCanvas(sideName);
                    loaded++;
                    if (loaded === 2) drawMatchOverlayLines();
                };
                s.image.src = url;
            } else {
                loaded++;
                if (loaded === 2) drawMatchOverlayLines();
            }
        });
    }

    function getMatchEndpointPos(pair, sideName, canvasRect, wrapperRect) {
        var s = state[sideName];
        var pt = sideName === 'original' ? pair.original : pair.partial;
        var cx = s.canvas.width / 2;
        var cy = s.canvas.height / 2;
        return {
            x: canvasRect.left - wrapperRect.left + (pt.x * s.scale - cx) * s.zoom + cx + s.offsetX,
            y: canvasRect.top  - wrapperRect.top  + (pt.y * s.scale - cy) * s.zoom + cy + s.offsetY
        };
    }

    function drawMatchOverlayLines() {
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
            var oPos = getMatchEndpointPos(pair, 'original', oR, wr);
            var pPos = getMatchEndpointPos(pair, 'partial', pR, wr);

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
    }

    function hideMatchOverlay() {
        var overlay = document.getElementById('match-overlay');
        if (overlay) { overlay.style.display = 'none'; overlay.style.pointerEvents = 'none'; }
    }

    function overlayToImageCoords(overlayX, overlayY, sideName) {
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
    }

    function drawBoth() {
        drawCanvas('original');
        drawCanvas('partial');
    }

    function drawCanvas(sideName) {
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
        var scale = Math.min(w / imgW, h / imgH);
        canvas.width = Math.floor(imgW * scale);
        canvas.height = Math.floor(imgH * scale);
        canvas.style.width = canvas.width + 'px';
        canvas.style.height = canvas.height + 'px';
        s.scale = scale;

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

        // Draw minutiae (both sides always visible)
        var invZ = 1 / s.zoom;
        s.minutiae.forEach(function (m, index) {
            var type = m.landmark_type || 'termination';
            if (state.visibleTypes[type] === false) return;
            var x = m.x * scale;
            var y = m.y * scale;
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

            // Only show symbol+label on active side (to reduce clutter)
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

        // Match dots on both sides
        if (state.matchData && state.matchData.matched_details) {
            var isOrig = sideName === 'original';
            state.matchData.matched_details.forEach(function (pair) {
                var pt = isOrig ? pair.original : pair.partial;
                var x = pt.x * scale;
                var y = pt.y * scale;
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
    }

    function updateUI() {
        var s = cur();
        elements.statusCount().textContent = s.minutiae.length;
        elements.statusAdded().textContent = s.minutiae.filter(function (m) { return m.manually_added; }).length;
        elements.statusDeleted().textContent = state.deletedCount;

        if (s.classification && (s.classification.finger_type || s.classification.pattern_type)) {
            var ft = s.classification.finger_type || 'unknown';
            var pattern = s.classification.pattern_type || '';
            var region = s.classification.region || 'unknown';
            var confidence = s.classification.confidence || 0;
            var ftName = FINGER_TYPE_NAMES_AR[ft] || ft;
            var regionName = REGION_NAMES_AR[region] || region;
            var patternName = pattern ? (PATTERN_NAMES_AR[pattern] || pattern) : '';
            var text = ftName + ' (' + ft + ')';
            if (patternName) text += ' | ' + patternName;
            text += ' | ' + regionName;
            text += ' | ثقة: ' + Math.round(confidence * 100) + '%';
            elements.statusClass().textContent = text;
        } else {
            elements.statusClass().textContent = 'غير متوفر | N/A';
        }

        renderMinutiaeList();
    }

    function renderMinutiaeList() {
        var s = cur();
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
                html += '<span style="padding:2px 6px;border-radius:3px;background:' + (LANDMARK_COLORS[t] || '#ccc') + ';color:#fff;">' + LANDMARK_NAMES_AR[t] + ' ' + counts[t] + '</span>';
            }
        });
        html += '</div>';

        list.innerHTML = html;

        s.minutiae.forEach(function (m, index) {
            var item = document.createElement('div');
            item.className = 'minutia-item' + (index === state.selectedMinutiaIndex ? ' selected' : '');
            var type = m.landmark_type || 'termination';
            var typeName = LANDMARK_NAMES_AR[type] || type;
            item.innerHTML = '<div class="minutia-info">' +
                '<div class="minutia-type" style="color:' + (LANDMARK_COLORS[type] || '#333') + '">' +
                (LANDMARK_SYMBOLS[type] || '•') + ' ' + typeName + '</div>' +
                '<div class="minutia-coords">X: ' + Math.round(m.x) + ', Y: ' + Math.round(m.y) + ' | ' + Math.round(m.angle) + '°</div></div>' +
                '<div class="minutia-actions" style="display:flex;gap:4px;">' +
                '<button class="action-btn edit-type" data-index="' + index + '" title="تعديل النوع | Edit Type" style="background:#FF9800;">✎</button>' +
                '<button class="action-btn delete" data-index="' + index + '" title="حذف | Delete">×</button></div>';

            item.addEventListener('click', function () {
                state.selectedMinutiaIndex = index;
                drawBoth();
                renderMinutiaeList();
            });

            item.querySelector('.edit-type').addEventListener('click', function (e) {
                e.stopPropagation();
                showEditTypeOverlay(index);
            });

            item.querySelector('.delete').addEventListener('click', function (e) {
                e.stopPropagation();
                deleteMinutia(index);
            });

            list.appendChild(item);
        });
    }

    function showEditTypeOverlay(index) {
        var s = cur();
        var m = s.minutiae[index];
        if (!m) return;
        state.selectedMinutiaIndex = index;
        drawBoth();
        renderMinutiaeList();

        var select = elements.editTypeSelect();
        var overlay = elements.editOverlay();
        if (!select || !overlay) return;
        select.innerHTML = '';
        var types = ['termination', 'bifurcation', 'island', 'ridge', 'loop_eye', 'bridge', 'lake', 'dot'];
        types.forEach(function (t) {
            var opt = document.createElement('option');
            opt.value = t;
            opt.textContent = LANDMARK_NAMES_AR[t] + ' | ' + t.charAt(0).toUpperCase() + t.slice(1);
            if (t === m.landmark_type) opt.selected = true;
            select.appendChild(opt);
        });
        overlay.classList.add('show');
    }

    function setupEditOverlay() {
        var overlay = elements.editOverlay();
        var confirmBtn = elements.editConfirmBtn();
        var cancelBtn = elements.editCancelBtn();
        var select = elements.editTypeSelect();
        if (!overlay || !confirmBtn || !cancelBtn) return;

        confirmBtn.addEventListener('click', async function () {
            var idx = state.selectedMinutiaIndex;
            if (idx === -1) return;
            var newType = select.value;
            var s = cur();
            var m = s.minutiae[idx];
            if (!m || m.landmark_type === newType) { overlay.classList.remove('show'); return; }

            try {
                var r = await fetch('/api/editor/update-landmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fingerprint_id: s.fingerprintId, minutia_index: idx, new_landmark_type: newType, reason: 'Manual edit' })
                });
                if (r.ok) await refreshMinutiae();
                else { var err = await r.json(); alert('Error: ' + (err.detail || 'Failed to update')); }
            } catch (error) { console.error('Update failed:', error); }
            overlay.classList.remove('show');
        });

        cancelBtn.addEventListener('click', function () { overlay.classList.remove('show'); });
        overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.classList.remove('show'); });
    }

    function eachSide(fn) {
        fn('original');
        fn('partial');
    }

    function zoomIn() {
        eachSide(function (side) {
            var s = state[side];
            s.zoom = Math.min(s.zoom * 1.2, 10);
        });
        drawBoth();
        if (state.currentVizType === '__match__') drawMatchOverlayLines();
        updateZoomDisplay();
    }

    function zoomOut() {
        eachSide(function (side) {
            var s = state[side];
            s.zoom = Math.max(s.zoom / 1.2, 0.1);
        });
        drawBoth();
        if (state.currentVizType === '__match__') drawMatchOverlayLines();
        updateZoomDisplay();
    }

    function resetZoom() {
        eachSide(function (side) {
            var s = state[side];
            s.zoom = 1; s.offsetX = 0; s.offsetY = 0;
        });
        drawBoth();
        if (state.currentVizType === '__match__') drawMatchOverlayLines();
        updateZoomDisplay();
    }

    function updateZoomDisplay() {
        var el = elements.zoomLevel();
        var s = cur();
        if (el) el.textContent = Math.round(s.zoom * 100) + '%';
    }

    async function refreshMinutiae() {
        var s = cur();
        try {
            var r = await fetch('/api/editor/fingerprint/' + s.fingerprintId);
            if (r.ok) {
                var d = await r.json();
                s.minutiae = d.minutiae || [];
                s.landmarks = d.landmarks || {};
                s.classification = d.classification || {};
                drawBoth();
                updateUI();
            }
        } catch (error) { console.error('Refresh failed:', error); }
    }

    async function addMinutia(x, y) {
        var type = elements.landmarkSelect().value;
        var angle = parseFloat(elements.angleInput().value) || 0;
        var s = cur();
        try {
            var r = await fetch('/api/editor/add-minutia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fingerprint_id: s.fingerprintId, x: x, y: y, landmark_type: type, angle: angle })
            });
            if (r.ok) await refreshMinutiae();
            else { var err = await r.json(); alert('Error: ' + (err.detail || 'Failed to add point')); }
        } catch (error) { console.error('Add failed:', error); }
    }

    async function deleteMinutia(index) {
        var s = cur();
        var origCount = s.minutiae.length;
        try {
            var r = await fetch('/api/editor/delete-minutia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fingerprint_id: s.fingerprintId, minutia_index: index, reason: 'Manual deletion' })
            });
            if (r.ok) {
                state.deletedCount = (state.deletedCount || 0) + 1;
                state.selectedMinutiaIndex = -1;
                await refreshMinutiae();
            } else {
                var err = await r.json();
                console.error('Delete server error:', err);
            }
        } catch (error) {
            console.error('Delete network error:', error);
        }
    }

    function getCanvasSide(canvasEl) {
        if (canvasEl === state.original.canvas) return 'original';
        if (canvasEl === state.partial.canvas) return 'partial';
        return null;
    }

    function setupEventListeners() {
        var cvs = [state.original.canvas, state.partial.canvas];

        cvs.forEach(function (cv) {
            if (!cv) return;
            cv.addEventListener('contextmenu', function (e) { e.preventDefault(); });

            cv.addEventListener('mousedown', function (e) {
                var sideName = getCanvasSide(this);
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

                // Set active side on click
                setActiveSide(sideName);
                var s = cur();

                var rect = this.getBoundingClientRect();
                // Convert CSS-space mouse coords to canvas attribute space
                var px = (e.clientX - rect.left) * (s.canvas.width / rect.width);
                var py = (e.clientY - rect.top) * (s.canvas.height / rect.height);
                var cx = s.canvas.width / 2;
                var cy = s.canvas.height / 2;
                var imgX = ((px - cx - s.offsetX) / s.zoom + cx) / s.scale;
                var imgY = ((py - cy - s.offsetY) / s.zoom + cy) / s.scale;

                var currentMode = elements.modeSelect().value;

                if (currentMode === 'add') {
                    addMinutia(imgX, imgY);
                } else if (currentMode === 'move') {
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
                        drawBoth();
                        renderMinutiaeList();
                    }
                } else {
                    // delete / edit / view: find nearest point
                    var nearestIdx = -1, minDist = 40;
                    s.minutiae.forEach(function (m, idx) {
                        var sx = (m.x * s.scale - cx) * s.zoom + cx + s.offsetX;
                        var sy = (m.y * s.scale - cy) * s.zoom + cy + s.offsetY;
                        var d = Math.sqrt((sx - px) * (sx - px) + (sy - py) * (sy - py));
                        if (d < minDist) { minDist = d; nearestIdx = idx; }
                    });
                    if (nearestIdx !== -1) {
                        state.selectedMinutiaIndex = nearestIdx;
                        drawBoth();
                        renderMinutiaeList();
                        if (currentMode === 'delete') {
                            deleteMinutia(nearestIdx);
                        } else if (currentMode === 'edit') {
                            showEditTypeOverlay(nearestIdx);
                        }
                    } else {
                        console.log('No point within 40px. px=' + px + ' py=' + py + ' canvasW=' + s.canvas.width + ' rectW=' + rect.width);
                    }
                }
            });

            cv.addEventListener('wheel', function (e) {
                e.preventDefault();
                setActiveSide(getCanvasSide(this));
                if (e.deltaY < 0) zoomIn(); else zoomOut();
            }, { passive: false });
        });

        // Match overlay dragging
        var overlay = document.getElementById('match-overlay');
        if (overlay) {
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
                    var oPos = getMatchEndpointPos(pair, 'original', oR, wr);
                    var pPos = getMatchEndpointPos(pair, 'partial', pR, wr);
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
                var ic = overlayToImageCoords(ox, oy, side);
                pt.x = ic.x;
                pt.y = ic.y;
                drawMatchOverlayLines();
                drawBoth();
                return;
            }
            if (state.isMovingMinutia) {
                var s = cur();
                var rect = s.canvas.getBoundingClientRect();
                var px = (e.clientX - rect.left) * (s.canvas.width / rect.width);
                var py = (e.clientY - rect.top) * (s.canvas.height / rect.height);
                var cx = s.canvas.width / 2;
                var cy = s.canvas.height / 2;
                var imgX = ((px - cx - s.offsetX) / s.zoom + cx) / s.scale;
                var imgY = ((py - cy - s.offsetY) / s.zoom + cy) / s.scale;
                var m = s.minutiae[state.movingMinutiaIndex];
                if (m) { m.x = Math.round(imgX); m.y = Math.round(imgY); drawBoth(); renderMinutiaeList(); }
                return;
            }
            if (state.isDragging) {
                eachSide(function (side) {
                    var s = state[side];
                    s.offsetX = state.dragStartOffset[side].x + (e.clientX - state.dragStartX);
                    s.offsetY = state.dragStartOffset[side].y + (e.clientY - state.dragStartY);
                });
                drawBoth();
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
                var s = cur();
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
                var s = cur();
                s.canvas.style.cursor = 'crosshair';
            }
        });

        if (elements.btnZoomIn()) elements.btnZoomIn().addEventListener('click', zoomIn);
        if (elements.btnZoomOut()) elements.btnZoomOut().addEventListener('click', zoomOut);
        if (elements.btnZoomReset()) elements.btnZoomReset().addEventListener('click', resetZoom);

        elements.modeSelect().addEventListener('change', function () {
            state.selectedMinutiaIndex = -1;
            // Show landmark type selector only in add mode
            var lmGroup = document.querySelector('.control-group:has(#landmark-select)');
            var angleGroup = document.querySelector('.control-group:has(#angle-input)');
            var isAdd = this.value === 'add';
            if (lmGroup) lmGroup.style.display = isAdd ? '' : 'none';
            if (angleGroup) angleGroup.style.display = isAdd ? '' : 'none';
            drawBoth();
        });
        // Initial hide
        elements.modeSelect().dispatchEvent(new Event('change'));

        elements.btnReset().addEventListener('click', function () {
            if (confirm('Reset all changes?')) {
                var fpId = cur().fingerprintId;
                loadSide(state.activeSide);
            }
        });

        elements.btnApprove().addEventListener('click', function () { elements.approveModal().classList.add('show'); });
        elements.btnReject().addEventListener('click', function () { elements.rejectModal().classList.add('show'); });

        elements.confirmApprove().addEventListener('click', async function () {
            var s = cur();
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
            var s = cur();
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

        window.addEventListener('resize', function () { drawBoth(); });
    }

    setupEditOverlay();

    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', init);
    else
        init();
})();
