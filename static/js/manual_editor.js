(function () {
    "use strict";

    const state = {
        fingerprintId: null,
        originalId: null,
        partialId: null,
        image: new Image(),
        minutiae: [],
        landmarks: {},
        classification: {},
        visualizations: {},
        selectedMinutiaIndex: -1,
        scale: 1,
        zoom: 1,
        offsetX: 0,
        offsetY: 0,
        canvas: null,
        ctx: null,
        mode: 'add',
        history: [],
        isDragging: false,
        dragStartX: 0,
        dragStartY: 0,
        dragOffsetX: 0,
        dragOffsetY: 0,
        currentFpLabel: 'البصمة الأصلية',
    };

    const LANDMARK_COLORS = {
        termination: '#FF6B6B',
        bifurcation: '#4ECDC4',
        island: '#FFD93D',
        ridge: '#A8E6CF',
        loop_eye: '#FF8B94',
        bridge: '#C7CEEA',
        lake: '#B5EAD7',
        dot: '#95E1D3'
    };

    const LANDMARK_SYMBOLS = {
        termination: '◇',
        bifurcation: '⊢',
        island: '⊗',
        ridge: '─',
        loop_eye: '◯',
        bridge: '⌢',
        lake: '◈',
        dot: '•'
    };

    const LANDMARK_NAMES_AR = {
        termination: 'نهاية',
        bifurcation: 'تفرع',
        island: 'جزيرة',
        ridge: 'شرطة',
        loop_eye: 'عين',
        bridge: 'جسر',
        lake: 'بحيرة',
        dot: 'نقطة'
    };

    const PATTERN_NAMES_AR = {
        arch: 'قوس',
        tented_arch: 'قوس خيمي',
        left_loop: 'أنشوطة يسرى',
        right_loop: 'أنشوطة يمنى',
        whorl: 'دوامة',
    };

    const FINGER_TYPE_NAMES_AR = {
        thumb: 'إبهام',
        index: 'سبابة',
        middle: 'وسطى',
        ring: 'بنصر',
        pinky: 'خنصر',
        unknown: 'غير معروف'
    };

    const REGION_NAMES_AR = {
        fingertip: 'طرف الإصبع',
        palm_root: 'جذور الأصابع',
        sub_index: 'تحت السبابة',
        palm_general: 'راحة اليد',
        unknown: 'غير معروف'
    };

    const elements = {
        canvas: () => document.getElementById('fingerprint-canvas'),
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
                </div>
            `;
        }
    }

    function renderFpSelector() {
        const container = document.querySelector('.fp-selector');
        if (!container) return;
        var lang = new URLSearchParams(window.location.search).get('lang') || 'ar';
        container.innerHTML = '';
        var fps = [
            { id: state.originalId, label: 'البصمة الأصلية | Original', cls: state.fingerprintId == state.originalId ? 'active' : '' },
            { id: state.partialId, label: 'البصمة المقارنة | Partial', cls: state.fingerprintId == state.partialId ? 'active' : '' },
        ];
        fps.forEach(function (fp) {
            if (!fp.id) return;
            var btn = document.createElement('button');
            btn.className = 'fp-select-btn ' + fp.cls;
            btn.textContent = fp.label;
            btn.addEventListener('click', function () {
                switchFingerprint(fp.id);
            });
            container.appendChild(btn);
        });
    }

    async function switchFingerprint(fpId) {
        if (!fpId || fpId == state.fingerprintId) return;
        state.fingerprintId = fpId;
        state.currentFpLabel = fpId == state.originalId ? 'البصمة الأصلية' : 'البصمة المقارنة';
        state.selectedMinutiaIndex = -1;
        document.querySelector('.panel h2').textContent = state.currentFpLabel + ' | Fingerprint';
        renderFpSelector();
        await loadFingerprintData();
    }

    async function init() {
        const urlParams = new URLSearchParams(window.location.search);
        state.fingerprintId = urlParams.get('fingerprint_id');
        state.originalId = urlParams.get('original_id');
        state.partialId = urlParams.get('partial_id');
        
        if (!state.fingerprintId && state.originalId) {
            state.fingerprintId = state.originalId;
        }
        
        if (!state.fingerprintId) {
            renderNoIdMessage(urlParams);
            return;
        }

        state.canvas = elements.canvas();
        state.ctx = state.canvas.getContext('2d');

        renderFpSelector();
        setupEventListeners();
        await loadFingerprintData();
    }

    async function loadFingerprintData() {
        try {
            const response = await fetch(`/api/editor/fingerprint/${state.fingerprintId}`);
            if (!response.ok) throw new Error('Failed to fetch data');
            
            const data = await response.json();
            state.minutiae = data.minutiae || [];
            state.landmarks = data.landmarks || {};
            state.classification = data.classification || {};
            state.visualizations = data.visualizations || {};
            
            setupVizSelector(state.visualizations);
            
            state.image.onload = () => {
                resizeCanvas();
                draw();
                updateUI();
            };
            state.image.src = data.image_url;
            
        } catch (error) {
            console.error('Error loading data:', error);
            alert('Error loading fingerprint data');
        }
    }

    function setupVizSelector(viz) {
        var sel = elements.vizSelect();
        if (!sel) return;
        var keys = Object.keys(viz);
        if (keys.length <= 1) { sel.style.display = 'none'; return; }
        sel.style.display = '';
        sel.innerHTML = '';
        var labels = { processed: 'معالجة (Processed)', ridges: 'تموجات (Ridges)', skeleton: 'هيكل (Skeleton)' };
        keys.forEach(function (k) {
            var opt = document.createElement('option');
            opt.value = k;
            opt.textContent = labels[k] || k;
            sel.appendChild(opt);
        });
        sel.value = 'processed';
        sel.onchange = function () {
            var url = viz[sel.value];
            if (url) {
                state.image.src = url;
            }
        };
    }

    function resizeCanvas() {
        const container = state.canvas.parentElement;
        const maxWidth = container.clientWidth - 4;
        const maxHeight = 600;

        const imgWidth = state.image.width;
        const imgHeight = state.image.height;

        const ratio = Math.min(maxWidth / imgWidth, maxHeight / imgHeight);
        
        state.canvas.width = maxWidth;
        state.canvas.height = Math.min(imgHeight * ratio, maxHeight);
        state.scale = ratio;
        state.zoom = 1;
        state.offsetX = 0;
        state.offsetY = 0;
        updateZoomDisplay();
    }

    function draw() {
        if (!state.image.complete) return;

        const { ctx, canvas, image, scale, zoom, offsetX, offsetY, minutiae, selectedMinutiaIndex } = state;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Apply zoom and pan
        ctx.save();
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        ctx.translate(cx + offsetX, cy + offsetY);
        ctx.scale(zoom, zoom);
        ctx.translate(-cx, -cy);

        // Draw image with current scale
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

        // Draw minutiae
        minutiae.forEach((m, index) => {
            const x = m.x * scale;
            const y = m.y * scale;
            const type = m.landmark_type || 'termination';
            const color = LANDMARK_COLORS[type] || '#FF0000';
            const isSelected = index === selectedMinutiaIndex;

            // Outer glow for selection
            if (isSelected) {
                ctx.beginPath();
                ctx.arc(x, y, 14, 0, Math.PI * 2);
                ctx.fillStyle = 'rgba(255,255,255,0.3)';
                ctx.fill();
                ctx.strokeStyle = '#FFFFFF';
                ctx.lineWidth = 3;
                ctx.stroke();
            }

            // Point circle
            const radius = isSelected ? 8 : 6;
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Symbol label
            ctx.fillStyle = '#000000';
            ctx.font = 'bold 16px Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(LANDMARK_SYMBOLS[type] || '•', x, y - 20);

            // Type name
            ctx.fillStyle = isSelected ? '#FFFFFF' : color;
            ctx.font = 'bold 11px Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText(LANDMARK_NAMES_AR[type] || type, x, y + radius + 4);
        });

        ctx.restore();

        // Draw zoom indicator
        ctx.fillStyle = 'rgba(0,0,0,0.6)';
        ctx.fillRect(8, 8, 120, 24);
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial, sans-serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('Zoom: ' + Math.round(zoom * 100) + '%', 14, 20);
    }

    function updateZoomDisplay() {
        const el = elements.zoomLevel();
        if (el) el.textContent = Math.round(state.zoom * 100) + '%';
    }

    function updateUI() {
        elements.statusCount().textContent = state.minutiae.length;
        elements.statusAdded().textContent = state.minutiae.filter(m => m.manually_added).length;
        
        if (state.classification && (state.classification.finger_type || state.classification.pattern_type)) {
            var ft = state.classification.finger_type || 'unknown';
            var pattern = state.classification.pattern_type || '';
            var region = state.classification.region || 'unknown';
            var confidence = state.classification.confidence || 0;
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
        const list = elements.minutiaeList();
        list.innerHTML = '';

        if (state.minutiae.length === 0) {
            list.innerHTML = '<div style="padding: 20px; text-align: center; color: #999;">لم يتم تحميل أي نقاط | No minutiae loaded</div>';
            return;
        }

        state.minutiae.forEach((m, index) => {
            const item = document.createElement('div');
            item.className = `minutia-item ${index === state.selectedMinutiaIndex ? 'selected' : ''}`;
            
            const type = m.landmark_type || 'termination';
            const typeName = LANDMARK_NAMES_AR[type] || type;
            
            item.innerHTML = `
                <div class="minutia-info">
                    <div class="minutia-type" style="color: ${LANDMARK_COLORS[type] || '#333'}">
                        ${LANDMARK_SYMBOLS[type] || '•'} ${typeName}
                    </div>
                    <div class="minutia-coords">X: ${Math.round(m.x)}, Y: ${Math.round(m.y)} | ${Math.round(m.angle)}°</div>
                </div>
                <div class="minutia-actions" style="display:flex;gap:4px;">
                    <button class="action-btn edit-type" data-index="${index}" title="تعديل النوع | Edit Type" style="background:#FF9800;">✎</button>
                    <button class="action-btn delete" data-index="${index}" title="حذف | Delete">×</button>
                </div>
            `;

            item.addEventListener('click', () => {
                state.selectedMinutiaIndex = index;
                draw();
                renderMinutiaeList();
            });

            const editBtn = item.querySelector('.edit-type');
            editBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                showEditTypeOverlay(index);
            });

            const delBtn = item.querySelector('.delete');
            delBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteMinutia(index);
            });

            list.appendChild(item);
        });
    }

    function showEditTypeOverlay(index) {
        const m = state.minutiae[index];
        if (!m) return;

        state.selectedMinutiaIndex = index;
        draw();
        renderMinutiaeList();

        const select = elements.editTypeSelect();
        const overlay = elements.editOverlay();
        if (!select || !overlay) return;

        select.innerHTML = '';
        const types = ['termination', 'bifurcation', 'island', 'ridge', 'loop_eye', 'bridge', 'lake', 'dot'];
        types.forEach(t => {
            const opt = document.createElement('option');
            opt.value = t;
            opt.textContent = LANDMARK_NAMES_AR[t] + ' | ' + t.charAt(0).toUpperCase() + t.slice(1);
            if (t === m.landmark_type) opt.selected = true;
            select.appendChild(opt);
        });

        overlay.classList.add('show');
    }

    function setupEditOverlay() {
        const overlay = elements.editOverlay();
        const confirmBtn = elements.editConfirmBtn();
        const cancelBtn = elements.editCancelBtn();
        const select = elements.editTypeSelect();
        if (!overlay || !confirmBtn || !cancelBtn) return;

        confirmBtn.addEventListener('click', async () => {
            const idx = state.selectedMinutiaIndex;
            if (idx === -1) return;
            const newType = select.value;
            const m = state.minutiae[idx];
            if (!m || m.landmark_type === newType) {
                overlay.classList.remove('show');
                return;
            }

            try {
                const response = await fetch('/api/editor/update-landmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        fingerprint_id: state.fingerprintId,
                        minutia_index: idx,
                        new_landmark_type: newType,
                        reason: 'Manual edit'
                    })
                });

                if (response.ok) {
                    await refreshMinutiae();
                } else {
                    const err = await response.json();
                    alert('Error: ' + (err.detail || 'Failed to update'));
                }
            } catch (error) {
                console.error('Update failed:', error);
            }
            overlay.classList.remove('show');
        });

        cancelBtn.addEventListener('click', () => {
            overlay.classList.remove('show');
        });

        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) overlay.classList.remove('show');
        });
    }

    function zoomIn() {
        state.zoom = Math.min(state.zoom * 1.2, 10);
        draw();
        updateZoomDisplay();
    }

    function zoomOut() {
        state.zoom = Math.max(state.zoom / 1.2, 0.1);
        draw();
        updateZoomDisplay();
    }

    function resetZoom() {
        state.zoom = 1;
        state.offsetX = 0;
        state.offsetY = 0;
        draw();
        updateZoomDisplay();
    }

    async function refreshMinutiae() {
        try {
            const response = await fetch(`/api/editor/fingerprint/${state.fingerprintId}`);
            if (response.ok) {
                const data = await response.json();
                state.minutiae = data.minutiae || [];
                state.landmarks = data.landmarks || {};
                state.classification = data.classification || {};
                draw();
                updateUI();
            }
        } catch (error) {
            console.error('Refresh failed:', error);
        }
    }

    async function addMinutia(x, y) {
        const type = elements.landmarkSelect().value;
        const angle = parseFloat(elements.angleInput().value) || 0;

        try {
            const response = await fetch('/api/editor/add-minutia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fingerprint_id: state.fingerprintId,
                    x: x,
                    y: y,
                    landmark_type: type,
                    angle: angle
                })
            });

            if (response.ok) {
                await refreshMinutiae();
            } else {
                const err = await response.json();
                alert('Error: ' + (err.detail || 'Failed to add point'));
            }
        } catch (error) {
            console.error('Add failed:', error);
        }
    }

    async function deleteMinutia(index) {
        if (!confirm('Are you sure you want to delete this point?')) return;

        try {
            const response = await fetch('/api/editor/delete-minutia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    fingerprint_id: state.fingerprintId,
                    minutia_index: index,
                    reason: 'Manual deletion'
                })
            });

            if (response.ok) {
                state.selectedMinutiaIndex = -1;
                await refreshMinutiae();
            } else {
                const err = await response.json();
                alert('Failed to delete: ' + (err.detail || 'Unknown error'));
            }
        } catch (error) {
            console.error('Delete failed:', error);
        }
    }

    function setupEventListeners() {
        // Prevent context menu on canvas
        state.canvas.addEventListener('contextmenu', (e) => e.preventDefault());

        // Canvas mousedown: left button for points, right button for pan
        state.canvas.addEventListener('mousedown', (e) => {
            // Right-click always starts panning
            if (e.button === 2) {
                state.isDragging = true;
                state.dragStartX = e.clientX;
                state.dragStartY = e.clientY;
                state.dragOffsetX = state.offsetX;
                state.dragOffsetY = state.offsetY;
                state.canvas.style.cursor = 'grabbing';
                return;
            }

            // Left click for actions
            if (e.button !== 0) return;

            const rect = state.canvas.getBoundingClientRect();
            const canvasX = e.clientX - rect.left;
            const canvasY = e.clientY - rect.top;

            // Convert to image coordinates accounting for zoom/pan
            const cx = state.canvas.width / 2;
            const cy = state.canvas.height / 2;
            const imgX = ((canvasX - cx - state.offsetX) / state.zoom + cx) / state.scale;
            const imgY = ((canvasY - cy - state.offsetY) / state.zoom + cy) / state.scale;

            const currentMode = elements.modeSelect().value;

            if (currentMode === 'add') {
                addMinutia(imgX, imgY);
            } else if (currentMode === 'delete' || currentMode === 'edit' || currentMode === 'view') {
                let nearestIdx = -1;
                let minDist = 25;

                state.minutiae.forEach((m, idx) => {
                    const dx = (m.x * state.scale - imgX * state.scale) * state.zoom;
                    const dy = (m.y * state.scale - imgY * state.scale) * state.zoom;
                    const dist = Math.sqrt(dx*dx + dy*dy);
                    if (dist < minDist) {
                        minDist = dist;
                        nearestIdx = idx;
                    }
                });

                if (nearestIdx !== -1) {
                    state.selectedMinutiaIndex = nearestIdx;
                    if (currentMode === 'delete') {
                        deleteMinutia(nearestIdx);
                    } else if (currentMode === 'edit') {
                        showEditTypeOverlay(nearestIdx);
                    } else {
                        draw();
                        renderMinutiaeList();
                    }
                }
            }
        });

        // Mouse wheel zoom
        state.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            if (e.deltaY < 0) {
                zoomIn();
            } else {
                zoomOut();
            }
        }, { passive: false });

        // Global mouse move/up for dragging
        window.addEventListener('mousemove', (e) => {
            if (state.isDragging) {
                state.offsetX = state.dragOffsetX + (e.clientX - state.dragStartX);
                state.offsetY = state.dragOffsetY + (e.clientY - state.dragStartY);
                draw();
            }
        });

        window.addEventListener('mouseup', () => {
            if (state.isDragging) {
                state.isDragging = false;
                state.canvas.style.cursor = 'crosshair';
            }
        });

        // Zoom buttons
        if (elements.btnZoomIn()) elements.btnZoomIn().addEventListener('click', zoomIn);
        if (elements.btnZoomOut()) elements.btnZoomOut().addEventListener('click', zoomOut);
        if (elements.btnZoomReset()) elements.btnZoomReset().addEventListener('click', resetZoom);

        elements.modeSelect().addEventListener('change', () => {
            state.selectedMinutiaIndex = -1;
            draw();
        });

        elements.btnReset().addEventListener('click', () => {
            if (confirm('Reset all changes?')) loadFingerprintData();
        });

        // Approve / Reject
        elements.btnApprove().addEventListener('click', () => {
            elements.approveModal().classList.add('show');
        });

        elements.btnReject().addEventListener('click', () => {
            elements.rejectModal().classList.add('show');
        });

        elements.confirmApprove().addEventListener('click', async () => {
            try {
                const response = await fetch('/api/editor/approve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        fingerprint_id: state.fingerprintId,
                        minutiae: state.minutiae,
                        notes: elements.approveNotes().value,
                        user_id: 1
                    })
                });

                if (response.ok) {
                    alert('Fingerprint approved successfully!');
                    window.close();
                } else {
                    alert('Failed to approve');
                }
            } catch (error) {
                console.error('Approve failed:', error);
            }
        });

        elements.confirmReject().addEventListener('click', async () => {
            try {
                const response = await fetch('/api/editor/reject', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        fingerprint_id: state.fingerprintId,
                        reason: elements.rejectReason().value,
                        user_id: 1
                    })
                });

                if (response.ok) {
                    alert('Fingerprint rejected');
                    window.close();
                } else {
                    alert('Failed to reject');
                }
            } catch (error) {
                console.error('Reject failed:', error);
            }
        });

        window.closeModal = (id) => {
            document.getElementById(id).classList.remove('show');
        };

        window.addEventListener('resize', () => {
            resizeCanvas();
            draw();
        });
    }

    setupEditOverlay();

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();