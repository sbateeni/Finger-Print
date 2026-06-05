(function () {
    "use strict";
    window.EditorApp = window.EditorApp || {};

    var App = window.EditorApp;

    App.state = {
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

    App.LANDMARK_COLORS = { termination: '#FF6B6B', bifurcation: '#4ECDC4', island: '#FFD93D', ridge: '#A8E6CF', loop_eye: '#FF8B94', bridge: '#C7CEEA', lake: '#B5EAD7', dot: '#95E1D3' };
    App.LANDMARK_SYMBOLS = { termination: '◇', bifurcation: '⊢', island: '⊗', ridge: '─', loop_eye: '◯', bridge: '⌢', lake: '◈', dot: '•' };
    App.LANDMARK_NAMES_AR = { termination: 'نهاية', bifurcation: 'تفرع', island: 'جزيرة', ridge: 'شرطة', loop_eye: 'عين', bridge: 'جسر', lake: 'بحيرة', dot: 'نقطة' };
    App.PATTERN_NAMES_AR = { arch: 'قوس', tented_arch: 'قوس خيمي', left_loop: 'أنشوطة يسرى', right_loop: 'أنشوطة يمنى', whorl: 'دوامة' };
    App.FINGER_TYPE_NAMES_AR = { thumb: 'إبهام', index: 'سبابة', middle: 'وسطى', ring: 'بنصر', pinky: 'خنصر', unknown: 'غير معروف' };
    App.REGION_NAMES_AR = { fingertip: 'طرف الإصبع', palm_root: 'جذور الأصابع', sub_index: 'تحت السبابة', palm_general: 'راحة اليد', unknown: 'غير معروف' };

    App.elements = {
        canvasOriginal: function () { return document.getElementById('canvas-original'); },
        canvasPartial: function () { return document.getElementById('canvas-partial'); },
        modeSelect: function () { return document.getElementById('mode-select'); },
        landmarkSelect: function () { return document.getElementById('landmark-select'); },
        angleInput: function () { return document.getElementById('angle-input'); },
        statusCount: function () { return document.getElementById('status-count'); },
        statusAdded: function () { return document.getElementById('status-added'); },
        statusDeleted: function () { return document.getElementById('status-deleted'); },
        statusClass: function () { return document.getElementById('status-class'); },
        minutiaeList: function () { return document.getElementById('minutiae-list'); },
        btnAddManual: function () { return document.getElementById('btn-add-manual'); },
        btnDeleteSelected: function () { return document.getElementById('btn-delete-selected'); },
        btnUndo: function () { return document.getElementById('btn-undo'); },
        btnReset: function () { return document.getElementById('btn-reset'); },
        btnApprove: function () { return document.getElementById('btn-approve'); },
        btnReject: function () { return document.getElementById('btn-reject'); },
        approveModal: function () { return document.getElementById('approve-modal'); },
        rejectModal: function () { return document.getElementById('reject-modal'); },
        confirmApprove: function () { return document.getElementById('confirm-approve'); },
        confirmReject: function () { return document.getElementById('confirm-reject'); },
        approveNotes: function () { return document.getElementById('approve-notes'); },
        rejectReason: function () { return document.getElementById('reject-reason'); },
        zoomLevel: function () { return document.getElementById('zoom-level'); },
        btnZoomIn: function () { return document.getElementById('btn-zoom-in'); },
        btnZoomOut: function () { return document.getElementById('btn-zoom-out'); },
        btnZoomReset: function () { return document.getElementById('btn-zoom-reset'); },
        editTypeSelect: function () { return document.getElementById('edit-type-select'); },
        editConfirmBtn: function () { return document.getElementById('edit-confirm-btn'); },
        editCancelBtn: function () { return document.getElementById('edit-cancel-btn'); },
        editOverlay: function () { return document.getElementById('edit-overlay'); },
        vizSelect: function () { return document.getElementById('viz-select'); },
        boxOriginal: function () { return document.getElementById('fp-box-original'); },
        boxPartial: function () { return document.getElementById('fp-box-partial'); },
        statusToast: function () { return document.getElementById('status-toast'); },
    };

    App.cur = function () { return App.state[App.state.activeSide]; };

    App.isTypeVisible = function (type) {
        return App.state.visibleTypes[type] !== false;
    };

    App.getCanvasSide = function (canvasEl) {
        if (canvasEl === App.state.original.canvas) return 'original';
        if (canvasEl === App.state.partial.canvas) return 'partial';
        return null;
    };

    App.eachSide = function (fn) {
        fn('original');
        fn('partial');
    };

    App.showToast = function (msg, isError) {
        var el = App.elements.statusToast();
        if (!el) return;
        el.textContent = msg;
        el.style.background = isError ? '#ffebee' : '#e8f5e9';
        el.style.color = isError ? '#c62828' : '#2e7d32';
        el.style.border = '1px solid ' + (isError ? '#ef9a9a' : '#a5d6a7');
        el.style.padding = '6px 10px';
    };

    App.renderNoIdMessage = function (urlParams) {
        var container = document.querySelector('.main-content');
        if (container) {
            container.innerHTML = [
                '<div class="panel" style="grid-column: 1 / -1; text-align: center; padding: 3rem;">',
                '  <h2 style="color: #555; margin-bottom: 1rem;">' + (document.querySelector('header h1')?.textContent || 'Manual Editor') + '</h2>',
                '  <div class="warning-box" style="display: inline-block; text-align: right;">',
                '    <strong>⚠️ لم يتم تحديد بصمة</strong>',
                '    <p style="margin: 0.75rem 0; font-size: 15px;">يرجى تحليل بصمة أولاً من الصفحة الرئيسية ثم الدخول إلى المحرر.</p>',
                '    <p style="margin: 0.75rem 0; font-size: 15px; direction: ltr;">No fingerprint selected. Please analyze a fingerprint from the main page first.</p>',
                '    <a href="/?lang=' + (urlParams.get('lang') || 'ar') + '" style="display: inline-block; margin-top: 1rem; padding: 0.75rem 2rem; background: #667eea; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">',
                '      ← العودة إلى الصفحة الرئيسية',
                '    </a>',
                '  </div>',
                '</div>'
            ].join('\n');
        }
    };
})();
