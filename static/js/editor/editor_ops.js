(function () {
    "use strict";
    var App = window.EditorApp;
    if (!App) return;
    var state = App.state;
    var elements = App.elements;

    App.refreshMinutiae = async function () {
        var s = App.cur();
        try {
            var r = await fetch('/api/editor/fingerprint/' + s.fingerprintId + '?_=' + Date.now());
            if (r.ok) {
                var d = await r.json();
                var oldCount = s.minutiae.length;
                s.minutiae = d.minutiae || [];
                s.landmarks = d.landmarks || {};
                s.classification = d.classification || {};
                console.log('refresh: side=' + state.activeSide + ' old=' + oldCount + ' new=' + s.minutiae.length);
                App.drawBoth();
                App.updateUI();
            } else {
                App.showToast('فشل تحديث البيانات', true);
            }
        } catch (error) { console.error('Refresh failed:', error); App.showToast('خطأ في التحديث', true); }
    };

    App.addMinutia = async function (x, y) {
        var type = elements.landmarkSelect().value;
        var angle = parseFloat(elements.angleInput().value) || 0;
        var s = App.cur();
        try {
            var r = await fetch('/api/editor/add-minutia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fingerprint_id: s.fingerprintId, x: x, y: y, landmark_type: type, angle: angle })
            });
            if (r.ok) {
                App.showToast('تمت الإضافة ✓');
                await App.refreshMinutiae();
            } else { var err = await r.json(); App.showToast('خطأ: ' + (err.detail || 'فشلت الإضافة'), true); }
        } catch (error) { console.error('Add failed:', error); App.showToast('تعذر الاتصال بالخادم', true); }
    };

    App.deleteMinutia = async function (index) {
        var s = App.cur();
        try {
            var r = await fetch('/api/editor/delete-minutia', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ fingerprint_id: s.fingerprintId, minutia_index: index, reason: 'Manual deletion' })
            });
            if (r.ok) {
                state.deletedCount = (state.deletedCount || 0) + 1;
                state.selectedMinutiaIndex = -1;
                App.showToast('تم الحذف ✓');
                await App.refreshMinutiae();
            } else {
                var err = await r.json();
                console.error('Delete server error:', err);
                App.showToast('فشل الحذف: ' + (err.detail || 'خطأ في الخادم'), true);
            }
        } catch (error) {
            console.error('Delete network error:', error);
            App.showToast('تعذر الاتصال بالخادم', true);
        }
    };

    App.showEditTypeOverlay = function (index) {
        var s = App.cur();
        var m = s.minutiae[index];
        if (!m) return;
        state.selectedMinutiaIndex = index;
        App.drawBoth();
        App.renderMinutiaeList();

        var select = elements.editTypeSelect();
        var overlay = elements.editOverlay();
        if (!select || !overlay) return;
        select.innerHTML = '';
        var types = ['termination', 'bifurcation', 'island', 'ridge', 'loop_eye', 'bridge', 'lake', 'dot'];
        types.forEach(function (t) {
            var opt = document.createElement('option');
            opt.value = t;
            opt.textContent = App.LANDMARK_NAMES_AR[t] + ' | ' + t.charAt(0).toUpperCase() + t.slice(1);
            if (t === m.landmark_type) opt.selected = true;
            select.appendChild(opt);
        });
        overlay.classList.add('show');
    };

    App.setupEditOverlay = function () {
        var overlay = elements.editOverlay();
        var confirmBtn = elements.editConfirmBtn();
        var cancelBtn = elements.editCancelBtn();
        var select = elements.editTypeSelect();
        if (!overlay || !confirmBtn || !cancelBtn) return;

        confirmBtn.addEventListener('click', async function () {
            var idx = state.selectedMinutiaIndex;
            if (idx === -1) return;
            var newType = select.value;
            var s = App.cur();
            var m = s.minutiae[idx];
            if (!m || m.landmark_type === newType) { overlay.classList.remove('show'); return; }

            try {
                var r = await fetch('/api/editor/update-landmark', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ fingerprint_id: s.fingerprintId, minutia_index: idx, new_landmark_type: newType, reason: 'Manual edit' })
                });
                if (r.ok) { App.showToast('تم التعديل ✓'); await App.refreshMinutiae(); }
                else { var err = await r.json(); App.showToast('خطأ: ' + (err.detail || 'فشل التعديل'), true); }
            } catch (error) { console.error('Update failed:', error); App.showToast('تعذر الاتصال بالخادم', true); }
            overlay.classList.remove('show');
        });

        cancelBtn.addEventListener('click', function () { overlay.classList.remove('show'); });
        overlay.addEventListener('click', function (e) { if (e.target === overlay) overlay.classList.remove('show'); });
    };
})();
