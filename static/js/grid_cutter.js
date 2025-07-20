document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fullFingerprintInput = document.getElementById('fullFingerprint');
    const fullFingerprintPreview = document.getElementById('fullFingerprintPreview');
    const uploadSuccess = document.getElementById('uploadSuccess');
    const loadingArea = document.getElementById('loadingArea');
    const resultsArea = document.getElementById('resultsArea');
    const gridsContainer = document.getElementById('gridsContainer');
    const debugInfo = document.getElementById('debugInfo');
    const debugContent = document.getElementById('debugContent');

    function showDebugInfo(info) {
        debugContent.textContent = JSON.stringify(info, null, 2);
        debugInfo.classList.remove('d-none');
    }

    // عرض معاينة الصورة عند اختيارها
    function showImagePreview(input, previewElement) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.innerHTML = `<img src="${e.target.result}" class="img-fluid" alt="معاينة البصمة">`;
                previewElement.classList.add('has-image');
                uploadSuccess.classList.remove('d-none');
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    fullFingerprintInput.addEventListener('change', function() {
        showImagePreview(this, fullFingerprintPreview);
    });

    // معالجة تقديم النموذج
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!fullFingerprintInput.files[0]) {
            alert('الرجاء اختيار صورة البصمة');
            return;
        }

        // إظهار منطقة التحميل
        loadingArea.classList.remove('d-none');
        resultsArea.classList.add('d-none');
        gridsContainer.innerHTML = '';
        debugInfo.classList.add('d-none');

        const formData = new FormData();
        formData.append('fullFingerprint', fullFingerprintInput.files[0]);

        try {
            const response = await fetch('/process_grids', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'حدث خطأ أثناء معالجة الصورة');
            }

            // عرض معلومات التصحيح
            showDebugInfo(data);
            
            // عرض الصورة التوضيحية للشبكة
            const gridVisualization = document.createElement('div');
            gridVisualization.className = 'col-12 mb-4';
            gridVisualization.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">تقسيم البصمة إلى مربعات</h5>
                    </div>
                    <div class="card-body">
                        <img src="${data.visualization}" class="img-fluid" alt="تقسيم البصمة" 
                             onerror="this.onerror=null; this.src=''; this.alt='خطأ في تحميل الصورة'; this.classList.add('error');">
                        <div class="mt-2 text-center">
                            <small class="text-muted">
                                تم تقسيم البصمة إلى ${data.grid_info.rows} صفوف و ${data.grid_info.cols} أعمدة
                            </small>
                        </div>
                    </div>
                </div>
            `;
            gridsContainer.appendChild(gridVisualization);
            
            // عرض المربعات المقصوصة
            const gridsWrapper = document.createElement('div');
            gridsWrapper.className = 'col-12';
            gridsWrapper.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">المربعات المقصوصة</h5>
                    </div>
                    <div class="card-body">
                        <div class="row" id="gridSquares"></div>
                    </div>
                </div>
            `;
            gridsContainer.appendChild(gridsWrapper);
            
            const gridSquares = document.getElementById('gridSquares');
            
            data.grids.forEach(grid => {
                const gridElement = document.createElement('div');
                gridElement.className = 'col-md-4 mb-3';
                gridElement.innerHTML = `
                    <div class="card h-100">
                        <img src="${grid.image_url}" class="card-img-top" alt="مربع ${grid.position.row}-${grid.position.col}"
                             onerror="this.onerror=null; this.src=''; this.alt='خطأ في تحميل الصورة'; this.classList.add('error');">
                        <div class="card-body">
                            <h6 class="card-title text-center">مربع (${grid.position.row}, ${grid.position.col})</h6>
                        </div>
                    </div>
                `;
                gridSquares.appendChild(gridElement);
            });

            // إعادة توجيه إلى صفحة النتائج
            window.location.href = data.result_url;

        } catch (error) {
            console.error('Error:', error);
            alert('حدث خطأ: ' + error.message);
            loadingArea.classList.add('d-none');
        }
    });
}); 