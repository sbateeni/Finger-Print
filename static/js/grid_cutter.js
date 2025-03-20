document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fullFingerprintInput = document.getElementById('fullFingerprint');
    const referenceFingerprintInput = document.getElementById('referenceFingerprint');
    const fullFingerprintPreview = document.getElementById('fullFingerprintPreview');
    const referenceFingerprintPreview = document.getElementById('referenceFingerprintPreview');
    const loadingArea = document.getElementById('loadingArea');
    const resultsArea = document.getElementById('resultsArea');
    const gridsContainer = document.getElementById('gridsContainer');

    // عرض معاينة الصور عند اختيارها
    function showImagePreview(input, previewElement) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.innerHTML = `<img src="${e.target.result}" class="img-fluid" alt="معاينة البصمة">`;
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    fullFingerprintInput.addEventListener('change', function() {
        showImagePreview(this, fullFingerprintPreview);
    });

    referenceFingerprintInput.addEventListener('change', function() {
        showImagePreview(this, referenceFingerprintPreview);
    });

    // معالجة تقديم النموذج
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!fullFingerprintInput.files[0] || !referenceFingerprintInput.files[0]) {
            alert('الرجاء اختيار كلا الصورتين');
            return;
        }

        // إظهار منطقة التحميل
        loadingArea.classList.remove('d-none');
        resultsArea.classList.add('d-none');
        gridsContainer.innerHTML = '';

        const formData = new FormData();
        formData.append('fullFingerprint', fullFingerprintInput.files[0]);
        formData.append('referenceFingerprint', referenceFingerprintInput.files[0]);

        try {
            const response = await fetch('/process_grids', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('حدث خطأ أثناء معالجة الصور');
            }

            const data = await response.json();
            
            // عرض الصورة التوضيحية للشبكة
            const gridVisualization = document.createElement('div');
            gridVisualization.className = 'col-12 mb-4';
            gridVisualization.innerHTML = `
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">تقسيم البصمة إلى مربعات</h5>
                    </div>
                    <div class="card-body">
                        <img src="${data.visualization}" class="img-fluid" alt="تقسيم البصمة">
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
            
            // ترتيب المربعات حسب الصفوف والأعمدة
            const sortedGrids = data.grids.sort((a, b) => {
                if (a.position.row !== b.position.row) {
                    return a.position.row - b.position.row;
                }
                return a.position.col - b.position.col;
            });
            
            sortedGrids.forEach(grid => {
                const gridElement = document.createElement('div');
                gridElement.className = 'col-md-4 mb-3';
                gridElement.innerHTML = `
                    <div class="card h-100">
                        <img src="${grid.image_url}" class="card-img-top" alt="مربع ${grid.position.row}-${grid.position.col}">
                        <div class="card-body">
                            <h6 class="card-title text-center">مربع (${grid.position.row}, ${grid.position.col})</h6>
                            <p class="card-text text-center">
                                <small class="text-muted">
                                    المسافة بين الخطوط: ${grid.ridge_distance.toFixed(2)}
                                </small>
                            </p>
                        </div>
                    </div>
                `;
                gridSquares.appendChild(gridElement);
            });

            // إظهار النتائج
            loadingArea.classList.add('d-none');
            resultsArea.classList.remove('d-none');

        } catch (error) {
            alert('حدث خطأ: ' + error.message);
            loadingArea.classList.add('d-none');
        }
    });
}); 