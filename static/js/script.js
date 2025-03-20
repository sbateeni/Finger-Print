document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');
    
    const uploadForm = document.getElementById('uploadForm');
    const fingerprint1Input = document.getElementById('fingerprint1');
    const fingerprint2Input = document.getElementById('fingerprint2');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    const submitButton = document.getElementById('submitButton');
    const matchingButtons = document.getElementById('matchingButtons');
    const normalizedGridsButton = document.getElementById('normalizedGridsButton');
    const gridCutMatchingButton = document.getElementById('gridCutMatchingButton');
    const loadingArea = document.getElementById('loadingArea');
    const loadingText = document.getElementById('loadingText');
    const gridCutArea = document.getElementById('gridCutArea');
    const gridSquares = document.getElementById('gridSquares');
    const resultsArea = document.getElementById('resultsArea');
    const minutiaeCount = document.getElementById('minutiaeCount');
    const minutiaeCountValue = document.getElementById('minutiaeCountValue');
    const fingerprint1Area = document.getElementById('fingerprint1Area');
    const fingerprint2Area = document.getElementById('fingerprint2Area');

    // التحقق من تهيئة العناصر
    if (!uploadForm) console.error('uploadForm not found');
    if (!fingerprint1Input) console.error('fingerprint1Input not found');
    if (!fingerprint2Input) console.error('fingerprint2Input not found');
    if (!submitButton) console.error('submitButton not found');
    if (!matchingButtons) console.error('matchingButtons not found');
    if (!normalizedGridsButton) console.error('normalizedGridsButton not found');
    if (!gridCutMatchingButton) console.error('gridCutMatchingButton not found');

    console.log('Elements initialized:', {
        uploadForm: !!uploadForm,
        fingerprint1Input: !!fingerprint1Input,
        fingerprint2Input: !!fingerprint2Input,
        submitButton: !!submitButton,
        matchingButtons: !!matchingButtons,
        normalizedGridsButton: !!normalizedGridsButton,
        gridCutMatchingButton: !!gridCutMatchingButton
    });

    // إضافة مستمعي الأحداث لمناطق الرفع
    if (fingerprint1Area) {
        fingerprint1Area.addEventListener('click', () => fingerprint1Input.click());
        fingerprint1Area.addEventListener('dragover', (e) => {
            e.preventDefault();
            fingerprint1Area.classList.add('dragover');
        });
        fingerprint1Area.addEventListener('dragleave', () => {
            fingerprint1Area.classList.remove('dragover');
        });
        fingerprint1Area.addEventListener('drop', (e) => {
            e.preventDefault();
            fingerprint1Area.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fingerprint1Input.files = e.dataTransfer.files;
                showImagePreview(fingerprint1Input, preview1);
            }
        });
    }

    if (fingerprint2Area) {
        fingerprint2Area.addEventListener('click', () => fingerprint2Input.click());
        fingerprint2Area.addEventListener('dragover', (e) => {
            e.preventDefault();
            fingerprint2Area.classList.add('dragover');
        });
        fingerprint2Area.addEventListener('dragleave', () => {
            fingerprint2Area.classList.remove('dragover');
        });
        fingerprint2Area.addEventListener('drop', (e) => {
            e.preventDefault();
            fingerprint2Area.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                fingerprint2Input.files = e.dataTransfer.files;
                showImagePreview(fingerprint2Input, preview2);
            }
        });
    }

    // مستمعو أحداث تغيير الملفات
    fingerprint1Input.addEventListener('change', function() {
        showImagePreview(this, preview1);
    });

    fingerprint2Input.addEventListener('change', function() {
        showImagePreview(this, preview2);
    });

    // تحديث عرض عدد النقاط المميزة
    minutiaeCount.addEventListener('input', function() {
        minutiaeCountValue.textContent = this.value + ' نقطة';
    });

    // عرض معاينة الصور عند اختيارها
    function showImagePreview(input, previewElement) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewElement.innerHTML = `
                    <div class="preview-container">
                        <img src="${e.target.result}" class="img-fluid rounded" alt="معاينة البصمة">
                        <div class="success-message mt-2">
                            <i class="bi bi-check-circle-fill text-success"></i>
                            تم اختيار الصورة بنجاح
                        </div>
                    </div>
                `;
                previewElement.classList.add('has-image');
                validateForm();
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    // التحقق من النموذج
    function validateForm() {
        const isValid = fingerprint1Input.files.length > 0 && fingerprint2Input.files.length > 0;
        submitButton.disabled = !isValid;
        return isValid;
    }

    // إعداد مناطق الرفع
    function setupUploadArea(area, input, preview) {
        if (!area) return;

        // منع السحب والإفلات الافتراضي
        area.addEventListener('dragover', (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        });

        area.addEventListener('dragleave', () => {
            area.classList.remove('dragover');
        });

        area.addEventListener('drop', (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                input.files = e.dataTransfer.files;
                handleFileSelect(input, preview);
            }
        });

        // معالجة النقر على منطقة الرفع
        area.addEventListener('click', () => {
            input.click();
        });

        // معالجة اختيار الملف
        input.addEventListener('change', () => {
            handleFileSelect(input, preview);
        });
    }

    // معالجة اختيار الملف
    function handleFileSelect(input, preview) {
        if (input.files && input.files[0]) {
            const file = input.files[0];
            
            // التحقق من نوع الملف
            if (!file.type.startsWith('image/')) {
                showError(preview, 'يرجى اختيار ملف صورة صالح');
                input.value = '';
                return;
            }

            // التحقق من حجم الملف (الحد الأقصى 5 ميجابايت)
            if (file.size > 5 * 1024 * 1024) {
                showError(preview, 'حجم الملف كبير جداً. الحد الأقصى هو 5 ميجابايت');
                input.value = '';
                return;
            }

            // عرض الصورة المختارة
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.innerHTML = `
                    <div class="preview-container">
                        <img src="${e.target.result}" class="img-fluid rounded" alt="معاينة البصمة">
                        <div class="success-message mt-2">
                            <i class="bi bi-check-circle-fill text-success"></i>
                            تم اختيار الصورة بنجاح
                        </div>
                    </div>
                `;
                preview.classList.add('has-image');
            };
            
            reader.onerror = function() {
                showError(preview, 'حدث خطأ أثناء قراءة الملف');
                input.value = '';
            };

            reader.readAsDataURL(file);
        }
        validateForm();
    }

    // عرض رسالة خطأ
    function showError(preview, message) {
        preview.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-circle-fill"></i>
                ${message}
            </div>
        `;
        preview.classList.remove('has-image');
    }

    // معالجة تقديم النموذج
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        if (!validateForm()) {
            alert('الرجاء اختيار البصمتين قبل المتابعة');
            return;
        }

        try {
            // إظهار منطقة التحميل
            document.getElementById('loadingArea').classList.remove('d-none');
            document.getElementById('loadingText').textContent = 'جاري معالجة البصمات...';
            submitButton.disabled = true;

            const formData = new FormData();
            formData.append('fingerprint1', fingerprint1Input.files[0]);
            formData.append('fingerprint2', fingerprint2Input.files[0]);
            formData.append('minutiaeCount', document.getElementById('minutiaeCount').value);

            const response = await fetch('/cut_fingerprint', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`حدث خطأ أثناء معالجة الصور (${response.status})`);
            }

            const data = await response.json();
            
            // معالجة النتائج
            handleResults(data);

        } catch (error) {
            console.error('Error:', error);
            alert(error.message);
        } finally {
            document.getElementById('loadingArea').classList.add('d-none');
            submitButton.disabled = false;
        }
    });

    // معالجة المطابقة باستخدام المربعات المعدلة
    normalizedGridsButton.addEventListener('click', async function() {
        console.log('Normalized grids button clicked');
        
        try {
            loadingArea.classList.remove('d-none');
            loadingText.textContent = 'جاري المطابقة باستخدام المربعات المعدلة...';
            this.disabled = true;
            gridCutMatchingButton.disabled = true;

            const formData = new FormData();
            formData.append('fingerprint1', fingerprint1Input.files[0]);
            formData.append('fingerprint2', fingerprint2Input.files[0]);
            formData.append('minutiaeCount', minutiaeCount.value);
            formData.append('matchingMode', 'normalized');

            console.log('Sending request to /match_fingerprints');
            const response = await fetch('/match_fingerprints', {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText);
                throw new Error(`حدث خطأ أثناء المطابقة (${response.status}): ${errorText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);
            displayMatchingResults(data);

        } catch (error) {
            console.error('Error during normalized matching:', error);
            alert('حدث خطأ: ' + error.message);
        } finally {
            loadingArea.classList.add('d-none');
            this.disabled = false;
            gridCutMatchingButton.disabled = false;
        }
    });

    // معالجة المطابقة مع المربعات المقطعة
    gridCutMatchingButton.addEventListener('click', async function() {
        console.log('Grid cut matching button clicked');
        
        try {
            loadingArea.classList.remove('d-none');
            loadingText.textContent = 'جاري المطابقة مع المربعات المقطعة...';
            this.disabled = true;
            normalizedGridsButton.disabled = true;

            const formData = new FormData();
            formData.append('fingerprint1', fingerprint1Input.files[0]);
            formData.append('fingerprint2', fingerprint2Input.files[0]);
            formData.append('minutiaeCount', minutiaeCount.value);
            formData.append('matchingMode', 'grid_cut');

            console.log('Sending request to /match_fingerprints');
            const response = await fetch('/match_fingerprints', {
                method: 'POST',
                body: formData
            });

            console.log('Response status:', response.status);
            if (!response.ok) {
                const errorText = await response.text();
                console.error('Server error:', errorText);
                throw new Error(`حدث خطأ أثناء المطابقة (${response.status}): ${errorText}`);
            }

            const data = await response.json();
            console.log('Response data:', data);
            displayMatchingResults(data);

        } catch (error) {
            console.error('Error during grid cut matching:', error);
            alert('حدث خطأ: ' + error.message);
        } finally {
            loadingArea.classList.add('d-none');
            this.disabled = false;
            normalizedGridsButton.disabled = false;
        }
    });

    // عرض نتائج المطابقة
    function displayMatchingResults(data) {
        console.log('Displaying matching results:', data);
        
        const resultsArea = document.getElementById('resultsArea');
        const matchVisualization = document.getElementById('matchVisualization');
        const matchDetails = document.getElementById('matchDetails');
        
        if (!data || typeof data !== 'object') {
            console.error('Invalid data received:', data);
            showError('حدث خطأ أثناء معالجة النتائج');
            return;
        }
        
        try {
            // إخفاء منطقة التحميل وإظهار منطقة النتائج
            document.getElementById('loadingArea').classList.add('d-none');
            resultsArea.classList.remove('d-none');
            
            // عرض صورة المطابقة
            if (data.visualization) {
                matchVisualization.innerHTML = `
                    <img src="${data.visualization}" class="img-fluid" alt="نتيجة المطابقة">
                `;
            } else {
                matchVisualization.innerHTML = `
                    <div class="alert alert-warning">
                        لم يتم العثور على تطابق مناسب
                    </div>
                `;
            }
            
            // عرض تفاصيل المطابقة
            const score = data.score || 0;
            const scoreDetails = data.score_details || {};
            const qualityAnalysis = data.quality_analysis || {};
            
            matchDetails.innerHTML = `
                <div class="card">
                    <div class="card-header bg-${score >= 70 ? 'success' : score >= 50 ? 'warning' : 'danger'} text-white">
                        <h5 class="mb-0">نتيجة المطابقة: ${score.toFixed(2)}%</h5>
                    </div>
                    <div class="card-body">
                        <h6>تفاصيل النتيجة:</h6>
                        <ul class="list-group mb-3">
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                تطابق النقاط المميزة
                                <span class="badge bg-primary rounded-pill">${(scoreDetails.minutiae_score || 0).toFixed(2)}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                تطابق الاتجاهات
                                <span class="badge bg-primary rounded-pill">${(scoreDetails.orientation_score || 0).toFixed(2)}%</span>
                            </li>
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                تطابق كثافة الخطوط
                                <span class="badge bg-primary rounded-pill">${(scoreDetails.density_score || 0).toFixed(2)}%</span>
                            </li>
                        </ul>
                        
                        <h6>تحليل جودة المطابقة:</h6>
                        <div class="alert alert-${score >= 70 ? 'success' : score >= 50 ? 'warning' : 'danger'}">
                            <strong>مستوى الجودة:</strong> ${qualityAnalysis.quality_level || 'غير متوفر'}
                        </div>
                        
                        ${qualityAnalysis.issues && qualityAnalysis.issues.length > 0 ? `
                            <h6>المشكلات المكتشفة:</h6>
                            <ul class="list-group mb-3">
                                ${qualityAnalysis.issues.map(issue => `
                                    <li class="list-group-item">
                                        <i class="bi bi-exclamation-triangle text-warning"></i>
                                        ${issue}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : ''}
                        
                        ${qualityAnalysis.recommendations && qualityAnalysis.recommendations.length > 0 ? `
                            <h6>التوصيات:</h6>
                            <ul class="list-group">
                                ${qualityAnalysis.recommendations.map(rec => `
                                    <li class="list-group-item">
                                        <i class="bi bi-lightbulb text-primary"></i>
                                        ${rec}
                                    </li>
                                `).join('')}
                            </ul>
                        ` : ''}
                    </div>
                </div>
            `;
            
            // إذا كان هناك مربع أفضل تطابق، عرضه
            if (data.best_match && data.best_match.grid_image) {
                matchVisualization.innerHTML += `
                    <div class="mt-3">
                        <h6>المربع الأفضل تطابقاً:</h6>
                        <img src="${data.best_match.grid_image}" class="img-fluid" alt="المربع الأفضل تطابقاً">
                        <p class="text-muted mt-2">
                            الموقع: (${data.best_match.position[0]}, ${data.best_match.position[1]})
                            <br>
                            درجة التطابق: ${data.best_match.score.toFixed(2)}%
                        </p>
                    </div>
                `;
            }
            
        } catch (error) {
            console.error('Error displaying results:', error);
            showError('حدث خطأ أثناء عرض النتائج');
        }
    }

    function showError(message) {
        const resultsArea = document.getElementById('resultsArea');
        document.getElementById('loadingArea').classList.add('d-none');
        resultsArea.classList.remove('d-none');
        resultsArea.innerHTML = `
            <div class="alert alert-danger">
                <i class="bi bi-exclamation-circle"></i>
                ${message}
            </div>
        `;
    }
}); 