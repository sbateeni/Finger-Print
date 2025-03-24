document.addEventListener('DOMContentLoaded', function() {
    // العناصر
    const fullFingerprintContainer = document.getElementById('fullFingerprint-container');
    const partialFingerprintContainer = document.getElementById('partialFingerprint-container');
    const fullFingerprintInput = document.getElementById('fullFingerprint');
    const partialFingerprintInput = document.getElementById('partialFingerprint');
    const previewFull = document.getElementById('previewFull');
    const previewPartial = document.getElementById('previewPartial');
    const minutiaeCount = document.getElementById('minutiaeCount');
    const minutiaeCountValue = document.getElementById('minutiaeCountValue');
    const sensitivity = document.getElementById('sensitivity');
    const compareBtn = document.getElementById('compareBtn');
    const resultContainer = document.getElementById('resultContainer');
    const generateReportBtn = document.getElementById('generateReportBtn');
    const matchRegion = document.getElementById('matchRegion');

    // تحديث قيمة عدد نقاط التفاصيل
    minutiaeCount.addEventListener('input', function() {
        minutiaeCountValue.textContent = this.value;
    });

    // معالجة تحميل البصمة الكاملة
    fullFingerprintContainer.addEventListener('click', function() {
        fullFingerprintInput.click();
    });

    fullFingerprintInput.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewFull.src = e.target.result;
                previewFull.classList.remove('d-none');
                updateCompareButton();
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });

    // معالجة تحميل البصمة الجزئية
    partialFingerprintContainer.addEventListener('click', function() {
        partialFingerprintInput.click();
    });

    partialFingerprintInput.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewPartial.src = e.target.result;
                previewPartial.classList.remove('d-none');
                updateCompareButton();
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });

    // تحديث حالة زر المقارنة
    function updateCompareButton() {
        compareBtn.disabled = !(previewFull.src && previewPartial.src);
    }

    // معالجة المقارنة
    compareBtn.addEventListener('click', async function() {
        try {
            compareBtn.disabled = true;
            compareBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> جاري المقارنة...';

            const formData = new FormData();
            formData.append('fingerprint1', fullFingerprintInput.files[0]);
            formData.append('fingerprint2', partialFingerprintInput.files[0]);
            formData.append('minutiaeCount', minutiaeCount.value);
            formData.append('matchingMode', 'true');
            formData.append('useGridMatching', 'false');
            formData.append('useGridCutMatching', 'false');
            formData.append('sensitivity', sensitivity.value);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('فشل في معالجة البصمات');
            }

            const data = await response.json();
            displayResults(data);
            resultContainer.style.display = 'block';

        } catch (error) {
            alert('حدث خطأ أثناء المقارنة: ' + error.message);
        } finally {
            compareBtn.disabled = false;
            compareBtn.innerHTML = '<i class="bi bi-search"></i> بدء المقارنة';
        }
    });

    // عرض النتائج
    function displayResults(data) {
        // تحديث شريط التقدم والقيم
        updateProgressBar('totalScore', data.score.total);
        updateProgressBar('minutiaeScore', data.score.minutiae);
        updateProgressBar('orientationScore', data.score.orientation);
        updateProgressBar('densityScore', data.score.density);

        // تحديث تحليل الجودة
        document.getElementById('qualityLevel').textContent = data.quality.level;
        document.getElementById('qualityLevel').className = `badge bg-${getQualityColor(data.quality.level)}`;

        // تحديث موقع المطابقة
        if (data.partial_match) {
            document.getElementById('matchLocation').textContent = 
                `المنطقة: ${data.partial_match.region}`;
            
            // تحديث منطقة المطابقة في الصورة
            if (data.partial_match.location) {
                const location = data.partial_match.location;
                const matchingImage = document.getElementById('matchingImage');
                
                // حساب النسب المئوية للموقع
                const left = (location.x / matchingImage.naturalWidth) * 100;
                const top = (location.y / matchingImage.naturalHeight) * 100;
                const width = (location.width / matchingImage.naturalWidth) * 100;
                const height = (location.height / matchingImage.naturalHeight) * 100;
                
                matchRegion.style.left = `${left}%`;
                matchRegion.style.top = `${top}%`;
                matchRegion.style.width = `${width}%`;
                matchRegion.style.height = `${height}%`;
                matchRegion.classList.remove('d-none');
            }
        }

        // تحديث المشاكل والتوصيات
        const issuesList = document.getElementById('issues');
        const recommendationsList = document.getElementById('recommendations');
        
        issuesList.innerHTML = data.quality.issues.map(issue => `<li>${issue}</li>`).join('');
        recommendationsList.innerHTML = data.quality.recommendations.map(rec => `<li>${rec}</li>`).join('');

        // تحديث صورة المطابقة
        document.getElementById('matchingImage').src = data.matching_image;
    }

    // تحديث شريط التقدم
    function updateProgressBar(id, value) {
        const progressBar = document.querySelector(`#${id}`).previousElementSibling.querySelector('.progress-bar');
        const valueDisplay = document.getElementById(id);
        
        progressBar.style.width = `${value}%`;
        valueDisplay.textContent = `${value.toFixed(1)}%`;
    }

    // تحديد لون مستوى الجودة
    function getQualityColor(level) {
        switch (level.toLowerCase()) {
            case 'جيد':
                return 'success';
            case 'متوسط':
                return 'warning';
            case 'ضعيف':
                return 'danger';
            default:
                return 'secondary';
        }
    }

    // معالجة إنشاء التقرير
    generateReportBtn.addEventListener('click', function() {
        // سيتم إضافة وظيفة إنشاء التقرير لاحقاً
        alert('سيتم إضافة وظيفة إنشاء التقرير قريباً');
    });
}); 