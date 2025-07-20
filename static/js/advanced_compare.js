document.addEventListener('DOMContentLoaded', function() {
    // العناصر الأساسية
    const fingerprint1Container = document.getElementById('fingerprint1-container');
    const fingerprint2Container = document.getElementById('fingerprint2-container');
    const fingerprint1Input = document.getElementById('fingerprint1');
    const fingerprint2Input = document.getElementById('fingerprint2');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    const compareBtn = document.getElementById('compareBtn');
    const resultContainer = document.getElementById('resultContainer');
    
    // خيارات المطابقة
    const minutiaeCount = document.getElementById('minutiaeCount');
    const minutiaeCountValue = document.getElementById('minutiaeCountValue');
    const sensitivity = document.getElementById('sensitivity');
    const toggleAdvanced = document.getElementById('toggleAdvanced');
    const advancedOptions = document.getElementById('advancedOptions');
    
    // الخيارات المتقدمة
    const enhanceContrast = document.getElementById('enhanceContrast');
    const removeNoise = document.getElementById('removeNoise');
    const normalizeRidges = document.getElementById('normalizeRidges');
    const algorithms = document.getElementsByName('algorithm');
    const matchThreshold = document.getElementById('matchThreshold');
    const matchThresholdValue = document.getElementById('matchThresholdValue');
    const minMatchingPoints = document.getElementById('minMatchingPoints');
    const analyzeQuality = document.getElementById('analyzeQuality');
    const analyzeDistortion = document.getElementById('analyzeDistortion');
    const generateReport = document.getElementById('generateReport');
    
    // عناصر النتائج
    const totalScore = document.getElementById('totalScore');
    const minutiaeScore = document.getElementById('minutiaeScore');
    const orientationScore = document.getElementById('orientationScore');
    const densityScore = document.getElementById('densityScore');
    const qualityLevel = document.getElementById('qualityLevel');
    const fingerprintQuality = document.getElementById('fingerprintQuality');
    const distortionAnalysis = document.getElementById('distortionAnalysis');
    const issues = document.getElementById('issues');
    const recommendations = document.getElementById('recommendations');
    const matchingImage = document.getElementById('matchingImage');
    const generateReportBtn = document.getElementById('generateReportBtn');
    
    // معالجة رفع الملفات
    fingerprint1Container.addEventListener('click', () => fingerprint1Input.click());
    fingerprint2Container.addEventListener('click', () => fingerprint2Input.click());
    
    fingerprint1Input.addEventListener('change', handleFileSelect);
    fingerprint2Input.addEventListener('change', handleFileSelect);
    
    // تحديث القيم المعروضة
    minutiaeCount.addEventListener('input', () => {
        minutiaeCountValue.textContent = minutiaeCount.value;
    });
    
    matchThreshold.addEventListener('input', () => {
        matchThresholdValue.textContent = matchThreshold.value + '%';
    });
    
    // إظهار/إخفاء الخيارات المتقدمة
    toggleAdvanced.addEventListener('click', () => {
        const isVisible = advancedOptions.style.display === 'block';
        advancedOptions.style.display = isVisible ? 'none' : 'block';
        toggleAdvanced.innerHTML = isVisible ? 
            '<i class="bi bi-chevron-down"></i> إظهار' : 
            '<i class="bi bi-chevron-up"></i> إخفاء';
    });
    
    // معالجة اختيار الملفات
    function handleFileSelect(event) {
        const file = event.target.files[0];
        const preview = event.target.id === 'fingerprint1' ? preview1 : preview2;
        
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                preview.classList.remove('d-none');
                updateCompareButton();
            };
            reader.readAsDataURL(file);
        }
    }
    
    // تحديث حالة زر المقارنة
    function updateCompareButton() {
        compareBtn.disabled = !(fingerprint1Input.files[0] && fingerprint2Input.files[0]);
    }
    
    // معالجة المقارنة
    compareBtn.addEventListener('click', async () => {
        try {
            compareBtn.disabled = true;
            compareBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> جاري المقارنة...';
            
            const formData = new FormData();
            formData.append('fingerprint1', fingerprint1Input.files[0]);
            formData.append('fingerprint2', fingerprint2Input.files[0]);
            formData.append('minutiaeCount', minutiaeCount.value);
            formData.append('sensitivity', sensitivity.value);
            formData.append('enhanceContrast', enhanceContrast.checked);
            formData.append('removeNoise', removeNoise.checked);
            formData.append('normalizeRidges', normalizeRidges.checked);
            formData.append('algorithm', Array.from(algorithms).find(a => a.checked).value);
            formData.append('matchThreshold', matchThreshold.value);
            formData.append('minMatchingPoints', minMatchingPoints.value);
            formData.append('analyzeQuality', analyzeQuality.checked);
            formData.append('analyzeDistortion', analyzeDistortion.checked);
            formData.append('generateReport', generateReport.checked);
            
            const response = await fetch('/advanced_compare', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('فشلت عملية المقارنة');
            }
            
            const result = await response.json();
            displayResults(result);
            
        } catch (error) {
            alert('حدث خطأ أثناء المقارنة: ' + error.message);
        } finally {
            compareBtn.disabled = false;
            compareBtn.innerHTML = '<i class="bi bi-search"></i> بدء المقارنة';
        }
    });
    
    // عرض النتائج
    function displayResults(result) {
        resultContainer.style.display = 'block';
        
        // تحديث شريط التقدم والقيم
        updateProgressBar('totalScore', result.totalScore);
        updateProgressBar('minutiaeScore', result.minutiaeScore);
        updateProgressBar('orientationScore', result.orientationScore);
        updateProgressBar('densityScore', result.densityScore);
        
        // تحديث تحليل الجودة
        qualityLevel.textContent = result.qualityLevel;
        qualityLevel.className = 'badge ' + getQualityBadgeClass(result.qualityLevel);
        
        // تحديث جودة البصمة
        fingerprintQuality.innerHTML = result.fingerprintQuality.map(q => 
            `<div class="mb-2">
                <strong>${q.name}:</strong> ${q.value}
                <div class="progress">
                    <div class="progress-bar" role="progressbar" style="width: ${q.score}%"></div>
                </div>
            </div>`
        ).join('');
        
        // تحديث تحليل التشوهات
        distortionAnalysis.innerHTML = result.distortionAnalysis.map(d => 
            `<div class="mb-2">
                <strong>${d.type}:</strong> ${d.description}
                <div class="progress">
                    <div class="progress-bar bg-warning" role="progressbar" style="width: ${d.severity}%"></div>
                </div>
            </div>`
        ).join('');
        
        // تحديث المشاكل والتوصيات
        issues.innerHTML = result.issues.map(issue => 
            `<li class="text-danger mb-1"><i class="bi bi-exclamation-circle"></i> ${issue}</li>`
        ).join('');
        
        recommendations.innerHTML = result.recommendations.map(rec => 
            `<li class="text-success mb-1"><i class="bi bi-check-circle"></i> ${rec}</li>`
        ).join('');
        
        // تحديث صورة المطابقة
        if (result.matchingImage) {
            matchingImage.src = result.matchingImage;
        }
    }
    
    // تحديث شريط التقدم
    function updateProgressBar(elementId, value) {
        const element = document.getElementById(elementId);
        const progressBar = element.previousElementSibling.querySelector('.progress-bar');
        progressBar.style.width = value + '%';
        element.textContent = value + '%';
    }
    
    // الحصول على فئة شارة الجودة
    function getQualityBadgeClass(level) {
        switch (level.toLowerCase()) {
            case 'ممتاز':
                return 'bg-success';
            case 'جيد':
                return 'bg-primary';
            case 'متوسط':
                return 'bg-warning';
            case 'ضعيف':
                return 'bg-danger';
            default:
                return 'bg-secondary';
        }
    }
    
    // معالجة إنشاء التقرير
    generateReportBtn.addEventListener('click', async () => {
        try {
            generateReportBtn.disabled = true;
            generateReportBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> جاري إنشاء التقرير...';
            
            const response = await fetch('/generate_report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: 'advanced',
                    timestamp: new Date().toISOString()
                })
            });
            
            if (!response.ok) {
                throw new Error('فشل إنشاء التقرير');
            }
            
            const result = await response.json();
            window.location.href = `/report/${result.reportId}`;
            
        } catch (error) {
            alert('حدث خطأ أثناء إنشاء التقرير: ' + error.message);
        } finally {
            generateReportBtn.disabled = false;
            generateReportBtn.innerHTML = '<i class="bi bi-file-earmark-text"></i> إنشاء تقرير';
        }
    });
}); 