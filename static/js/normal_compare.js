document.addEventListener('DOMContentLoaded', function() {
    // العناصر
    const fingerprint1Container = document.getElementById('fingerprint1-container');
    const fingerprint2Container = document.getElementById('fingerprint2-container');
    const fingerprint1Input = document.getElementById('fingerprint1');
    const fingerprint2Input = document.getElementById('fingerprint2');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');
    const minutiaeCount = document.getElementById('minutiaeCount');
    const minutiaeCountValue = document.getElementById('minutiaeCountValue');
    const compareBtn = document.getElementById('compareBtn');
    const resultContainer = document.getElementById('resultContainer');
    const generateReportBtn = document.getElementById('generateReportBtn');

    // تحديث قيمة عدد نقاط التفاصيل
    minutiaeCount.addEventListener('input', function() {
        minutiaeCountValue.textContent = this.value;
    });

    // معالجة تحميل البصمة الأولى
    fingerprint1Container.addEventListener('click', function() {
        fingerprint1Input.click();
    });

    fingerprint1Input.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview1.src = e.target.result;
                preview1.classList.remove('d-none');
                updateCompareButton();
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });

    // معالجة تحميل البصمة الثانية
    fingerprint2Container.addEventListener('click', function() {
        fingerprint2Input.click();
    });

    fingerprint2Input.addEventListener('change', function(e) {
        if (e.target.files && e.target.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview2.src = e.target.result;
                preview2.classList.remove('d-none');
                updateCompareButton();
            };
            reader.readAsDataURL(e.target.files[0]);
        }
    });

    // تحديث حالة زر المقارنة
    function updateCompareButton() {
        compareBtn.disabled = !(preview1.src && preview2.src);
    }

    // معالجة المقارنة
    compareBtn.addEventListener('click', async function() {
        try {
            compareBtn.disabled = true;
            compareBtn.innerHTML = '<i class="bi bi-hourglass-split"></i> جاري المقارنة...';

            const formData = new FormData();
            formData.append('fingerprint1', fingerprint1Input.files[0]);
            formData.append('fingerprint2', fingerprint2Input.files[0]);
            formData.append('minutiaeCount', minutiaeCount.value);
            formData.append('matchingMode', 'false');
            formData.append('useGridMatching', 'false');
            formData.append('useGridCutMatching', 'false');

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