// متغيرات عامة
let selectedGrid1 = null;
let selectedGrid2 = null;
let gridSize = 3;

// تحديث حالة الأزرار
function updateButtons() {
    const splitBtn1 = document.getElementById('splitBtn');
    const splitBtn2 = document.getElementById('splitBtn2');
    const compareBtn = document.getElementById('compareBtn');
    const multipleBtn = document.getElementById('multipleBtn');
    const crossCompareBtn = document.getElementById('crossCompareBtn');

    splitBtn1.disabled = !document.getElementById('fingerprint1').files.length;
    splitBtn2.disabled = !document.getElementById('fingerprint2').files.length;
    compareBtn.disabled = !(selectedGrid1 !== null && selectedGrid2 !== null);
    multipleBtn.disabled = !document.getElementById('fingerprint1').files.length;
    crossCompareBtn.disabled = !(document.getElementById('fingerprint1').files.length && 
                               document.getElementById('fingerprint2').files.length);
}

// معاينة الصورة
function previewImage(input, preview) {
    if (input.files && input.files[0]) {
        const reader = new FileReader();
        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.classList.remove('d-none');
        }
        reader.readAsDataURL(input.files[0]);
    }
}

// تقسيم البصمة إلى شبكة
async function splitFingerprint(file, gridSize, gridContainer) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('grid_size', gridSize);

    try {
        const response = await fetch('/split_fingerprint', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        gridContainer.innerHTML = '';
        
        result.grid_images.forEach((img, index) => {
            const div = document.createElement('div');
            div.className = 'grid-item';
            div.innerHTML = `
                <img src="${img}" alt="Grid ${index + 1}">
                <div class="match-score">المربع ${index + 1}</div>
            `;
            div.addEventListener('click', () => {
                document.querySelectorAll(`#${gridContainer.id} .grid-item`).forEach(item => 
                    item.classList.remove('selected'));
                div.classList.add('selected');
                if (gridContainer.id === 'grid1') {
                    selectedGrid1 = index;
                } else {
                    selectedGrid2 = index;
                }
                updateButtons();
            });
            gridContainer.appendChild(div);
        });
    } catch (error) {
        alert('حدث خطأ: ' + error.message);
    }
}

// مقارنة المربعات المحددة
async function compareGrids() {
    if (selectedGrid1 === null || selectedGrid2 === null) return;

    const formData = new FormData();
    formData.append('grid1_index', selectedGrid1);
    formData.append('grid2_index', selectedGrid2);
    formData.append('grid_size', gridSize);

    try {
        const response = await fetch('/compare_grids', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // عرض النتائج
        document.getElementById('resultContainer').style.display = 'block';
        document.getElementById('overallScore').style.width = `${result.match_score}%`;
        document.getElementById('overallScore').textContent = `${result.match_score}%`;
        document.getElementById('avgScore').textContent = result.avg_score;
        document.getElementById('matchingSquares').textContent = result.matching_squares;
        document.getElementById('quality1').textContent = result.quality1;
        document.getElementById('quality2').textContent = result.quality2;

        // تحديث ألوان شريط التقدم
        const progressBar = document.getElementById('overallScore');
        progressBar.className = 'progress-bar ' + 
            (result.match_score >= 70 ? 'bg-success' : 
             result.match_score >= 50 ? 'bg-warning' : 'bg-danger');
    } catch (error) {
        alert('حدث خطأ: ' + error.message);
    }
}

// تحليل بصمات متعددة
async function analyzeMultipleFingerprints() {
    const file = document.getElementById('fingerprint1').files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('grid_size', gridSize);

    try {
        const response = await fetch('/split_fingerprint', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // إنشاء مصفوفة لتخزين نتائج المقارنة
        const comparisonResults = [];
        
        // مقارنة كل مربع مع باقي المربعات
        for (let i = 0; i < result.grid_images.length; i++) {
            for (let j = i + 1; j < result.grid_images.length; j++) {
                const compareFormData = new FormData();
                compareFormData.append('grid1_index', i);
                compareFormData.append('grid2_index', j);
                compareFormData.append('grid_size', gridSize);

                const compareResponse = await fetch('/compare_grids', {
                    method: 'POST',
                    body: compareFormData
                });
                const compareResult = await compareResponse.json();
                
                if (!compareResult.error) {
                    comparisonResults.push({
                        grid1: i + 1,
                        grid2: j + 1,
                        score: compareResult.match_score
                    });
                }
            }
        }

        // عرض النتائج
        const gridResults = document.getElementById('gridResults');
        gridResults.innerHTML = '';
        
        comparisonResults.forEach(result => {
            const div = document.createElement('div');
            div.className = 'grid-item';
            div.innerHTML = `
                <div class="match-score ${result.score >= 70 ? 'high' : result.score >= 50 ? 'medium' : 'low'}">
                    المربع ${result.grid1} مع المربع ${result.grid2}: ${result.score.toFixed(1)}%
                </div>
            `;
            gridResults.appendChild(div);
        });

        // عرض إحصائيات عامة
        const avgScore = comparisonResults.reduce((sum, r) => sum + r.score, 0) / comparisonResults.length;
        document.getElementById('avgScore').textContent = avgScore.toFixed(1);
        document.getElementById('overallScore').style.width = `${avgScore}%`;
        document.getElementById('overallScore').textContent = `${avgScore.toFixed(1)}%`;
        document.getElementById('overallScore').className = 'progress-bar ' + 
            (avgScore >= 70 ? 'bg-success' : avgScore >= 50 ? 'bg-warning' : 'bg-danger');

    } catch (error) {
        alert('حدث خطأ: ' + error.message);
    }
}

// مقارنة البصمات المتعددة بين الصورتين
async function crossCompareFingerprints() {
    const file1 = document.getElementById('fingerprint1').files[0];
    const file2 = document.getElementById('fingerprint2').files[0];
    if (!file1 || !file2) return;

    try {
        // تقسيم البصمة الأولى
        const formData1 = new FormData();
        formData1.append('file', file1);
        formData1.append('grid_size', gridSize);
        const response1 = await fetch('/split_fingerprint', {
            method: 'POST',
            body: formData1
        });
        const result1 = await response1.json();

        // تقسيم البصمة الثانية
        const formData2 = new FormData();
        formData2.append('file', file2);
        formData2.append('grid_size', gridSize);
        const response2 = await fetch('/split_fingerprint', {
            method: 'POST',
            body: formData2
        });
        const result2 = await response2.json();

        if (result1.error || result2.error) {
            throw new Error(result1.error || result2.error);
        }

        // إنشاء مصفوفة لتخزين نتائج المقارنة
        const comparisonResults = [];
        
        // مقارنة كل مربع من البصمة الأولى مع كل مربع من البصمة الثانية
        for (let i = 0; i < result1.grid_images.length; i++) {
            for (let j = 0; j < result2.grid_images.length; j++) {
                const compareFormData = new FormData();
                compareFormData.append('grid1_index', i);
                compareFormData.append('grid2_index', j);
                compareFormData.append('grid_size', gridSize);

                const compareResponse = await fetch('/compare_grids', {
                    method: 'POST',
                    body: compareFormData
                });
                const compareResult = await compareResponse.json();
                
                if (!compareResult.error) {
                    comparisonResults.push({
                        grid1: i + 1,
                        grid2: j + 1,
                        score: compareResult.match_score
                    });
                }
            }
        }

        // عرض النتائج
        const gridResults = document.getElementById('gridResults');
        gridResults.innerHTML = '';
        
        comparisonResults.forEach(result => {
            const div = document.createElement('div');
            div.className = 'grid-item';
            div.innerHTML = `
                <div class="match-score ${result.score >= 70 ? 'high' : result.score >= 50 ? 'medium' : 'low'}">
                    المربع ${result.grid1} من البصمة الأولى مع المربع ${result.grid2} من البصمة الثانية: ${result.score.toFixed(1)}%
                </div>
            `;
            gridResults.appendChild(div);
        });

        // عرض إحصائيات عامة
        const avgScore = comparisonResults.reduce((sum, r) => sum + r.score, 0) / comparisonResults.length;
        document.getElementById('avgScore').textContent = avgScore.toFixed(1);
        document.getElementById('overallScore').style.width = `${avgScore}%`;
        document.getElementById('overallScore').textContent = `${avgScore.toFixed(1)}%`;
        document.getElementById('overallScore').className = 'progress-bar ' + 
            (avgScore >= 70 ? 'bg-success' : avgScore >= 50 ? 'bg-warning' : 'bg-danger');

    } catch (error) {
        alert('حدث خطأ: ' + error.message);
    }
}

// إضافة مستمعي الأحداث
document.addEventListener('DOMContentLoaded', function() {
    // تحديث حجم الشبكة
    document.getElementById('gridSize').addEventListener('change', function() {
        gridSize = parseInt(this.value);
    });

    // معاينة الصور
    document.getElementById('fingerprint1').addEventListener('change', function() {
        previewImage(this, document.getElementById('preview1'));
        updateButtons();
    });

    document.getElementById('fingerprint2').addEventListener('change', function() {
        previewImage(this, document.getElementById('preview2'));
        updateButtons();
    });

    // تقسيم البصمات
    document.getElementById('splitBtn').addEventListener('click', function() {
        const file = document.getElementById('fingerprint1').files[0];
        if (!file) return;
        splitFingerprint(file, gridSize, document.getElementById('grid1'));
    });

    document.getElementById('splitBtn2').addEventListener('click', function() {
        const file = document.getElementById('fingerprint2').files[0];
        if (!file) return;
        splitFingerprint(file, gridSize, document.getElementById('grid2'));
    });

    // مقارنة البصمات
    document.getElementById('compareBtn').addEventListener('click', compareGrids);

    // إضافة مستمعي الأحداث للأزرار الجديدة
    document.getElementById('multipleBtn').addEventListener('click', analyzeMultipleFingerprints);
    document.getElementById('crossCompareBtn').addEventListener('click', crossCompareFingerprints);
}); 