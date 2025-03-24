// متغيرات عامة
let gridSize1 = 3;
let gridSize2 = 3;
let gridImages1 = [];
let gridImages2 = [];
let timestamp1 = '';
let timestamp2 = '';

// تحديث حالة الأزرار
function updateButtons() {
    const splitBtn1 = document.getElementById('splitBtn1');
    const splitBtn2 = document.getElementById('splitBtn2');
    const compareBtn = document.getElementById('compareBtn');

    splitBtn1.disabled = !document.getElementById('fingerprint1').files.length;
    splitBtn2.disabled = !document.getElementById('fingerprint2').files.length;
    compareBtn.disabled = !(gridImages1.length > 0 && gridImages2.length > 0);
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
async function splitFingerprint(file, gridSize, gridContainerId, gridImagesArray, timestampVar) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('grid_size', gridSize);

    try {
        console.log('بدء إرسال الطلب...');
        console.log('حجم الشبكة:', gridSize);
        console.log('اسم الملف:', file.name);

        const response = await fetch('/cut_fingerprint', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'حدث خطأ أثناء معالجة الصورة');
        }

        const result = await response.json();
        console.log('تم استلام النتيجة:', result);
        
        if (result.error) {
            throw new Error(result.error);
        }

        const gridContainer = document.getElementById(gridContainerId);
        gridContainer.innerHTML = '';
        gridImagesArray.length = 0;
        gridImagesArray.push(...result.grids.map(grid => grid.image));
        window[timestampVar] = result.timestamp;
        
        result.grids.forEach((grid, index) => {
            const div = document.createElement('div');
            div.className = 'grid-item';
            div.innerHTML = `
                <img src="${grid.image}" alt="Grid ${index + 1}">
                <div class="match-score">المربع ${index + 1}</div>
            `;
            gridContainer.appendChild(div);
        });

        updateButtons();
        console.log('تم تقسيم البصمة بنجاح');
    } catch (error) {
        console.error('حدث خطأ:', error);
        alert('حدث خطأ: ' + error.message);
    }
}

// معالجة نتائج المقارنة
function handleComparisonResults(results) {
    const gridContainer1 = document.getElementById('grid1');
    const gridContainer2 = document.getElementById('grid2');
    const gridItems1 = gridContainer1.getElementsByClassName('grid-item');
    const gridItems2 = gridContainer2.getElementsByClassName('grid-item');
    
    // تحديث درجات التطابق لكل مربع
    results.forEach(result => {
        const gridItem1 = gridItems1[result.grid1 - 1];
        const gridItem2 = gridItems2[result.grid2 - 1];
        
        if (gridItem1 && gridItem2) {
            const scoreElement1 = gridItem1.querySelector('.match-score');
            const scoreElement2 = gridItem2.querySelector('.match-score');
            
            scoreElement1.textContent = `المربع ${result.grid1}: ${result.score.toFixed(1)}%`;
            scoreElement2.textContent = `المربع ${result.grid2}: ${result.score.toFixed(1)}%`;
            
            const scoreClass = result.score >= 70 ? 'high' : 
                             result.score >= 50 ? 'medium' : 'low';
            
            scoreElement1.className = 'match-score ' + scoreClass;
            scoreElement2.className = 'match-score ' + scoreClass;
        }
    });

    // حساب وإظهار النتائج الإجمالية
    const bestMatch = results.reduce((best, current) => 
        current.score > best.score ? current : best
    );

    document.getElementById('resultContainer').style.display = 'block';
    document.getElementById('overallScore').style.width = `${bestMatch.score}%`;
    document.getElementById('overallScore').textContent = `${bestMatch.score.toFixed(1)}%`;
    document.getElementById('avgScore').textContent = (results.reduce((sum, r) => sum + r.score, 0) / results.length).toFixed(1);
    document.getElementById('matchingSquares').textContent = results.filter(r => r.score >= 70).length;

    // تحديث ألوان شريط التقدم
    const progressBar = document.getElementById('overallScore');
    progressBar.className = 'progress-bar ' + 
        (bestMatch.score >= 70 ? 'bg-success' : 
         bestMatch.score >= 50 ? 'bg-warning' : 'bg-danger');
}

// مقارنة المربعات
async function compareGrids() {
    if (gridImages1.length === 0 || gridImages2.length === 0) return;

    try {
        const comparisonResults = [];
        
        // مقارنة كل مربع من البصمة الأولى مع كل مربع من البصمة الثانية
        for (let i = 0; i < gridImages1.length; i++) {
            for (let j = 0; j < gridImages2.length; j++) {
                const compareFormData = new FormData();
                compareFormData.append('grid1_index', i);
                compareFormData.append('grid2_index', j);
                compareFormData.append('grid_size', Math.max(gridSize1, gridSize2));

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

        // معالجة وعرض النتائج
        handleComparisonResults(comparisonResults);

    } catch (error) {
        alert('حدث خطأ: ' + error.message);
    }
}

// إضافة مستمعي الأحداث
document.addEventListener('DOMContentLoaded', function() {
    // تحديث حجم الشبكة
    document.getElementById('gridSize1').addEventListener('change', function() {
        gridSize1 = parseInt(this.value);
    });

    document.getElementById('gridSize2').addEventListener('change', function() {
        gridSize2 = parseInt(this.value);
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
    document.getElementById('splitBtn1').addEventListener('click', function() {
        const file = document.getElementById('fingerprint1').files[0];
        if (!file) return;
        splitFingerprint(file, gridSize1, 'grid1', gridImages1, 'timestamp1');
    });

    document.getElementById('splitBtn2').addEventListener('click', function() {
        const file = document.getElementById('fingerprint2').files[0];
        if (!file) return;
        splitFingerprint(file, gridSize2, 'grid2', gridImages2, 'timestamp2');
    });

    // مقارنة المربعات
    document.getElementById('compareBtn').addEventListener('click', compareGrids);
}); 