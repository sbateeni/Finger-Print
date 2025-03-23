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
}); 