// تحديث عرض عدد النقاط المميزة
document.getElementById('minutiaeCount').addEventListener('input', function() {
    document.getElementById('minutiaeCountValue').textContent = this.value;
});

// معاينة الصور
document.getElementById('fingerprint1').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview1').src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

document.getElementById('fingerprint2').addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('preview2').src = e.target.result;
        }
        reader.readAsDataURL(file);
    }
});

// معالجة تقديم النموذج
document.getElementById('compareForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    // إظهار مؤشر التحميل
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').style.display = 'none';
    
    const formData = new FormData(this);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        
        // تحديث الصور الناتجة
        document.getElementById('result1').src = data.processed_images.img1;
        document.getElementById('result2').src = data.processed_images.img2;
        document.getElementById('matchedImage').src = data.matching_image;
        
        // تحديث النتائج
        const matchScore = data.score || 0;
        const qualityScore = data.quality_score || 0;
        
        document.getElementById('matchScore').textContent = `${matchScore.toFixed(2)}%`;
        document.getElementById('qualityScore').textContent = `${qualityScore.toFixed(2)}%`;
        
        // تحديث شريط التقدم
        document.getElementById('matchProgress').style.width = `${matchScore}%`;
        document.getElementById('qualityProgress').style.width = `${qualityScore}%`;
        
        // إظهار النتائج
        document.getElementById('loading').style.display = 'none';
        document.getElementById('results').style.display = 'block';
    })
    .catch(error => {
        console.error('Error:', error);
        alert('حدث خطأ أثناء المقارنة: ' + error.message);
        document.getElementById('loading').style.display = 'none';
    });
}); 