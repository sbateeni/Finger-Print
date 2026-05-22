import numpy as np
from typing import List, Dict, Any, Tuple

# MCC Constants (Optimized for 500 DPI camera images)
NS = 8          # Number of cells along one dimension (Grid size)
ND = 6          # Number of orientation layers
R = 70.0        # Radius of the cylinder (pixels)
SIGMA_S = 7.0   # Spatial smoothing parameter
SIGMA_D = (2.0 * np.pi) / ND / 2.0  # Directional smoothing parameter

def get_cylinder(center: Dict[str, Any], minutiae: List[Dict[str, Any]]) -> np.ndarray:
    """
    بناء أسطوانة (Cylinder) لنقطة دقيقة واحدة تصف جيرانها.
    Cylinder-Code representation for a single minutia.
    """
    cylinder = np.zeros((NS, NS, ND), dtype=np.float32)
    cx, cy = center['x'], center['y']
    # استخدام 'angle' من مستخرج النقاط (بالدرجات) وتحويله لراديان
    co = np.radians(center.get('angle', 0))
    
    # تحويل الإحداثيات لتكون نسبية للنقطة المركزية واتجاهها
    cos_o = np.cos(co)
    sin_o = np.sin(co)
    
    for m in minutiae:
        if m is center: continue
        
        # المسافة والاتجاه النسبي
        dx = m['x'] - cx
        dy = m['y'] - cy
        
        # التدوير لتطبيع الاتجاه (Relative Rotation)
        rx = dx * cos_o + dy * sin_o
        ry = -dx * sin_o + dy * cos_o
        
        dist_sq = rx*rx + ry*ry
        if dist_sq > R*R: continue
        
        # الزاوية النسبية (تحويل زاوية النقطة المجاورة لراديان أيضاً)
        mo = np.radians(m.get('angle', 0))
        d_angle = (mo - co + np.pi) % (2 * np.pi) - np.pi
        
        # توزيع المساهمة على خلايا الأسطوانة (Spatial & Directional Smoothing)
        # هذا الجزء هو "جوهر" MCC حيث لا توضع النقطة في خلية واحدة بل توزع كثافتها
        for i in range(NS):
            for j in range(NS):
                # إحداثيات مركز الخلية
                xi = (i - (NS-1)/2.0) * (2.0*R / NS)
                yj = (j - (NS-1)/2.0) * (2.0*R / NS)
                
                # المساهمة المكانية (Gaussian)
                ds_sq = (rx - xi)**2 + (ry - yj)**2
                gs = np.exp(-ds_sq / (2.0 * SIGMA_S**2))
                
                if gs < 0.01: continue
                
                for k in range(ND):
                    # زاوية الطبقة
                    phi_k = (k * 2.0 * np.pi / ND) - np.pi
                    
                    # المساهمة الزاوية (Wrapped Gaussian)
                    da = (d_angle - phi_k + np.pi) % (2 * np.pi) - np.pi
                    gd = np.exp(-(da**2) / (2.0 * SIGMA_D**2))
                    
                    cylinder[i, j, k] += gs * gd
                    
    # تطبيع الأسطوانة لجعلها Sigmoid-like (Saturated)
    cylinder = 1.0 / (1.0 + np.exp(-10.0 * (cylinder - 0.1)))
    return cylinder

def compute_mcc_descriptors(minutiae: List[Dict[str, Any]]) -> List[np.ndarray]:
    """توليد الواصفات لجميع النقاط."""
    descriptors = []
    for m in minutiae:
        descriptors.append(get_cylinder(m, minutiae))
    return descriptors

def local_similarity(c1: np.ndarray, c2: np.ndarray) -> float:
    """حساب التشابه المحلي بين أسطوانتين."""
    # استخدام المسافة المقلوبة أو التشابه المباشر
    diff = c1 - c2
    norm_diff = np.linalg.norm(diff)
    norm_c1 = np.linalg.norm(c1)
    norm_c2 = np.linalg.norm(c2)
    
    if (norm_c1 + norm_c2) == 0: return 0.0
    
    # 1 - المسافة المعيارية
    return 1.0 - (norm_diff / (norm_c1 + norm_c2))

def match_mcc(descriptors1: List[np.ndarray], descriptors2: List[np.ndarray]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    مطابقة مجموعتين من الأسطوانات وإيجاد أفضل الأزواج.
    Returns: (Global Similarity Score, List of matched indices)
    """
    if not descriptors1 or not descriptors2:
        return 0.0, []
        
    # بناء مصفوفة التشابه المحلي
    sim_matrix = np.zeros((len(descriptors1), len(descriptors2)), dtype=np.float32)
    for i, d1 in enumerate(descriptors1):
        for j, d2 in enumerate(descriptors2):
            sim_matrix[i, j] = local_similarity(d1, d2)
            
    # استراتيجية المطابقة الطماعة (Greedy Matching) للتبسيط حالياً
    # في الأنظمة المعقدة يتم استخدام Relaxation Labeling
    matches = []
    used_1 = set()
    used_2 = set()
    
    # ترتيب التشابهات من الأعلى للأقل
    flat_indices = np.argsort(sim_matrix, axis=None)[::-1]
    
    for idx in flat_indices:
        i, j = np.unravel_index(idx, sim_matrix.shape)
        if sim_matrix[i, j] < 0.4: break # عتبة التشابه المحلي
        
        if i not in used_1 and j not in used_2:
            matches.append((i, j))
            used_1.add(i)
            used_2.add(j)
            
    if not matches:
        return 0.0, []
        
    # حساب النتيجة الإجمالية بناءً على أفضل n تطابقات
    n_best = min(len(matches), 12) # نركز على أفضل 12 زوج
    best_sims = [sim_matrix[i, j] for i, j in matches[:n_best]]
    global_score = np.mean(best_sims) * (len(matches) / max(len(descriptors1), len(descriptors2)))
    
    return float(global_score), matches
