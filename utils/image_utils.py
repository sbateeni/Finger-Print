import base64
import io
import numpy as np
import cv2
from PIL import Image


def _img_data_uri(img: np.ndarray, ext: str = ".jpg") -> str:
    if img is None:
        return ""
    if len(img.shape) == 2:
        ok, buf = cv2.imencode(ext, img)
    else:
        ok, buf = cv2.imencode(ext, img)
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _decode_upload_type(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        return img
    pil = Image.open(io.BytesIO(raw))
    return np.array(pil.convert("L"))
