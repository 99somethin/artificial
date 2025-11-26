import cv2
import numpy as np
import os
from utils import to_grayscale, resample_contour, contour_to_complex_vector, nsp_similarity

def extract_primary_contours(img, min_area=100):
    gray = to_grayscale(img)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    good = [c for c in cnts if cv2.contourArea(c) >= min_area]
    return good, th

def contour_descriptor(contour, k=128):
    res = resample_contour(contour, k)
    vec = contour_to_complex_vector(res)
    # normalize energy
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def load_templates_from_dir(dirpath, k=128):
    templates = []
    for fn in os.listdir(dirpath):
        p = os.path.join(dirpath, fn)
        img = cv2.imread(p)
        if img is None: continue
        cnts, _ = extract_primary_contours(img)
        if len(cnts) == 0: continue
        # take largest by area
        c = max(cnts, key=cv2.contourArea)
        desc = contour_descriptor(c, k=k)
        templates.append((fn, desc))
    return templates

def match_frame_to_templates(frame, templates, k=128):
    cnts, th = extract_primary_contours(frame)
    found = []
    for c in cnts:
        desc = contour_descriptor(c, k=k)
        best = None
        best_score = -1
        for name, tdesc in templates:
            sc = nsp_similarity(desc, tdesc)
            if sc > best_score:
                best_score = sc
                best = name
        found.append((c, best, best_score))
    return found, th

def run_lab3(templates_dir, camera_index=None):
    templates = load_templates_from_dir(templates_dir)
    print(f"Загружено шаблонов: {len(templates)}")
    if camera_index is None:
        # process sample images in templates_dir and print scores
        for name, desc in templates:
            print("Шаблон:", name, "длина desc:", len(desc))
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Не удалось открыть камеру"); return
    while True:
        ret, frame = cap.read()
        if not ret: break
        found, th = match_frame_to_templates(frame, templates)
        out = frame.copy()
        for c, name, score in found:
            color = (0,255,0) if score > 0.6 else (0,0,255)
            cv2.drawContours(out, [c], -1, color, 2)
            # centroid
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"]/M["m00"])
                cy = int(M["m01"]/M["m00"])
                cv2.putText(out, f"{name}:{score:.2f}", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imshow("Lab3: contour matching", out)
        cv2.imshow("binary", th)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
