import cv2
import numpy as np

def get_detector():
    # Prefer SURF if available, otherwise fallback to ORB
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        detector = surf
        name = "SURF"
    except Exception:
        detector = cv2.ORB_create(nfeatures=1000)
        name = "ORB"
    return detector, name

def match_and_draw(img1, img2, detector, ratio=0.75):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if hasattr(detector, 'detectAndCompute'):
        kp1, des1 = detector.detectAndCompute(gray1, None)
        kp2, des2 = detector.detectAndCompute(gray2, None)
    else:
        kp1 = detector.detect(gray1, None)
        kp1, des1 = detector.compute(gray1, kp1)
        kp2 = detector.detect(gray2, None)
        kp2, des2 = detector.compute(gray2, kp2)

    if des1 is None or des2 is None:
        return None

    # matcher: for ORB use Hamming, for SURF use BF with NORM_L2
    if des1.dtype == np.uint8 or des2.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m,n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)

    draw = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return draw, len(kp1), len(kp2), len(good)

def run_lab2(template_path, camera_index=None):
    img_template = cv2.imread(template_path)
    if img_template is None:
        print("Не найден шаблон:", template_path); return
    detector, name = get_detector()
    print("Используется детектор:", name)
    if camera_index is None:
        # Compare template to an image file provided as second arg? For now just show matches template->itself
        draw, _,_,_ = match_and_draw(img_template, img_template, detector)
        cv2.imshow("Matches (template vs itself)", draw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Не удалось открыть камеру", camera_index); return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = match_and_draw(frame, img_template, detector)
        if out is None:
            cv2.imshow("Lab2: no descriptors", frame)
        else:
            draw, n1,n2,ng = out
            cv2.putText(draw, f"kp_frame {n1} kp_template {n2} matches {ng}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.imshow("Lab2: feature matching (press ESC to quit)", draw)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
