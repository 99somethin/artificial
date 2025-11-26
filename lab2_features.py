# lab2_features.py
import cv2
import numpy as np
import os

def init_detector(prefer="AKAZE"):
    try:
        if prefer.upper() == "SIFT":
            det = cv2.SIFT_create()
            norm = cv2.NORM_L2
            return det, norm, "SIFT"
        if prefer.upper() == "SURF":
            det = cv2.xfeatures2d.SURF_create(400)
            norm = cv2.NORM_L2
            return det, norm, "SURF"
    except Exception:
        pass

    try:
        det = cv2.AKAZE_create()
        return det, cv2.NORM_HAMMING, "AKAZE"
    except Exception:
        det = cv2.ORB_create(nfeatures=1500)
        return det, cv2.NORM_HAMMING, "ORB"

def detect_and_match(detector, norm_type, img_template, img_scene, ratio=0.75):
    g1 = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_scene, cv2.COLOR_BGR2GRAY)
    kp1, des1 = detector.detectAndCompute(g1, None)
    kp2, des2 = detector.detectAndCompute(g2, None)
    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
        return kp1 or [], kp2 or [], [], []
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    knn = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return kp1, kp2, good, knn

def compute_homography(kp1, kp2, matches, ransac_thresh=5.0):
    if len(matches) < 4:
        return None, None
    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    return H, mask

def draw_keypoints_img(img, keypoints):
    return cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def draw_matches_img(img_template, kp1, img_scene, kp2, matches):
    # compact visual: scale template down if too big for matching image composition
    out = cv2.drawMatches(img_template, kp1, img_scene, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return out

def draw_box_on_scene(img_template, img_scene, H):
    img = img_scene.copy()
    h, w = img_template.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners, H)
    cv2.polylines(img, [np.int32(projected)], True, (0,255,0), 3, cv2.LINE_AA)
    return img

def process_pair_return_imgs(detector, norm_type, img_template, img_scene,
                             do_keypoints=True, do_matches=True, do_contour=True):
    """
    Process a pair (template, scene). Return dict with images (BGR numpy) and stats.
    Keys in returned dict:
      'kp_template_img', 'kp_scene_img', 'matches_img', 'scene_box_img',
      'kp_counts', 'good_matches_count', 'H', 'mask'
    Any missing visualization will be None.
    """
    result = {
        'kp_template_img': None,
        'kp_scene_img': None,
        'matches_img': None,
        'scene_box_img': None,
        'kp_counts': (0,0),
        'good_matches_count': 0,
        'H': None,
        'mask': None
    }

    kp1, kp2, good, knn = detect_and_match(detector, norm_type, img_template, img_scene)

    result['kp_counts'] = (len(kp1), len(kp2))
    result['good_matches_count'] = len(good)

    if do_keypoints:
        try:
            result['kp_template_img'] = draw_keypoints_img(img_template, kp1)
            result['kp_scene_img'] = draw_keypoints_img(img_scene, kp2)
        except Exception:
            result['kp_template_img'] = img_template.copy()
            result['kp_scene_img'] = img_scene.copy()

    if do_matches:
        try:
            result['matches_img'] = draw_matches_img(img_template, kp1, img_scene, kp2, good)
        except Exception:
            result['matches_img'] = None

    if do_contour:
        H, mask = compute_homography(kp1, kp2, good)
        result['H'] = H
        result['mask'] = mask
        if H is not None:
            try:
                result['scene_box_img'] = draw_box_on_scene(img_template, img_scene, H)
            except Exception:
                result['scene_box_img'] = img_scene.copy()
    return result
