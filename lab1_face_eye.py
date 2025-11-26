import cv2
import os

def run_lab1(camera_index=0,
             eye_upper_fraction=0.55,     # искать глаза в верхних 55% прямоугольника лица
             face_min_size=(80, 80),      # minSize для детектора лица
             eye_min_neighbors=8,         # minNeighbors для детекции глаз (чем больше - тем жёстче)
             eye_min_size_ratio=0.06):    # минимальный размер глаза как доля ширины лица
    """
    Улучшенный запуск Лаб.1: поиск лиц и глаз с фильтрацией ложных срабатываний (например, носа).

    Параметры можно подбирать под вашу камеру / освещение:
    - eye_upper_fraction: доля высоты лица, в которой мы ищем глаза (0.5..0.65 обычно хорошо)
    - eye_min_neighbors: большее значение уменьшит ложные срабатывания, но может потерять слабые глаза
    - eye_min_size_ratio: минимальная ширина обнаруженного глаза относительно ширины лица
    """

    # каскады: предпочитаем "tree_eyeglasses", если есть — он стабильнее
    haar_dir = cv2.data.haarcascades
    eye_cascade_path = os.path.join(haar_dir, "haarcascade_eye_tree_eyeglasses.xml")
    if not os.path.exists(eye_cascade_path):
        eye_cascade_path = os.path.join(haar_dir, "haarcascade_eye.xml")

    face_cascade = cv2.CascadeClassifier(os.path.join(haar_dir, "haarcascade_frontalface_default.xml"))
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

    # попытка загрузить каскад носа (если есть в вашей сборке OpenCV)
    nose_cascade_path = os.path.join(haar_dir, "haarcascade_mcs_nose.xml")
    nose_cascade = None
    if os.path.exists(nose_cascade_path):
        nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Не удалось открыть камеру", camera_index)
        return

    # CLAHE для улучшения контраста в ROI
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=face_min_size)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # ограничиваем зону поиска глаз верхней частью лица
            y_end = int(y + h * eye_upper_fraction)
            roi_gray = gray[y:y_end, x:x + w]
            roi_color = frame[y:y_end, x:x + w]

            # предобработка: CLAHE (лучше для разных освещений), небольшое размытие для снижения шума
            if roi_gray.size == 0:
                continue
            equalized = clahe.apply(roi_gray)
            equalized = cv2.GaussianBlur(equalized, (3,3), 0)

            # minSize для глаз завязан на ширине лица
            min_eye_w = max(8, int(w * eye_min_size_ratio))
            min_eye_h = max(6, int(h * eye_min_size_ratio))

            eyes = eye_cascade.detectMultiScale(equalized,
                                                scaleFactor=1.05,
                                                minNeighbors=eye_min_neighbors,
                                                minSize=(min_eye_w, min_eye_h))

            # опционально: детектируем нос в полном лице и запомним прямоугольники,
            # чтобы исключать кандидаты глаз, которые сильно перекрываются с носом
            noses_global = []
            if nose_cascade is not None:
                face_gray_full = gray[y:y+h, x:x+w]
                noses = nose_cascade.detectMultiScale(face_gray_full, scaleFactor=1.1, minNeighbors=5, minSize=(int(w*0.08), int(h*0.08)))
                for (nx, ny, nw, nh) in noses:
                    noses_global.append((x + nx, y + ny, nw, nh))

            # перебор найденных глаз; фильтрация по положению и перекрытию с носом
            for (ex, ey, ew, eh) in eyes:
                # глобальные координаты глаза
                gx, gy = x + ex, y + ey

                # фильтр: глаз должен быть в верхней части лица (мы уже ограничили ROI, но дополнительно проверим)
                if gy > y + h * 0.6:
                    # слишком низко в лице — скорее всего нос
                    continue

                # дополнительная проверка формы: глаз обычно шире, чем выше
                aspect = ew / float(eh) if eh > 0 else 0
                if aspect < 0.8:  # если слишком «квадратно» или вертикально — подозрительно
                    continue

                # фильтр перекрытия с носом: если каскад носа нашёл пересечение — пропускаем
                skip = False
                for (nx, ny, nw, nh) in noses_global:
                    # простая проверка перекрытия центров
                    if (gx + ew/2) >= nx and (gx + ew/2) <= (nx + nw) and (gy + eh/2) >= ny and (gy + eh/2) <= (ny + nh):
                        skip = True
                        break
                if skip:
                    continue

                # нарисовать глаз (координаты в координатах кадра)
                cv2.rectangle(frame, (gx, gy), (gx + ew, gy + eh), (0, 255, 0), 2)

        cv2.imshow("Lab1: face & eyes (press ESC or q to quit)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
