import cv2
from ultralytics import YOLO
import datetime
import os

# YOLO modelini yükle
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

save_dir = "captures"
os.makedirs(save_dir, exist_ok=True)

previous_centers = {}

# Hareket algılama eşikleri
dx_threshold = 100  # x yönündeki değişim için eşik
dy_threshold = 100  # y yönündeki değişim için eşik

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, stream=True)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    movement_detected = False



    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            obj_id = id(box)
            prev = previous_centers.get(obj_id)

            if prev is not None:
                dx, dy = abs(cx - prev[0]), abs(cy - prev[1])
                # Geliştirilmiş eşik: küçük titreşimleri yok say
                if dx > dx_threshold or dy > dy_threshold:
                    movement_detected = True
                    cv2.putText(gray, "HAREKET ALGILANDI", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            previous_centers[obj_id] = (cx, cy)

            # Kutucuk çiz
            cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if movement_detected:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"alert_{timestamp}.jpg")
        cv2.imwrite(filename, gray)
        print(f"[ALARM] Hareket algılandı, kaydedildi: {filename}")

    cv2.imshow("YOLO Alarm", gray)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC ile çık
        break

cap.release()
cv2.destroyAllWindows()
