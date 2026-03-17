import cv2
import time
import os
import serial
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
SERIAL_PORT    = '/dev/ttyACM0'
BAUD_RATE      = 9600
SAVE_PATH      = "captured_images"
CROP_W, CROP_H = 612 * 2, 320 * 2
X_START, Y_START = 348, 200 * 2
FREEZE_DUR     = 0.5
# ─────────────────────────────────────────────────────────────

os.makedirs(SAVE_PATH, exist_ok=True)

arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

last_capture = 0
print("System active. q=quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cropped = frame[Y_START:Y_START + CROP_H, X_START:X_START + CROP_W]
    live    = cropped.copy()

    if time.time() - last_capture < FREEZE_DUR:
        cv2.rectangle(live, (0, 0), (CROP_W - 1, CROP_H - 1), (0, 255, 0), 20)
        cv2.putText(live, "CAPTURED", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Data Collection", live)

    # arduino trigger
    if arduino.in_waiting > 0:
        if arduino.readline().decode().strip() == "C":
            ts    = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = os.path.join(SAVE_PATH, f"trigger_{ts}.jpg")
            cv2.imwrite(fname, cropped)
            arduino.write(b'F\n')
            last_capture = time.time()
            print(f"Captured: {fname}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()
