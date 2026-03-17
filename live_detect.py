
import cv2
import numpy as np
import time
import os
import serial
from datetime import datetime

# ________________________ CONFIG ______________________________________
SERIAL_PORT      = '/dev/ttyACM0' # arduino connection
BAUD_RATE        = 9600
RESULTS_PATH     = "results"
CROP_W, CROP_H   = 612 * 2, 320 * 2
X_START, Y_START = 348, 200 * 2
FREEZE_DUR       = 1.5

# paste stats from get_stats.py
WIDTH_LOWER  = 134
WIDTH_UPPER  = 150
DEV_LOWER    = -10.3
DEV_UPPER    = 4.6
LIGHT_MEAN   = 188.0
LIGHT_TOLERANCE = 0.10

# Black
INJECTOR_LOWER = np.array([0,   20,   20])
INJECTOR_UPPER = np.array([180, 255, 80])

# Orange
HSV_LOWER  = np.array([2,  100,  100])
HSV_UPPER  = np.array([20, 255, 255])
PATCH_SIZE = 60
# _____________________________________________________________________


# _________________ LIGHTING __________________________________________
def check_light(img):
    """Checks if the lighting patch at the top of the image is within acceptable brightness range."""
    patch      = img[0:PATCH_SIZE, :, :]
    brightness = float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]))
    lo = LIGHT_MEAN * (1 - LIGHT_TOLERANCE)
    hi = LIGHT_MEAN * (1 + LIGHT_TOLERANCE)
    if brightness < lo:
        return False, brightness, 'TOO DARK' #status, brightness level, message
    if brightness > hi:
        return False, brightness, 'TOO BRIGHT'
    return True, brightness, 'OK'


def draw_light(img, light_ok, brightness, message):
    """Draws the lighting status text and a red border overlay if lighting is out of range."""
    h, w = img.shape[:2]
    if light_ok:
        text, color = f"LIGHT {message}  {brightness:.1f}", (0, 100, 0)
    else:
        text, color = f"LIGHT {message}  {brightness:.1f}", (0, 0, 255)
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, text, (w - tw - 10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.rectangle(img, (0, 0), (w - 1, PATCH_SIZE + 10), (255, 255, 255), 1) # draw lighting test patch


# ______________________________INJECTOR__________________________________________
def get_injector_cx(img):
    """Returns the center of injector, bounding box, and detection flag of the injector body."""
    h, w = img.shape[:2]

    hsv  = cv2.cvtColor(cv2.GaussianBlur(img, (11, 11), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, INJECTOR_LOWER, INJECTOR_UPPER)

    # restrict to middle third, upper two-thirds 
    mask[h//2:, :]  = 0   # zero out bottom third
    mask[:, :w//3]    = 0   # zero out left third
    mask[:, 2*w//3:]  = 0   # zero out right third

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.erode(mask,  np.ones((3, 3), np.uint8), iterations=1)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None, None, None, False

    cnt  = max(contours, key=cv2.contourArea) # eliminate noise contours
    rect = cv2.minAreaRect(cnt)
    box  = np.intp(cv2.boxPoints(rect))
    (cx, cy), _, _ = rect

    return int(cx), int(cy), box, True


# __________________________DETECTION_____________________________________________
def detect(img, decision=False):

    """
    Detects the clip and returns an annotated frame, clip width, and status.
    In live mode (decision=False) just draws the clip. (within middle third of the frame)
    In decision mode checks width and deviation (injector at the middle).
    Injector is good, if clip is detected -> wdth check is passed -> deviation check is passed
    
    """
    out          = img.copy()
    h_img, w_img = out.shape[:2]
    x_lo = w_img // 3
    x_hi = 2 * w_img // 3

    # middle third lines
    cv2.line(out, (x_lo, 0), (x_lo, h_img), (200, 200, 200), 1)
    cv2.line(out, (x_hi, 0), (x_hi, h_img), (200, 200, 200), 1)

    # Clip mask 
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), HSV_LOWER, HSV_UPPER)
    mask[:, :x_lo] = 0
    mask[:, x_hi:] = 0

    clips = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    if not clips:
        label = 'NO CLIP' if decision else 'NO DETECTION'
        cv2.putText(out, label, (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
        return out, None, 'NO CLIP'

    cnt       = max(clips, key=cv2.contourArea)
    clip_rect = cv2.minAreaRect(cnt)
    (clip_cx, clip_cy), (clip_rw, clip_rh), _ = clip_rect
    width     = float(max(clip_rw, clip_rh)) 
    clip_cx   = int(clip_cx)
    clip_cy   = int(clip_cy)
    clip_box  = np.intp(cv2.boxPoints(clip_rect))
    clip_bottom = max(clip_box[:, 1]) + 10 # for annotation

    # __________________________LIVE MODE (injector within middle third)__________________________
    if not decision:
        cv2.drawContours(out, [clip_box], 0, (255, 100, 0), 2)
        cv2.line(out, (clip_cx, 0), (clip_cx, h_img), (255, 100, 0), 1)
        cv2.putText(out, 'Clip Detected', (50, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 100, 0), 3)
        width_text = f"Width:{width:.0f}px"
        (ww, _), _ = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(out, width_text, (clip_cx - ww // 2, clip_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)
        return out, width, 'Clip Detected'

    # ________________________DECISION MODE (injector in the middle) _____________________________

    # Stage 1 (Width check)
    if width < WIDTH_LOWER or width > WIDTH_UPPER:
        color, status = (0, 0, 255), 'DEFECT'
        cv2.drawContours(out, [clip_box], 0, color, 2)
        cv2.putText(out, status, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
        width_text = f"Width:{width:.0f} (width fail)"
        (ww, _), _ = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(out, width_text, (clip_cx - ww // 2, clip_bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(out, f"Width:[{WIDTH_LOWER:.0f}-{WIDTH_UPPER:.0f}]",
                    (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        return out, width, status # if width check fails, injector is defective

    # if width check is passed, go to stage 2
    # Stage 2 (Deviation check)
    inj_cx, inj_cy, inj_box, inj_found = get_injector_cx(img)

    # if there is no injector, pass dummy values
    if not inj_found:
        inj_cx  = w_img // 2
        inj_cy  = h_img // 2
        inj_box = None

    deviation = clip_cx - inj_cx   # minAreaRect center difference ( assume rotation of clip or injector is negligible)

    if DEV_LOWER <= deviation <= DEV_UPPER:
        color, status = (0, 255, 0), 'GOOD'
    else:
        color, status = (0, 0, 255), 'DEFECT'

    # Draw clip box + vertical line
    cv2.drawContours(out, [clip_box], 0, color, 2)
    cv2.line(out, (clip_cx, 0), (clip_cx, h_img), color, 1)

    # Draw injector box + vertical line
    if inj_box is not None:
        cv2.drawContours(out, [inj_box], 0, (0, 200, 255), 2)
    line_color = (0, 255, 255) if inj_found else (0, 100, 100)
    cv2.line(out, (inj_cx, 0), (inj_cx, h_img), line_color, 1)
    inj_label = f"Inj cx={inj_cx}" if inj_found else "Fallback"
    cv2.putText(out, inj_label, (inj_cx + 5, PATCH_SIZE + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

    # Horizontal deviation line between clip center and injector center
    cv2.line(out, (inj_cx, clip_cy), (clip_cx, clip_cy), (255, 100, 0), 2)
    cv2.circle(out, (inj_cx, clip_cy), 4, (255, 100, 0), -1)

    # Labels
    cv2.putText(out, status, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
    width_text = f"Width:{width:.0f}px"
    dev_text   = f"Dev:{deviation:+.1f}px"
    (ww, _), _ = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (dw, _), _ = cv2.getTextSize(dev_text,   cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(out, width_text, (clip_cx - ww // 2, clip_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(out, dev_text,   (clip_cx - dw // 2, clip_bottom + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(out, f"Width:[{WIDTH_LOWER:.0f}-{WIDTH_UPPER:.0f}]  Dev:[{DEV_LOWER:.1f}-{DEV_UPPER:.1f}]",
                (10, h_img - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return out, width, status


# _________________________SAVE (detection zone only)__________________________________
def save(frame, width, status, prefix):
    """Saves the detection zone (middle third) of the frame to a dated folder based on status."""
    ts      = datetime.now()
    day_dir = ts.strftime("%Y-%m-%d")
    if status == 'GOOD':
        folder = os.path.join(RESULTS_PATH, day_dir, "good")
    elif status == 'NO CLIP':
        folder = os.path.join(RESULTS_PATH, day_dir, "no_clip")
    else:
        folder = os.path.join(RESULTS_PATH, day_dir, "defect")
    os.makedirs(folder, exist_ok=True)

    # crop to middle third (detection zone) before saving
    h, w   = frame.shape[:2]
    x_lo   = w // 3
    x_hi   = 2 * w // 3
    zone   = frame[:, x_lo:x_hi]

    fname = os.path.join(folder, f"{ts.strftime('%H-%M-%S-%f')}_{prefix}_{status}.jpg")
    cv2.imwrite(fname, zone)
    print(f"Saved [{status}] → {fname}")


# _______________________MAIN___________________________________________________
print(f"Limits — Width:[{WIDTH_LOWER}-{WIDTH_UPPER}]  "
      f"Dev:[{DEV_LOWER}-{DEV_UPPER}]  L:{LIGHT_MEAN}±{LIGHT_TOLERANCE*100:.0f}%")
print("System active.   q=quit\n")

# arduino sends signal when injector is in the middle of the detection zone (physical detection)
arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)

# camera connection
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)

last_capture   = 0
decision_frame = None
capture_time   = [] # to collect avg decision time

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cropped = frame[Y_START:Y_START + CROP_H, X_START:X_START + CROP_W]
    # detect light
    light_ok, brightness, reason = check_light(cropped)

    # main detection
    live, _, _ = detect(cropped.copy(), decision=False)
    draw_light(live, light_ok, brightness, reason)

    # freeze frame for 1.5s when decision is made
    if time.time() - last_capture < FREEZE_DUR and decision_frame is not None:
        live = decision_frame.copy()

    cv2.imshow("Live Defect Detection", live)

    if arduino.in_waiting > 0:
        if arduino.readline().decode().strip() == "C":
            t_start = time.time()
            df, w, s = detect(cropped.copy(), decision=True)
            t_ms = (time.time() - t_start) * 1000
            capture_time.append(t_ms)
            if not light_ok: # warning light not ok
                cv2.putText(df, f"! LIGHT {reason}", (50, df.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 80, 255), 2)
            cv2.putText(df, f"Decision: {t_ms:.1f}ms", (10, df.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 255), 2)
            decision_frame = df
            save(df, w, s, "trigger")
            arduino.write(b'F\n')
            last_capture = time.time()

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        if capture_time:
            print(f"Inference time — mean: {np.mean(capture_time):.2f}ms  "
                  f"min: {np.min(capture_time):.2f}ms  max: {np.max(capture_time):.2f}ms")
        break

cap.release()
arduino.close()
cv2.destroyAllWindows()