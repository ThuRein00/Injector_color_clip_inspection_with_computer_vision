import cv2
import numpy as np
import glob
import os

# orange
HSV_LOWER = np.array([2,  0,  0])
HSV_UPPER = np.array([20, 255, 255])

# black
INJECTOR_LOWER = np.array([0,   0,   0])
INJECTOR_UPPER = np.array([180, 255, 80])

DATA_DIR = "snr_data"
d2       = 1.128          # d2 constant for n=2 measurements


def get_injector_cx(img):
    h, w    = img.shape[:2]
    x_start = w // 3
    upper   = img[0 : h//2, x_start : 2*w//3]

    mask = cv2.inRange(
        cv2.cvtColor(cv2.GaussianBlur(upper, (11, 11), 0), cv2.COLOR_BGR2HSV),
        INJECTOR_LOWER, INJECTOR_UPPER
    )
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    mask = cv2.erode(mask,  np.ones((3, 3), np.uint8), iterations=1)

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None

    cnt_full = cnt + np.array([[[x_start, 0]]])
    rect     = cv2.minAreaRect(cnt_full)
    box      = np.intp(cv2.boxPoints(rect))

    M  = cv2.moments(cnt_full)
    cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else int(rect[0][0])

    return cx


def measure_width(img_path):
    """returns width in px of the color clip"""
    img = cv2.imread(img_path)
    if img is None:
        return None

    h_img, w_img = img.shape[:2]

    mask = cv2.inRange(
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV),
        HSV_LOWER, HSV_UPPER
    )

    # middle third
    mask[:, :w_img//3]   = 0
    mask[:, 2*w_img//3:] = 0

    clips = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    if not clips:
        return None

    cnt = max(clips, key=cv2.contourArea)

    # measure width of color clip
    clip_rect = cv2.minAreaRect(cnt)
    (clip_cx, clip_cy), (clip_rw, clip_rh), _ = clip_rect
    width = float(max(clip_rw, clip_rh))

    return width


# SNR calculation
def compute_snr(DATA):
    range, all_meas = [], []

    for path in sorted(glob.glob(os.path.join(DATA, "part_*"))):
        m1 = measure_width(os.path.join(path, "meas1.jpg"))
        m2 = measure_width(os.path.join(path, "meas2.jpg"))
        if m1 is None or m2 is None:
            print(f"  Skipping {os.path.basename(path)}")
            continue
        range.append(abs(m1 - m2))
        all_meas.extend([m1, m2])

    if not range:
        print(f"No data found")
        return

    R_bar       = np.mean(range)
    sigma_gauge = R_bar / d2
    var_gauge   = sigma_gauge ** 2
    var_total   = np.var(all_meas, ddof=1)
    rho_M       = var_gauge / var_total
    rho_P       = 1 - rho_M
    SNR         = np.sqrt(2 * rho_P / (1 - rho_P)) 

    print(f"\n── Clip Width SNR data ──")
    print(f"  Parts measured : {len(range)}")
    print(f"  R_bar          : {R_bar:.3f}")
    print(f"  sigma_gauge    : {sigma_gauge:.3f}")
    print(f"  var_gauge      : {var_gauge:.3f}")
    print(f"  var_total      : {var_total:.3f}")
    print(f"  rho_M          : {rho_M:.4f}")
    print(f"  SNR            : {SNR:.2f}")
    if   SNR >= 5: print("  → ACCEPTABLE (>= 5)")
    elif SNR >= 2: print("  → MARGINAL   (2–5)")
    else:          print("  → INADEQUATE (< 2)")


compute_snr(DATA_DIR)
