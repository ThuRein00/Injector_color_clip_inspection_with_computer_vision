import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt

# Configuration
HSV_LOWER  = np.array([2,  30,  30]) # orange range
HSV_UPPER  = np.array([20, 255, 255])
PATCH_SIZE = 60 # lighting test patch

INJECTOR_LOWER = np.array([0,   0,   0]) # black range
INJECTOR_UPPER = np.array([180, 255, 80])

def get_brightness(img):
    """Returns mean HSV-V brightness of the top patch (lighting reference)."""

    patch = img[0:PATCH_SIZE, :] #top of the photo (60xall)
    return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2])) # get V value


def get_injector_cx(img):
    """Finds the injector bounding box and center point"""
    h, w = img.shape[:2]

    hsv  = cv2.cvtColor(cv2.GaussianBlur(img, (11, 11), 0), cv2.COLOR_BGR2HSV) # convert from RGB to HSV
    mask = cv2.inRange(hsv, INJECTOR_LOWER, INJECTOR_UPPER) # filter background (get injector body)

    # detect only upper half, middle third of the photo (Injector is placted in the middle)
    mask[h//2:, :]   = 0   # zero out bottom half
    mask[:, :w//3]   = 0   # zero out left third
    mask[:, 2*w//3:] = 0   # zero out right third

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2) # dilation
    mask = cv2.erode(mask,  np.ones((3, 3), np.uint8), iterations=1) # erosion

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # find contours 
    if not contours:
        return None, None, None, False # ceinter x, center y , box , detected? 

    cnt  = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    box  = np.intp(cv2.boxPoints(rect)) # to draw a box on detected area
    (cx, cy), _, _ = rect # center point of min area rect

    return int(cx), int(cy), box, True


def measure(img, save_path):
    """Measures clip width and deviation from injector center"""
    #____________________get lighting status__________________
    bright = get_brightness(img)
    h_img, w_img = img.shape[:2]

    inj_cx, inj_cy, inj_box, inj_found = get_injector_cx(img)

    # if there is no injector, add dummy values (there should be injector at every photo)
    if not inj_found:
        inj_cx  = w_img // 2
        inj_cy  = h_img // 2
        inj_box = None

    #________________ detect clip ____________________________
    mask = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), HSV_LOWER, HSV_UPPER) # get only the color in the range

    #find clip in only middle third of the photo
    mask[:, :w_img//3]   = 0
    mask[:, 2*w_img//3:] = 0

    clips = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] # find clip contours

    out = img.copy()
    cv2.rectangle(out, (0, 0), (w_img - 1, PATCH_SIZE), (255, 255, 255), 1) # draw lighting sample area patch
    cv2.line(out, (w_img//3,   0), (w_img//3,   h_img), (200, 200, 200), 1) # draw detection area (middle third)
    cv2.line(out, (2*w_img//3, 0), (2*w_img//3, h_img), (200, 200, 200), 1)

    # print lighting status
    light_text = f"Light:{bright:.1f}"
    (lw, lh), _ = cv2.getTextSize(light_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(out, light_text, (w_img - lw - 10, lh + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # Draw injector box + vertical line form center from inj 
    line_color = (0, 255, 255) if inj_found else (0, 100, 100)
    if inj_box is not None:
        cv2.drawContours(out, [inj_box], 0, (0, 200, 255), 2)
    cv2.line(out, (inj_cx, 0), (inj_cx, h_img), line_color, 1)
    inj_label = f"Inj cx={inj_cx}" if inj_found else "Fallback"
    cv2.putText(out, inj_label, (inj_cx + 5, PATCH_SIZE + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_color, 1)

    # no clip detected case
    if not clips:
        cv2.imwrite(save_path, out)
        return None

    cnt = max(clips, key=cv2.contourArea)

    # ── Clip: measure minAreaRect width + deviation form injector ─────────────────────────
    clip_rect = cv2.minAreaRect(cnt)
    (clip_cx, clip_cy), (clip_rw, clip_rh), _ = clip_rect
    width   = float(max(clip_rw, clip_rh))
    clip_cx = int(clip_cx)
    clip_cy = int(clip_cy)
    clip_box = np.intp(cv2.boxPoints(clip_rect))

    # ── Deviation = clip cx - injector cx ────────────────────
    deviation = clip_cx - inj_cx

    # Draw clip box + vertical line
    cv2.drawContours(out, [clip_box], 0, (0, 255, 0), 2)
    cv2.line(out, (clip_cx, 0), (clip_cx, h_img), (0, 255, 0), 1)

    # Horizontal deviation line
    cv2.line(out, (inj_cx, clip_cy), (clip_cx, clip_cy), (255, 100, 0), 2)
    cv2.circle(out, (inj_cx, clip_cy), 4, (255, 100, 0), -1)

    # annotation
    clip_bottom = max(clip_box[:, 1]) + 10
    width_text  = f"Width:{width:.0f}px"
    dev_text    = f"Dev:{deviation:+.1f}px"
    (ww, _), _  = cv2.getTextSize(width_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (dw, _), _  = cv2.getTextSize(dev_text,   cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(out, width_text, (w_img//2 - ww//2, clip_bottom + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(out, dev_text,   (w_img//2 - dw//2, clip_bottom + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

    cv2.imwrite(save_path, out)
    return width, bright, deviation


# _______________________________MAIN______________________________
os.makedirs('train_results', exist_ok=True)

widths, lightings, deviations = [], [], []

print("--- TRAINING ---")
files = glob.glob('captured_images/*.jpg')
print(f"Found {len(files)} images in captured_images/")

for file_path in files:
    img = cv2.imread(file_path)
    if img is None:
        print(f"  Failed to load: {file_path}")
        continue

    file_name = os.path.basename(file_path)
    result    = measure(img, os.path.join('train_results', f"train_{file_name}"))

    if result:
        w, bright, dev = result
        widths.append(w)
        lightings.append(bright)
        deviations.append(dev)
        print(f"  {file_name}  Width:{w:.0f}  Light:{bright:.1f}  Dev:{dev:+.1f}px")


# _______________________________Collect stats______________________________
def sigma(arr, sigma=3):
    """Returns (mean, mean - sigma*std, mean + sigma*std) for threshold calibration."""
    m, s = np.mean(arr), np.std(arr)
    return m, m - sigma*s, m + sigma*s

w_mean, w_lo, w_hi = sigma(widths,     sigma=5)
d_mean, d_lo, d_hi = sigma(deviations, sigma=4)
l_mean              = float(np.mean(lightings))

print(f"\n{'─'*55}")
print(f"  Width    mean={w_mean:.1f}   3σ range [{w_lo:.1f} – {w_hi:.1f}]")
print(f"  Dev      mean={d_mean:.1f}   3σ range [{d_lo:.1f} – {d_hi:.1f}]")
print(f"  Light    mean={l_mean:.1f}")
print(f"{'─'*55}")
print(f"\nPaste into live_defect_detection.py:")
print(f"  WIDTH_LOWER  = {w_lo:.0f}")
print(f"  WIDTH_UPPER  = {w_hi:.0f}")
print(f"  DEV_LOWER    = {d_lo:.1f}")
print(f"  DEV_UPPER    = {d_hi:.1f}")
print(f"  LIGHT_MEAN   = {l_mean:.1f}")


# _______________________________Draw Histograms______________________________
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].hist(widths, bins=15, color='steelblue', edgecolor='white')
axes[0].axvline(w_lo,   color='red',    linestyle='--', label=f'Lower {w_lo:.0f}')
axes[0].axvline(w_hi,   color='red',    linestyle='--', label=f'Upper {w_hi:.0f}')
axes[0].axvline(w_mean, color='yellow', linestyle='-',  label=f'Mean {w_mean:.0f}')
axes[0].set_title('Width Distribution (minAreaRect)')
axes[0].set_xlabel('Width (px)')
axes[0].set_ylabel('Count')
axes[0].legend()

axes[1].hist(deviations, bins=15, color='mediumpurple', edgecolor='white')
axes[1].axvline(d_lo,   color='red',    linestyle='--', label=f'Lower {d_lo:.1f}')
axes[1].axvline(d_hi,   color='red',    linestyle='--', label=f'Upper {d_hi:.1f}')
axes[1].axvline(d_mean, color='yellow', linestyle='-',  label=f'Mean {d_mean:.1f}')
axes[1].axvline(0,      color='white',  linestyle=':',  label='Zero (centred)')
axes[1].set_title('Deviation  (clip_cx − inj_cx)')
axes[1].set_xlabel('Deviation (px)  [+ right / − left of injector]')
axes[1].set_ylabel('Count')
axes[1].legend()

axes[2].hist(lightings, bins=15, color='orange', edgecolor='white')
axes[2].axvline(l_mean, color='yellow', linestyle='-', label=f'Mean {l_mean:.1f}')
axes[2].set_title('Lighting Distribution')
axes[2].set_xlabel('Brightness (V)')
axes[2].set_ylabel('Count')
axes[2].legend()

plt.tight_layout()
plt.savefig('train_results/histograms.png', dpi=150)
plt.show()
print("Histograms saved to train_results/histograms.png")