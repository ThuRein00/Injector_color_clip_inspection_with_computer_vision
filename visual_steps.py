import cv2
import numpy as np
import os
import sys

# ── CONFIG ─────────────────────────────────────────────────────
HSV_LOWER  = np.array([2,  30,  30])
HSV_UPPER  = np.array([20, 255, 255])
PATCH_SIZE = 60

OUTPUT_DIR = "step_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# INPUT
image_path = "captured_images/trigger_20260227_002502_307877.jpg"
img = cv2.imread(image_path)

h, w = img.shape[:2]
x_lo = w // 3
x_hi = 2 * w // 3

def save(name, step_num, image, label=None):
    """Save image with a step label burned in."""
    out = image.copy()
    if len(out.shape) == 2:
        # if output is gray scale, change it back to BGR to put text
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    if label:
        cv2.putText(out, label, (12, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
    step_str = f"{step_num}" if isinstance(step_num, int) else f"{step_num}"
    fname = os.path.join(OUTPUT_DIR, f"step{step_str}_{name}.png")
    cv2.imwrite(fname, out)
    print(f"  Saved: {fname}")


# STEP 1: Original frame (full) 
save("original_full", 1, img, "Step 1: Original Frame")

# STEP 2: Crop to middle third 
crop = img[:, x_lo:x_hi].copy()
save("middle_third_crop", 2, crop, "Step 2: Middle-Third Crop")

# All subsequent steps operate on the cropped region only
h, w = crop.shape[:2]

# STEP 3: BGR -> HSV conversion
hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
save("hsv", 3, hsv, "Step 3: HSV Color Space")

# Show individual HSV channels 
h_ch, s_ch, v_ch = cv2.split(hsv)
save("hsv_hue",        "3a", h_ch, "Step 3a: Hue Channel")
save("hsv_saturation", "3b", s_ch, "Step 3b: Saturation Channel")
save("hsv_value",      "3c", v_ch, "Step 3c: Value Channel")

# STEP 4: Color threshold (binary mask) 
mask_raw = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
save("color_threshold", 4, mask_raw, "Step 4: Color Threshold Mask")

# STEP 5: Contour detection 
contours, _ = cv2.findContours(mask_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_vis  = crop.copy()

# draw all contours in blue, largest in cyan
cv2.drawContours(contour_vis, contours, -1, (255, 100, 0), 1)
cnt = max(contours, key=cv2.contourArea)
cv2.drawContours(contour_vis, [cnt], 0, (0, 220, 255), 2)
save("contours", 5, contour_vis, "Step 5: Contours")

# STEP 6: minAreaRect 
rect = cv2.minAreaRect(cnt)
(cx, cy), (rw, rh), angle = rect
width   = float(max(rw, rh))
clip_x  = int(cx)
clip_y  = int(cy)
box     = np.intp(cv2.boxPoints(rect))

mar_vis = crop.copy()
cv2.drawContours(mar_vis, [box], 0, (0, 255, 0), 2)
cv2.line(mar_vis, (clip_x, clip_y - 20), (clip_x, clip_y + 20), (0, 255, 0), 2)
cv2.circle(mar_vis, (clip_x, clip_y), 6, (0, 255, 0), -1)

clip_bottom = max(box[:, 1]) + 10
cv2.putText(mar_vis, f"Width: {width:.0f}px",
            (clip_x - 80, clip_bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
cv2.putText(mar_vis, f"Center: ({clip_x}, {clip_y})",
            (clip_x - 80, clip_bottom + 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

save("min_area_rect", 6, mar_vis, "Step 6: minAreaRect")

print(f"\nDone. {len(os.listdir(OUTPUT_DIR))} images saved to '{OUTPUT_DIR}/'")
print(f"  Detected clip width : {width:.1f} px")
print(f"  Clip centroid       : ({clip_x}, {clip_y})")