import cv2
import numpy as np

url = r"C:\Users\User\Desktop\CV\image\lalaland.jpg"
img = cv2.imread(url)

if img is None:
    print("Image not found")
    exit()

hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

selected_hsv = None

# ---------------- CREATE COLOR PALETTE ----------------
h = np.linspace(0, 179, 300).astype(np.uint8)
v = np.linspace(0, 255, 300).astype(np.uint8)

H, V = np.meshgrid(h, v)
S = np.full_like(H, 255)

palette = np.dstack((H, S, V))
palette_bgr = cv2.cvtColor(palette, cv2.COLOR_HSV2BGR)

# ---------------- MOUSE CLICK ----------------
def pick_color(event, x, y, flags, param):
    global selected_hsv

    if event == cv2.EVENT_LBUTTONDOWN:
        selected_hsv = palette[y, x]
        print("Selected HSV:", selected_hsv)

# ---------------- TRACKBAR ----------------
def nothing(x):
    pass

cv2.namedWindow("Image")
cv2.namedWindow("Color Panel")
cv2.namedWindow("Result")

cv2.createTrackbar("Tolerance", "Result", 20, 100, nothing)

cv2.setMouseCallback("Color Panel", pick_color)

# ---------------- MAIN LOOP ----------------
while True:

    cv2.imshow("Image", img)
    cv2.imshow("Color Panel", palette_bgr)

    if selected_hsv is not None:

        tolerance = cv2.getTrackbarPos("Tolerance", "Result")

        h, s, v = selected_hsv

        low = np.array([max(h - tolerance, 0), 50, 50])
        high = np.array([min(h + tolerance, 179), 255, 255])

        mask = cv2.inRange(hsv_img, low, high)
        result = cv2.bitwise_and(img, img, mask=mask)

        # -------- COLOR PREVIEW BOX --------
        preview = np.zeros((100, 100, 3), dtype=np.uint8)
        preview[:] = cv2.cvtColor(
            np.uint8([[[h, s, v]]]),
            cv2.COLOR_HSV2BGR
        )

        cv2.imshow("Selected Color", preview)
        cv2.imshow("Result", result)

    key = cv2.waitKey(1)

    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
