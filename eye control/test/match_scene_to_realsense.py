import cv2
import numpy as np

scene = cv2.imread("pupil_scene.png", cv2.IMREAD_GRAYSCALE)
rgb   = cv2.imread("realsense_color.png", cv2.IMREAD_GRAYSCALE)

if scene is None or rgb is None:
    raise FileNotFoundError("Missing pupil_scene.png or realsense_color.png")

# ORB features
orb = cv2.ORB_create(nfeatures=4000)

kp1, des1 = orb.detectAndCompute(scene, None)
kp2, des2 = orb.detectAndCompute(rgb, None)

if des1 is None or des2 is None:
    raise RuntimeError("No descriptors found. Try better lighting/texture.")

# Match with KNN + ratio test
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
matches_knn = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches_knn:
    if m.distance < 0.75 * n.distance:
        good.append(m)

print(f"[INFO] keypoints scene={len(kp1)} rgb={len(kp2)} good_matches={len(good)}")

# RANSAC to filter geometric outliers (homography just for inlier counting)
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
pts2 = np.float32([kp2[m.trainIdx].pt  for m in good]).reshape(-1, 1, 2)

if len(good) >= 8:
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inliers = int(mask.sum()) if mask is not None else 0
else:
    mask = None
    inliers = 0

print(f"[INFO] RANSAC inliers={inliers}")

# Draw matches (inliers in green-ish by default OpenCV styling)
draw = cv2.drawMatches(
    scene, kp1, rgb, kp2, good, None,
    matchesMask=mask.ravel().tolist() if mask is not None else None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

cv2.imshow("Matches (Scene ↔ RealSense RGB)", draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
