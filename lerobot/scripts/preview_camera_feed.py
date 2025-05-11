import cv2
import sys

idx = int(sys.argv[1])      # run: python preview.py 2
cap = cv2.VideoCapture(idx)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open camera {idx}")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow(f"Camera {idx}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

