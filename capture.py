import time

import cv2

WINDOW = "capture"


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera (index 0).")

    warm_start = time.time()
    while time.time() - warm_start < 2.0:
        ok, _ = cap.read()
        if ok:
            break
    else:
        cap.release()
        raise RuntimeError("Camera opened but produced no frames in 2s.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    print("Camera ready. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow(WINDOW, frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
