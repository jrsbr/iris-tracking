import time
import cv2
import numpy as np
import mediapipe as mp
from landmarks import build_landmarker, eye_coords, RIGHT_EYE, LEFT_EYE

CANVAS_W = 1920
CANVAS_H = 1080
GRID = [0.1, 0.5, 0.9]
TARGETS = [(fx,fy) for fx in GRID for fy in GRID]
SETTLE_TIME, RECORD_TIME = 1, 1
WINDOW_NAME = "calibration"

def feature(frame, face):
    uL, vL = eye_coords(frame, face, LEFT_EYE)
    uR, vR = eye_coords(frame, face, RIGHT_EYE)
    return [uL, vL, uR, vR]

def collect():
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
     
    landmarker = build_landmarker()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    X, Y = [], []

    for (fx, fy) in TARGETS:
        tx, ty = fx * CANVAS_W, fy * CANVAS_H
        t0 = time.time()

        while time.time() - t0 <= SETTLE_TIME + RECORD_TIME:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb) 
            ts_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
            cv2.circle(canvas, (int(tx),int(ty)), 20, (0,0,255), -1)

            cv2.imshow(WINDOW_NAME, canvas)
            cv2.waitKey(1)

            if time.time() - t0 > SETTLE_TIME and len(result.face_landmarks) > 0:
                cv2.circle(canvas, (int(tx),int(ty)), 20, (0,255,0), -1)
                face = result.face_landmarks[0]
                X.append(feature(frame, face))
                Y.append([tx,ty])

    cap.release()
    cv2.destroyAllWindows()

    return np.array(X), np.array(Y)

def fit(X, Y):
    bias = np.ones((len(X), 1))
    Xd = np.hstack([bias, X]) # adds a column of 1s to X (as the constant terms in the linear regression)  
    W, *_ = np.linalg.lstsq(Xd, Y, rcond=None) # solves Xd * W = Y (Xd is Nx5, Y is Nx2 and W is 5x2)
    return W

def predict(W, feat):
    return np.array([1, *feat]) @ W

def live(W):
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

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print("Camera ready. Press 'q' to quit.")

    landmarker = build_landmarker()

    predict_coord = np.array([CANVAS_W / 2, CANVAS_H / 2])

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms) 
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

        if len(result.face_landmarks) > 0:
            face = result.face_landmarks[0]
            feat = feature(frame, face)
            predict_coord = predict(W, feat)

        cv2.circle(canvas, (int(predict_coord[0]),int(predict_coord[1])), 20, (0,0,255), -1)

        cv2.imshow(WINDOW_NAME, canvas)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    X, Y = collect()
    W = fit(X, Y)
    live(W)

if __name__ == "__main__":
    main()