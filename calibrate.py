import time
import cv2
import numpy as np
import mediapipe as mp
from landmarks import build_landmarker, eye_coords, eye_vertical, RIGHT_EYE, LEFT_EYE, head_rotation, head_translation
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

CANVAS_W = 1920
CANVAS_H = 1080

GRID_N = 3
GRID = np.linspace(0.1, 0.9, GRID_N)
TARGETS = [(fx, fy) for fy in GRID for fx in GRID]

SETTLE_TIME = 0.7
RECORD_TIME = 1.0

POSES = [
    "look at the dot, head straight",
    "eyes on dot, turn head LEFT",
    "eyes on dot, turn head RIGHT",
    "eyes on dot, tilt head UP",
    "eyes on dot, tilt head DOWN",
]

POLY_DEGREE = 1
RIDGE_ALPHA = 1e-3
EMA_ALPHA = 0.2

WINDOW_NAME = "calibration"


def feature(frame, face, faceMatrix):
    uL, vL = eye_coords(frame, face, LEFT_EYE)
    uR, vR = eye_coords(frame, face, RIGHT_EYE)
    a1, a2, a3 = head_rotation(faceMatrix)
    tx, ty, tz = head_translation(faceMatrix)
    wL = eye_vertical(frame, face, LEFT_EYE)
    wR = eye_vertical(frame, face, RIGHT_EYE)
    return [uL, vL, uR, vR, a1, a2, a3, tx, ty, tz, wL, wR]


def draw_prompt(canvas, text, cx, cy):
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = int(cx - tw / 2)
    y = int(cy - 50) if cy > CANVAS_H / 2 else int(cy + 50 + th)
    x = max(10, min(x, CANVAS_W - tw - 10))
    y = max(th + 10, min(y, CANVAS_H - 10))
    cv2.putText(canvas, text, (x, y), font, scale, (255, 255, 255), thick)


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

        for prompt in POSES:
            TEMP = []
            t0 = time.time()

            while time.time() - t0 <= SETTLE_TIME + RECORD_TIME:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                ts_ms = int(time.time() * 1000)
                result = landmarker.detect_for_video(mp_image, ts_ms)

                canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
                recording = time.time() - t0 > SETTLE_TIME
                cv2.circle(canvas, (int(tx), int(ty)), 20, (0, 255, 0) if recording else (0, 0, 255), -1)
                draw_prompt(canvas, prompt, tx, ty)

                if recording and len(result.face_landmarks) > 0 and len(result.facial_transformation_matrixes) > 0:
                    face = result.face_landmarks[0]
                    faceMatrix = result.facial_transformation_matrixes[0]
                    TEMP.append(feature(frame, face, faceMatrix))

                cv2.imshow(WINDOW_NAME, canvas)
                cv2.waitKey(1)

            if TEMP:
                X.append(np.median(TEMP, axis=0))
                Y.append([tx, ty])

    cap.release()
    cv2.destroyAllWindows()

    return np.array(X), np.array(Y)


def fit(X, Y):
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=POLY_DEGREE, include_bias=True),
        Ridge(alpha=RIDGE_ALPHA),
    )
    model.fit(X, Y)
    return model


def predict(model, feat):
    return model.predict([feat])[0]


def live(model, alpha=EMA_ALPHA):
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

        if len(result.face_landmarks) > 0 and len(result.facial_transformation_matrixes) > 0:
            face = result.face_landmarks[0]
            faceMatrix = result.facial_transformation_matrixes[0]
            feat = feature(frame, face, faceMatrix)
            raw_predict_coord = predict(model, feat)
            predict_coord = raw_predict_coord * alpha + predict_coord * (1 - alpha)

        cv2.circle(canvas, (int(predict_coord[0]), int(predict_coord[1])), 20, (0, 0, 255), -1)

        cv2.imshow(WINDOW_NAME, canvas)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    X, Y = collect()
    model = fit(X, Y)
    live(model)


if __name__ == "__main__":
    main()
