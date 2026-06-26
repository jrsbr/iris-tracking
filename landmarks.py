import time

import cv2
import mediapipe as mp

MODEL_PATH = "models/face_landmarker.task"
WINDOW = "landmarks"


def build_landmarker():
    baseOpt = mp.tasks.BaseOptions(model_asset_path = MODEL_PATH)

    # creates an option file to create a FaceLandmarker
    faceLandmarkerOpt = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = baseOpt,
            running_mode = mp.tasks.vision.RunningMode.VIDEO, # possibly change to LIVE_STREAM mode if latency issues
            num_faces = 1,
    )

    # creates a FaceLandmarker with the options above
    faceLandmarker = mp.tasks.vision.FaceLandmarker.create_from_options(faceLandmarkerOpt)
    return faceLandmarker

def draw_landmarks(frame, result):
    if (not result.face_landmarks):
        return
    detectedFaces = result.face_landmarks # list of landmarks / each entry is a list for each face
    h, w = frame.shape[:2]

    for face in detectedFaces:
        for lm in face:
            px, py = int(lm.x * w), int(lm.y * h) # denormalizes x and y
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1) # draws a dot

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

    landmarker = build_landmarker()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # changes from BGR to RGB as MediaPipe wants
        mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb) # returns an mp.Image from the rgb data
        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms) # returns the object of all landmarks detected per face
        draw_landmarks(frame, result)

        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
