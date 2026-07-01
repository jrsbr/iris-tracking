import time

import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = "models/face_landmarker.task"
WINDOW = "landmarks"
RIGHT_EYE = {"corner1": 33,  "corner2": 133, "iris": (469, 470, 471, 472), "lid_up": 159, "lid_low": 145}
LEFT_EYE  = {"corner1": 362, "corner2": 263, "iris": (474, 475, 476, 477), "lid_up": 386, "lid_low": 374}

def build_landmarker():
    baseOpt = mp.tasks.BaseOptions(model_asset_path = MODEL_PATH)

    # creates an option file to create a FaceLandmarker
    faceLandmarkerOpt = mp.tasks.vision.FaceLandmarkerOptions(
            base_options = baseOpt,
            output_facial_transformation_matrixes=True,
            running_mode = mp.tasks.vision.RunningMode.VIDEO, # possibly change to LIVE_STREAM mode if latency issues
            num_faces = 1,
    )

    # creates a FaceLandmarker with the options above
    faceLandmarker = mp.tasks.vision.FaceLandmarker.create_from_options(faceLandmarkerOpt)
    return faceLandmarker

def draw_landmarks(frame, result, iris = 1):
    if (not result.face_landmarks):
        return
    detectedFaces = result.face_landmarks # list of landmarks / each entry is a list for each face
    h, w = frame.shape[:2]

    for face in detectedFaces:
        for i, lm in enumerate(face):
            px, py = int(lm.x * w), int(lm.y * h) # denormalizes x and y
            if (iris):
                if (i == 468 or i == 473):
                    cv2.circle(frame, (px, py), 1, (255, 0, 0), -1)
                    continue
                if (i > 468 and i <= 477):
                    cv2.circle(frame, (px, py), 1, (0, 0, 255), -1)
                    continue
            cv2.circle(frame, (px, py), 1, (0, 255, 0), -1) # draws a dot
        

def eye_coords(frame, face, eye):
    h, w = frame.shape[:2]

    def coord(index):
        lm = face[index]
        return np.array([lm.x * w , lm.y * h], np.float64)

    irisRing = np.array([coord(corner) for corner in eye["iris"]])
    irisCenter = irisRing.mean(axis=0)

    c1, c2 = eye["corner1"], eye["corner2"]
    left = coord(c1) if face[c1].x < face[c2].x else coord(c2)
    right = coord(c1) if face[c1].x > face[c2].x else coord(c2)

    eye_vec = right - left
    eye_width = np.linalg.norm(eye_vec)

    e_x = eye_vec/eye_width
    e_y = np.array([-e_x[1], e_x[0]])
    eye_origin = (left + right) / 2

    offset = irisCenter - eye_origin
    u = offset @ e_x / eye_width
    v = offset @ e_y / eye_width

    return u, v

def eye_vertical(frame, face, eye):
    h, w = frame.shape[:2]

    def coord(index):
        lm = face[index]
        return np.array([lm.x * w, lm.y * h], np.float64)

    iris = np.array([coord(i) for i in eye["iris"]]).mean(axis=0)
    up = coord(eye["lid_up"])
    low = coord(eye["lid_low"])

    span = low[1] - up[1]
    if abs(span) < 1e-6:
        return 0.0
    return (iris[1] - up[1]) / span

def head_rotation(faceTransMatrix):
    R = faceTransMatrix[:3, :3]
    angles, *_ = cv2.RQDecomp3x3(R)
    return angles[0], angles[1], angles[2]

def head_translation(faceTransMatrix):
    return faceTransMatrix[:3, 3]

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
        if len(result.face_landmarks) > 0:
            face = result.face_landmarks[0]
            hr, vr = eye_coords(frame, face, RIGHT_EYE)
            hl, vl = eye_coords(frame, face, LEFT_EYE)
            print(f"R({hr:.2f},{vr:.2f})  L({hl:.2f},{vl:.2f})")
        draw_landmarks(frame, result)

        cv2.imshow(WINDOW, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
