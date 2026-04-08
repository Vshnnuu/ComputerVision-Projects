import argparse

import cv2
import joblib
import numpy as np
import torch
from mediapipe.python.solutions import face_mesh as mp_face_mesh

from src import config as cfg
from src.model import LSTMClassifier
from src.utils import ensure_output_dir


LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14

NOSE_TIP = 1
LEFT_FACE_EDGE = 234
RIGHT_FACE_EDGE = 454

FOREHEAD = 10
CHIN = 152

KEYPOINT_INDICES = [
    LEFT_MOUTH,
    RIGHT_MOUTH,
    UPPER_LIP,
    LOWER_LIP,
    NOSE_TIP,
    LEFT_FACE_EDGE,
    RIGHT_FACE_EDGE,
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_point(landmarks, idx, width, height):
    lm = landmarks[idx]
    return np.array([lm.x * width, lm.y * height], dtype=np.float32)


def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)


def moving_average(signal, window_size=7):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) < window_size:
        return signal
    return np.convolve(signal, np.ones(window_size) / window_size, mode="same")


def fill_nans(arr):
    arr = np.array(arr, dtype=np.float32)
    if np.all(np.isnan(arr)):
        raise ValueError("All extracted signal values are NaN. No face was detected in the video.")
    valid = np.where(~np.isnan(arr))[0]
    arr[np.isnan(arr)] = np.interp(np.where(np.isnan(arr))[0], valid, arr[valid])
    return arr


def get_model_config():
    dropout = getattr(cfg, "DROPOUT", 0.3)
    bidirectional = getattr(cfg, "BIDIRECTIONAL", True)
    return dropout, bidirectional


def load_scaler():
    scaler_path = cfg.OUTPUT_DIR / "scaler.pkl"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Could not find scaler at {scaler_path}. Run training first with: python main.py"
        )
    return joblib.load(scaler_path)


def build_model(device: torch.device) -> LSTMClassifier:
    dropout, bidirectional = get_model_config()

    model = LSTMClassifier(
        input_size=len(cfg.FEATURE_COLUMNS),
        hidden_size=cfg.HIDDEN_SIZE,
        num_layers=cfg.NUM_LAYERS,
        num_classes=len(cfg.CLASS_NAMES),
        dropout=dropout,
        bidirectional=bidirectional,
    ).to(device)

    model_path = cfg.OUTPUT_DIR / "best_lstm_classifier.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find trained model at {model_path}. Run training first with: python main.py"
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def extract_signals_from_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    smile_signal = []
    mouth_open_signal = []
    head_turn_signal = []

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            left_mouth = get_point(face_landmarks, LEFT_MOUTH, w, h)
            right_mouth = get_point(face_landmarks, RIGHT_MOUTH, w, h)
            upper_lip = get_point(face_landmarks, UPPER_LIP, w, h)
            lower_lip = get_point(face_landmarks, LOWER_LIP, w, h)

            nose_tip = get_point(face_landmarks, NOSE_TIP, w, h)
            left_face = get_point(face_landmarks, LEFT_FACE_EDGE, w, h)
            right_face = get_point(face_landmarks, RIGHT_FACE_EDGE, w, h)

            forehead = get_point(face_landmarks, FOREHEAD, w, h)
            chin = get_point(face_landmarks, CHIN, w, h)

            face_width = euclidean_distance(left_face, right_face)
            face_height = euclidean_distance(forehead, chin)

            mouth_width = euclidean_distance(left_mouth, right_mouth)
            mouth_gap = euclidean_distance(upper_lip, lower_lip)

            smile_ratio = mouth_width / face_width if face_width > 0 else 0.0
            mouth_open_ratio = mouth_gap / face_height if face_height > 0 else 0.0

            face_center_x = (left_face[0] + right_face[0]) / 2.0
            head_turn_ratio = (nose_tip[0] - face_center_x) / face_width if face_width > 0 else 0.0

            smile_signal.append(smile_ratio)
            mouth_open_signal.append(mouth_open_ratio)
            head_turn_signal.append(head_turn_ratio)
        else:
            smile_signal.append(np.nan)
            mouth_open_signal.append(np.nan)
            head_turn_signal.append(np.nan)

    cap.release()
    face_mesh.close()

    smile_signal = fill_nans(smile_signal)
    mouth_open_signal = fill_nans(mouth_open_signal)
    head_turn_signal = fill_nans(head_turn_signal)

    smile_smooth = moving_average(smile_signal, window_size=7)
    mouth_open_smooth = moving_average(mouth_open_signal, window_size=7)
    head_turn_smooth = moving_average(head_turn_signal, window_size=7)
    head_turn_clipped = np.clip(head_turn_smooth, -0.25, 0.25)

    features = np.column_stack(
        [smile_smooth, mouth_open_smooth, head_turn_clipped]
    ).astype(np.float32)

    return {
        "features": features,
        "fps": fps,
        "width": width,
        "height": height,
        "smile": smile_smooth,
        "mouth_open": mouth_open_smooth,
        "head_turn": head_turn_clipped,
    }


def predict_frame_labels(features: np.ndarray, scaler, model, device: torch.device):
    predicted_labels = []

    with torch.no_grad():
        for frame_idx in range(len(features)):
            if frame_idx < cfg.SEQUENCE_LENGTH:
                predicted_labels.append("Collecting context...")
                continue

            seq = features[frame_idx - cfg.SEQUENCE_LENGTH:frame_idx]
            seq_scaled = scaler.transform(seq)

            x_tensor = torch.tensor(seq_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            logits = model(x_tensor)
            pred_id = torch.argmax(logits, dim=1).item()
            predicted_labels.append(cfg.CLASS_NAMES[pred_id])

    return predicted_labels


def annotate_video(
    video_path: str,
    output_path: str,
    predicted_labels,
    smile_signal,
    mouth_open_signal,
    head_turn_signal,
    fps,
    width,
    height,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video for annotation: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            for idx in KEYPOINT_INDICES:
                pt = get_point(face_landmarks, idx, w, h).astype(int)
                cv2.circle(frame, tuple(pt), 6, (0, 255, 0), -1)

            left_mouth = get_point(face_landmarks, LEFT_MOUTH, w, h).astype(int)
            right_mouth = get_point(face_landmarks, RIGHT_MOUTH, w, h).astype(int)
            upper_lip = get_point(face_landmarks, UPPER_LIP, w, h).astype(int)
            lower_lip = get_point(face_landmarks, LOWER_LIP, w, h).astype(int)
            nose_tip = get_point(face_landmarks, NOSE_TIP, w, h).astype(int)
            left_face = get_point(face_landmarks, LEFT_FACE_EDGE, w, h).astype(int)
            right_face = get_point(face_landmarks, RIGHT_FACE_EDGE, w, h).astype(int)

            face_center = ((left_face + right_face) / 2).astype(int)

            cv2.line(frame, tuple(left_mouth), tuple(right_mouth), (255, 0, 0), 3)
            cv2.line(frame, tuple(upper_lip), tuple(lower_lip), (0, 0, 255), 3)
            cv2.line(frame, tuple(face_center), tuple(nose_tip), (255, 255, 0), 3)

        label = predicted_labels[frame_id]

        cv2.putText(frame, "Prediction:", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 255), 4)
        cv2.putText(frame, label, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

        overlay_y = height - 220
        cv2.putText(
            frame,
            f"Smile Ratio: {smile_signal[frame_id]:.3f}",
            (20, overlay_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            3,
        )
        cv2.putText(
            frame,
            f"Mouth Open Ratio: {mouth_open_signal[frame_id]:.3f}",
            (20, overlay_y + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )
        cv2.putText(
            frame,
            f"Head Turn Ratio: {head_turn_signal[frame_id]:.3f}",
            (20, overlay_y + 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 0),
            3,
        )

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    face_mesh.close()


def main():
    parser = argparse.ArgumentParser(description="Run Project 2 LSTM inference on a video.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument(
        "--output",
        type=str,
        default=str(cfg.OUTPUT_DIR / "annotated_lstm_output.mp4"),
        help="Path to save annotated output video",
    )
    args = parser.parse_args()

    ensure_output_dir(cfg.OUTPUT_DIR)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading scaler...")
    scaler = load_scaler()

    print("Loading trained model...")
    model = build_model(device)

    print("Extracting Project 1 signals from video...")
    extracted = extract_signals_from_video(args.video)

    print("Running sequence predictions...")
    predicted_labels = predict_frame_labels(extracted["features"], scaler, model, device)

    print("Writing annotated output video...")
    annotate_video(
        video_path=args.video,
        output_path=args.output,
        predicted_labels=predicted_labels,
        smile_signal=extracted["smile"],
        mouth_open_signal=extracted["mouth_open"],
        head_turn_signal=extracted["head_turn"],
        fps=extracted["fps"],
        width=extracted["width"],
        height=extracted["height"],
    )

    print(f"Done. Saved annotated video to: {args.output}")


if __name__ == "__main__":
    main()