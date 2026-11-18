import cv2
import numpy as np

IMAGE_SIZE = (224, 224)


def prepare_rgb_image(rgb_image):
    """Prepare one RGB image using the same preprocessing as training."""
    image = np.asarray(rgb_image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )

    if len(faces):
        x, y, width, height = max(faces, key=lambda box: box[2] * box[3])
        pad_x = int(width * 0.15)
        pad_y = int(height * 0.15)

        image = image[
            max(0, y - pad_y):min(image.shape[0], y + height + pad_y),
            max(0, x - pad_x):min(image.shape[1], x + width + pad_x),
        ]

    return cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def predict_image(model, rgb_image):
    """Return the model's estimated fake probability for one RGB image."""
    prepared = prepare_rgb_image(rgb_image)
    batch = np.expand_dims(prepared, axis=0)
    score = model.predict(batch, verbose=0).reshape(-1)

    if score.size != 1:
        raise ValueError("Expected one fake-probability score")

    return float(score[0])

def predict_video(model, video_path, max_frames=24):
    """Score evenly sampled video frames and return their timestamps and mean score."""
    if max_frames < 1:
        raise ValueError("max_frames must be at least 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count > 0:
            frame_indices = np.linspace(
                0,
                frame_count - 1,
                num=min(max_frames, frame_count),
                dtype=int,
            )
        else:
            frame_indices = range(max_frames)

        results = []

        for frame_number in frame_indices:
            if frame_count > 0:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))

            ok, bgr_frame = capture.read()
            if not ok:
                continue

            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            score = predict_image(model, rgb_frame)

            results.append({
                "frame": int(frame_number),
                "timestamp_seconds": round(float(frame_number) / fps, 3),
                "fake_probability": score,
            })

        if not results:
            raise ValueError("No readable frames found in the video")

        mean_score = float(np.mean([item["fake_probability"] for item in results]))

        return {
            "frames": results,
            "mean_fake_probability": mean_score,
        }
    finally:
        capture.release()


