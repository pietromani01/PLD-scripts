import numpy as np
from pathlib import Path
import cv2


def make_video(input_path: Path | str, fps: int = 2) -> None:
    fps = 2  # 1 frame per 0.5s → 2 FPS

    workdir = Path(input_path).absolute()

    data_files = sorted(workdir.glob("*.npy"))

    frame: np.ndarray = np.load(data_files[0])
    height, width = frame.shape[:2]

    # Ensure the array is in uint8 format (0-255) for OpenCV
    frame = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame

    # Define video writer
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")  # Use 'XVID' for AVI
    video_writer = cv2.VideoWriter(
        filename=workdir / "video.mp4",
        fourcc=fourcc,
        fps=fps,
        frameSize=(width, height),
    )

    for data_file in data_files:
        frame = np.load(data_file)

        # Convert data to 8-bit image format if needed
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)

        # Convert grayscale to BGR (needed for OpenCV)
        if len(frame.shape) == 2:  # Grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        video_writer.write(frame)

    # Release resources
    video_writer.release()
