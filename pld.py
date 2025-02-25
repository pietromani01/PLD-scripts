from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FrameArray = NDArray[np.int64]


class PLDAnalyzer:
    frames: FrameArray = np.array([])

    def load_data(self, data_path: Path | str) -> None:
        data_path = Path(data_path).absolute()

        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")

        if data_path.is_file():
            if data_path.suffix == ".npy":
                self.frames = np.load(data_path)
                if self.frames.ndim == 2:
                    self.frames = self.frames[np.newaxis, ...]
            else:
                raise ValueError("Unsupported file format")

        elif data_path.is_dir():
            data_files = sorted(data_path.glob("*.npy"))
            if not data_files:
                raise FileNotFoundError(f"No supported data files found in {data_path}")

            frames = [np.load(file) for file in data_files]
            self.frames = np.stack(frames, axis=0)

        else:
            raise ValueError("Invalid path")

    def plot_frame(self, index: int) -> None:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        if index >= self.frames.shape[0]:
            raise IndexError("Frame index out of range")

        plt.imshow(self.frames[index], cmap="gray")
        plt.axis("off")
        plt.show()

    def make_video(self, fps: int = 2, outdir: Path | str = ".") -> None:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        outdir = Path(outdir).absolute()
        outdir.mkdir(parents=True, exist_ok=True)

        height, width = self.frames.shape[1:]

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(outdir / "video.mp4"), fourcc, fps, (width, height)
        )

        for frame in self.frames:
            frame = cv2.normalize(
                frame,
                None,
                alpha=0,
                beta=255,
                norm_type=cv2.NORM_MINMAX,
            ).astype(np.uint8)  # type: ignore
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            video_writer.write(frame)

        video_writer.release()

    def extract_roi(self):
        pass
