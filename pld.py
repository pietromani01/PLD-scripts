from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage as ndi

from matplotlib.colors import LinearSegmentedColormap

FrameArray = NDArray[np.int64]

MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "matrix_cmap",
    [
        (0, "black"),
        (0.4, "#002500"),
        (0.6, "#005000"),
        (0.8, "#007500"),
        (1, "lime"),
    ],
)


class RHEEDFrame:
    def __init__(
        self,
        frame_data: FrameArray,
        roi: tuple[int, int, int, int] | None = None,
        outdir: Path | str = Path("."),
    ) -> None:
        self.data = frame_data
        self.roi = roi
        self.outdir = Path(outdir).absolute()

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.data.shape[:2]  # type: ignore

    def center(self) -> "RHEEDFrame":
        if self.roi is None:
            raise ValueError("ROI not defined")

        x_min, x_max, y_min, y_max = self.roi
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        x_offset = self.data.shape[1] // 2 - x_center
        y_offset = self.data.shape[0] // 2 - y_center

        self.data = np.roll(self.data, x_offset, axis=1)
        self.data = np.roll(self.data, y_offset, axis=0)

        self.roi = (
            self.roi[0] + x_offset,
            self.roi[1] + x_offset,
            self.roi[2] + y_offset,
            self.roi[3] + y_offset,
        )

        return self

    def crop(self, top=0, bottom=0, left=0, right=0) -> "RHEEDFrame":
        if top:
            self.data = self.data[top:, :]
        if bottom:
            self.data = self.data[:-bottom, :]
        if left:
            self.data = self.data[:, left:]
        if right:
            self.data = self.data[:, :-right]
        return self

    def smooth(self, sigma: int = 2):
        self.data = ndi.gaussian_filter(self.data, sigma=sigma)
        return self

    def plot(
        self,
        figsize: tuple[int, int] = (8, 6),
        save: bool = False,
    ):
        plt.figure(figsize=figsize)
        plt.imshow(self.data, cmap=MATRIX_CMAP)
        plt.axis("off")

        if self.roi:
            x_min, x_max, y_min, y_max = self.roi
            plt.plot(
                [x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                color="red",
                linewidth=2,
            )

        if save:
            plt.savefig(self.outdir / "frame.png")

        plt.show()


class RHEEDAnalyzer:
    frames: FrameArray = np.array([])
    roi: tuple[int, int, int, int] | None = None
    outdir = Path(".")

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

    def crop_to_roi(
        self,
        margins: tuple[int, int] | tuple[int, int, int, int] = (10, 10),
    ) -> None:
        roi = self.extract_roi()
        if len(margins) == 2:
            mx, my = margins
            roi = [
                max(0, roi[0] - mx),
                min(self.frames.shape[2], roi[1] + mx),
                max(0, roi[2] - my),
                min(self.frames.shape[1], roi[3] + my),
            ]
        else:
            mt, mr, mb, ml = margins
            roi = [
                max(0, roi[0] - ml),
                min(self.frames.shape[2], roi[1] + mr),
                max(0, roi[2] - mt),
                min(self.frames.shape[1], roi[3] + mb),
            ]
        self.frames = self.frames[:, roi[2] : roi[3], roi[0] : roi[1]]

    def get_frame(self, index: int) -> RHEEDFrame:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        if index >= self.frames.shape[0]:
            raise IndexError("Frame index out of range")

        return RHEEDFrame(self.frames[index], self.roi, self.outdir)

    def set_outdir(self, outdir: Path | str = ".") -> None:
        self.outdir = Path(outdir).absolute()
        self.outdir.mkdir(parents=True, exist_ok=True)

    def extract_roi(self, threshold: float = 0.8) -> list[np.intp]:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        first_frame = self.frames[0]
        max_intensity = np.max(first_frame)
        mask = first_frame >= threshold * max_intensity

        y_indices, x_indices = np.where(mask)

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("No bright region detected")

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        return [x_min, x_max, y_min, y_max]

    def make_video(self, fps: int = 2) -> None:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        height, width = self.frames.shape[1:]

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(self.outdir / "video.mp4"),
            fourcc,
            fps,
            (width, height),
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
