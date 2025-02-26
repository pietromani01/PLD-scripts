from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from numpy.typing import NDArray
from skimage.feature import peak_local_max

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
        outdir: Path | str = Path("."),
    ) -> None:
        self.data = frame_data
        self.outdir = Path(outdir).absolute()

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.data.shape[:2]  # type: ignore

    def smooth(self, sigma: int = 2):
        self.data = ndi.gaussian_filter(self.data, sigma=sigma)
        return self

    def extract_peak_roi(
        self,
        min_distance: int = 15,
        width: int = 12,
        height: int = 50,
    ):
        coordinates = self._get_peak_coordinates(min_distance)
        return [
            [
                y - height // 2,
                y + height // 2,
                x - width // 2,
                x + width // 2,
            ]
            for y, x in coordinates
        ]

    def _get_peak_coordinates(self, min_distance: int = 15) -> list[tuple[int, int]]:
        coordinates = peak_local_max(self.data, min_distance=min_distance)
        coordinates = sorted(coordinates, key=lambda c: c[1])  # by x coordinate
        image_center_x = self.data.shape[1] // 2
        return sorted(coordinates, key=lambda c: abs(c[1] - image_center_x))[:3]

    def plot(
        self,
        figsize: tuple[int, int] = (8, 6),
        show_roi: bool = False,
        save: bool = False,
    ):
        _, ax = plt.subplots(figsize=figsize)
        ax.imshow(self.data, cmap=MATRIX_CMAP)
        plt.axis("off")

        if show_roi:
            self.plot_roi(ax)

        if save:
            plt.savefig(self.outdir / "frame.png")

        plt.show()

    def plot_roi(self, ax: Axes):
        roi = self.extract_peak_roi()
        colors = ["blue", "red", "lime"]
        for i, (top, bottom, left, right) in enumerate(roi):
            rect = Rectangle(
                (left, top),
                right - left,
                bottom - top,
                edgecolor=colors[i],
                facecolor="none",
                linewidth=2,
                label=f"Peak {i + 1}",
            )
            ax.text(
                right - (right - left) // 2 - 1,
                top - 2,
                f"{i + 1}",
                color=colors[i],
                fontsize=12,
            )
            ax.add_patch(rect)


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
        threshold: float = 0.8,
        margins: tuple[int, int] | tuple[int, int, int, int] = (40, 40, -20, 40),
    ) -> None:
        roi = self.extract_roi(threshold)
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

    def get_frame(self, index: int) -> RHEEDFrame:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        if index >= self.frames.shape[0]:
            raise IndexError("Frame index out of range")

        return RHEEDFrame(self.frames[index], self.outdir)

    def set_outdir(self, outdir: Path | str = ".") -> None:
        self.outdir = Path(outdir).absolute()
        self.outdir.mkdir(parents=True, exist_ok=True)

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
