from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import scipy.ndimage as ndi
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

FrameArray = NDArray[np.int64]

MATRIX_COLORSCALE = [
    (0.00, "#000000"),  # Black
    (0.60, "#0C2000"),  # Very dark green, almost black
    (0.75, "#184000"),  # Deep green
    (0.90, "#247F00"),  # Brighter green
    (1.00, "#32FF00"),  # Neon lime green
]


class RHEEDFrame:
    def __init__(
        self,
        frame_index: int,
        frame_data: FrameArray,
        outdir: Path | str = Path("."),
    ) -> None:
        self.index = frame_index
        self.data = frame_data
        self.outdir = Path(outdir).absolute()
        self.ROIs = []

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.data.shape[:2]  # type: ignore

    def smooth(self, sigma: int = 2):
        data = ndi.gaussian_filter(self.data, sigma=sigma)
        return RHEEDFrame(self.index, data, self.outdir)

    # TODO using fix width/height - do we require more flexibility?
    def define_regions_of_interest(
        self,
        min_distance: int = 18,
        width: int = 12,
        height: int = 50,
    ):
        coordinates = self.get_peak_coordinates(min_distance)
        self.ROIs = [
            [
                y - height // 2,
                y + height // 2,
                x - width // 2,
                x + width // 2,
            ]
            for y, x in coordinates
        ]

    def get_peak_coordinates(self, min_distance: int = 15) -> list[tuple[int, int]]:
        coordinates = peak_local_max(self.data, min_distance=min_distance)
        coordinates = sorted(coordinates, key=lambda c: c[1])  # by x coordinate
        image_center_x = self.data.shape[1] // 2
        return sorted(coordinates, key=lambda c: abs(c[1] - image_center_x))[:3]

    def get_peak_intensities(self) -> list[np.int64]:
        if not self.ROIs:
            print(self.index)
            raise ValueError("No regions of interest (ROIs) defined")
        return [
            np.max(self.data[top:bottom, left:right])
            for top, bottom, left, right in self.ROIs
        ]

    def plot(
        self,
        figsize: tuple[int, int] = (8, 6),
        colorscale: list[tuple[float, str]] | str = MATRIX_COLORSCALE,
        show_regions_of_interest: bool = False,
        save: bool = False,
    ):
        fig = go.Figure(layout=dict(width=figsize[0] * 100, height=figsize[1] * 100))

        # Add heatmap of data
        fig.add_trace(
            go.Heatmap(
                z=self.data,
                colorscale=colorscale,
                showscale=False,
            )
        )

        # Add Regions of Interest (ROIs) if requested
        if show_regions_of_interest:
            self.plot_regions_of_interest(fig)

        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False, scaleanchor="x", autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )

        # Save as interactive HTML if needed
        if save:
            fig.write_html(str(self.outdir / "frame.html"))

        fig.show(config={"displayModeBar": False})

    def plot_regions_of_interest(self, fig: go.Figure):
        if not self.ROIs:
            raise ValueError("No regions of interest (ROIs) defined")
        colors = ["blue", "red", "lime"]
        for i, (top, bottom, left, right) in enumerate(self.ROIs):
            fig.add_shape(
                type="rect",
                x0=left,
                x1=right,
                y0=top,
                y1=bottom,
                line=dict(color=colors[i], width=2),
            )
            fig.add_annotation(
                x=(left + right) / 2,
                y=top - 2,
                text=f"{i + 1}",
                showarrow=False,
                font=dict(size=12, color=colors[i]),
            )


class RHEEDAnalyzer:
    frames: FrameArray = np.array([])
    timestamps: list[datetime] = []
    outdir = Path(".")

    def load_data(self, data_path: Path | str) -> None:
        data_path = Path(data_path).absolute()
        if not data_path.exists():
            raise FileNotFoundError(f"File not found: {data_path}")
        if data_path.is_file():
            self.load_single_data_file(data_path)
        elif data_path.is_dir():
            self.load_multiple_data_files(data_path)
        else:
            raise ValueError("Invalid path")

    def load_single_data_file(self, data_path: Path):
        if data_path.suffix == ".npy":
            self.frames = np.load(data_path)
            self.timestamps.append(self.extract_timestamp(data_path))
            if self.frames.ndim == 2:
                self.frames = self.frames[np.newaxis, ...]
        else:
            raise ValueError("Unsupported file format")

    def load_multiple_data_files(self, data_path: Path):
        data_files = sorted(data_path.glob("*.npy"))
        if not data_files:
            raise FileNotFoundError(f"No supported data files found in {data_path}")
        frames = []
        timestamps = []
        timestamp_0 = self.extract_timestamp(data_files[0])
        for data_file in data_files:
            frames.append(np.load(data_file))
            timestamps.append(self.extract_timestamp(data_file) - timestamp_0)
        self.timestamps = timestamps
        self.frames = np.stack(frames, axis=0)

    def extract_timestamp(self, data_path: Path) -> float:
        return datetime.strptime(
            data_path.name.split("_frame")[0],
            r"%Y%m%d_%H%M%S%f",
        ).timestamp()

    def crop_to_global_region_of_interest(
        self,
        threshold: float = 0.8,
        margins: tuple[int, int] | tuple[int, int, int, int] = (40, 40, -20, 40),
    ) -> None:
        ROI = self.extract_global_region_of_interest(threshold)
        if len(margins) == 2:
            mx, my = margins
            ROI = [
                max(0, ROI[0] - mx),
                min(self.frames.shape[2], ROI[1] + mx),
                max(0, ROI[2] - my),
                min(self.frames.shape[1], ROI[3] + my),
            ]
        else:
            mt, mr, mb, ml = margins
            ROI = [
                max(0, ROI[0] - ml),
                min(self.frames.shape[2], ROI[1] + mr),
                max(0, ROI[2] - mt),
                min(self.frames.shape[1], ROI[3] + mb),
            ]
        self.frames = self.frames[:, ROI[2] : ROI[3], ROI[0] : ROI[1]]

    def extract_global_region_of_interest(
        self,
        threshold: float = 0.8,
    ) -> list[np.intp]:
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

        return RHEEDFrame(index, self.frames[index], self.outdir)

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

    def plot_intensity_time_series(
        self,
        sigma: int = 0,
        min_distance: int = 18,
        figsize: tuple[int, int] = (12, 6),
        save: bool = False,
    ) -> None:
        if self.frames.size == 0:
            raise ValueError("No data loaded")

        if not self.timestamps:
            raise ValueError("No timestamps available")

        peak_intensities = [[] for _ in range(3)]

        for i, frame in enumerate(self.frames):
            rheed_frame = RHEEDFrame(i, frame)
            if sigma:
                rheed_frame = rheed_frame.smooth(sigma)
            rheed_frame.define_regions_of_interest(min_distance)
            intensities = rheed_frame.get_peak_intensities()

            for j in range(3):
                intensity = intensities[j] if j < len(intensities) else None
                peak_intensities[j].append(intensity)

        fig = go.Figure(layout=dict(width=figsize[0] * 100, height=figsize[1] * 100))

        colors = ["blue", "red", "lime"]
        labels = [f"Peak {i + 1}" for i in range(3)]

        maximum_intensity = np.max(peak_intensities)
        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=self.timestamps,
                    y=peak_intensities[i] / maximum_intensity,
                    mode="lines",
                    name=labels[i],
                    line=dict(color=colors[i], width=2),
                    hovertemplate="Frame: %{customdata}<br>Time: %{x:.1f}s<br>Intensity: %{y:.2f}",
                    customdata=np.arange(len(self.timestamps)),
                    hoverlabel=dict(namelength=0),
                )
            )

        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Peak Max Intensity [a.u.]",
            xaxis=dict(
                showgrid=True,
                showline=True,
                linecolor="black",
            ),
            yaxis=dict(
                showgrid=True,
                showline=True,
                linecolor="black",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=20, b=20),
            template="plotly_white",
        )

        # Save as interactive HTML if needed
        if save:
            fig.write_html(str(self.outdir / "intensity_timeseries.html"))

        fig.show(config={"displayModeBar": False})

    def compute_decay_rate(self, intensities: NDArray) -> float:
        def exp_decay(t, I0, lambda_):
            return I0 * np.exp(-lambda_ * t)

        try:
            popt, _ = curve_fit(
                exp_decay, self.timestamps, intensities, p0=(intensities[0], 0.01)
            )
            return popt[1]  # lambda_
        except RuntimeError:
            print("Curve fitting failed; returning NaN")
            return np.nan

    def compute_oscillation_score(self, intensities: NDArray) -> float:
        # Find peaks and valleys
        peaks, _ = find_peaks(intensities)
        valleys, _ = find_peaks(-intensities)

        if len(peaks) == 0 or len(valleys) == 0:
            return 0  # No oscillations detected

        # Compute peak-to-valley differences
        peak_vals = intensities[peaks]
        valley_vals = intensities[valleys]
        min_len = min(len(peak_vals), len(valley_vals))
        peak_to_valley_diffs = np.abs(peak_vals[:min_len] - valley_vals[:min_len])

        # Oscillation score: sum of differences weighted by count
        return np.sum(peak_to_valley_diffs) * len(peaks)

    def get_all_frames(self) -> list[RHEEDFrame]:
        frames = []
        for i, data in enumerate(self.frames):
            frame = RHEEDFrame(i, data, self.outdir)
            frame.define_regions_of_interest()
            frames.append(frame)
        return frames

    def analyze_quality(self, ROI_index: int) -> dict:
        intensities = np.array(
            [frame.get_peak_intensities()[ROI_index] for frame in self.get_all_frames()]
        )
        return {
            "decay_rate": self.compute_decay_rate(intensities),
            "oscillation_score": self.compute_oscillation_score(intensities),
        }
