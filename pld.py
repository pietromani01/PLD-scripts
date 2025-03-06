from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import plotly.graph_objects as go
import scipy.ndimage as ndi
from numpy.typing import NDArray
from scipy.fftpack import fft2, fftshift
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

MATRIX_COLORSCALE = [
    (0.00, "#000000"),  # Black
    (0.60, "#0C2000"),  # Very dark green, almost black
    (0.75, "#184000"),  # Deep green
    (0.90, "#247F00"),  # Brighter green
    (1.00, "#32FF00"),  # Neon lime green
]

DIFFRACTION_BOTTOM_EDGE = 80  # pixels


class RHEEDFrame:
    def __init__(
        self,
        index: int,
        data: IntArray,
        sigma: int = 0,
        params: dict | None = None,
        outdir: Path | str = Path("."),
    ) -> None:
        self.index = index
        normalized = data / np.max(data)
        self.data = (
            self._smoothen_data(normalized, sigma).astype(np.float64)
            if sigma
            else normalized
        )

        self.params = {
            "min_distance": 18,
            "threshold_rel": 0.5,
            "width": 12,
            "height": 50,
        } | (params or {})

        self.outdir = Path(outdir).absolute()

        self._peaks: list[tuple[int, int]] = []
        self._ROIs: list[list[int]] = []
        self._peak_intensities: list[np.int64] = []
        self._sharpness: float | None = None
        self._power_spectrum: IntArray = None
        self._radial_profile: FloatArray = None
        self._mode: int = 0

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.data.shape[:2]  # type: ignore

    @property
    def peaks(self) -> list[tuple[int, int]]:
        if not self._peaks:
            self._peaks = self.get_peak_coordinates()
        return self._peaks

    @property
    def ROIs(self) -> list[list[int]]:
        if not self._ROIs:
            self._ROIs = self.get_regions_of_interest()
        return self._ROIs

    @property
    def peak_intensities(self) -> list[np.int64]:
        if not self._peak_intensities:
            self._peak_intensities = self.get_peak_intensities()
        return self._peak_intensities

    @property
    def sharpness(self) -> float:
        if self._sharpness is None:
            self._sharpness = self.get_laplacian_variance()
        return self._sharpness

    @property
    def power_spectrum(self) -> IntArray:
        if self._power_spectrum is None:
            self._power_spectrum = self.get_power_spectrum()
        return self._power_spectrum

    @property
    def radial_profile(self) -> FloatArray:
        if self._radial_profile is None:
            self._radial_profile = self.get_radial_profile()
        return self._radial_profile

    def smooth(self, sigma: int = 2) -> "RHEEDFrame":
        return RHEEDFrame(
            index=self.index,
            data=self._smoothen_data(self.data, sigma),
            outdir=self.outdir,
        )

    def _smoothen_data(self, data: IntArray, sigma: int):
        return ndi.gaussian_filter(data, sigma=sigma)

    def normalize(self) -> "RHEEDFrame":
        normalized = cv2.normalize(
            self.data,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
        ).astype(np.uint8)  # type: ignore
        return RHEEDFrame(
            index=self.index,
            data=normalized,
            outdir=self.outdir,
        )

    def get_peak_coordinates(
        self,
        min_distance: int | None = None,
        threshold_rel: float = 0.0,
    ) -> list[tuple[int, int]]:
        min_distance = min_distance or self.params["min_distance"]
        threshold_rel = threshold_rel or self.params["threshold_rel"]
        coordinates = peak_local_max(
            self.data,
            min_distance=min_distance,
            threshold_rel=threshold_rel,
        )
        coordinates = sorted(coordinates, key=lambda c: c[1])  # by x coordinate
        return [(x, y) for y, x in coordinates]

    # TODO using fix width/height - do we require more flexibility?
    def get_regions_of_interest(
        self,
        width: int | None = None,
        height: int | None = None,
    ):
        width = width or self.params["width"]
        height = height or self.params["height"]
        image_center_x = self.data.shape[1] // 2
        central_peaks = sorted(self.peaks, key=lambda c: abs(c[0] - image_center_x))[:3]
        return [
            [
                max(y - height // 2, 0),
                min(y + height // 2, DIFFRACTION_BOTTOM_EDGE),
                x - width // 2,
                x + width // 2,
            ]
            for x, y in central_peaks
        ]

    def get_peak_intensities(self):
        return [
            np.max(self.data[top:bottom, left:right])
            for top, bottom, left, right in self.ROIs
        ]

    def get_laplacian_variance(self) -> float:
        laplacian = cv2.Laplacian(self.data, cv2.CV_64F)
        return laplacian.var()

    def get_power_spectrum(self) -> IntArray:
        fft_result = fft2(self.data)
        fft_shifted = fftshift(fft_result)
        return np.abs(fft_shifted) ** 2

    def get_radial_profile(self) -> FloatArray:
        h, w = self.power_spectrum.shape
        y, x = np.indices((h, w))
        center = (h // 2, w // 2)
        r = np.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
        r: np.int64 = r.astype(np.int64)
        radial_mean = np.bincount(r.ravel(), weights=self.power_spectrum.ravel())
        radial_count = np.bincount(r.ravel())
        radial_profile = radial_mean / (radial_count + 1e-8)  # Avoid division by zero
        return radial_profile

    def plot_power_spectrum(
        self,
        colorscale: list[tuple[float, str]] | str = "cividis",
        figsize: tuple[int] = (4, 3),
    ):
        fig = go.Figure(
            data=go.Heatmap(
                z=np.log1p(self.power_spectrum),
                colorscale=colorscale,
                hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z:.2f}",
                hoverlabel=dict(namelength=0),
            ),
            layout=dict(width=figsize[0] * 100, height=figsize[1] * 100),
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            coloraxis_colorbar=dict(title="Log Power"),
            margin=dict(l=0, r=0, t=0, b=0),
        )
        fig.show(config={"displayModeBar": False})

    def plot(
        self,
        figsize: tuple[int, int] = (8, 6),
        colorscale: list[tuple[float, str]] | str = MATRIX_COLORSCALE,
        show_peaks: bool = False,
        show_regions_of_interest: bool = False,
        save: bool = False,
    ):
        fig = go.Figure(layout=dict(width=figsize[0] * 100, height=figsize[1] * 100))

        fig.add_trace(
            go.Heatmap(
                z=self.data,
                colorscale=colorscale,
                showscale=False,
                hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z:.2f}",
                hoverlabel=dict(namelength=0),
            )
        )

        if show_peaks:
            fig.add_trace(
                go.Scatter(
                    x=[x for x, _ in self.peaks],
                    y=[y for _, y in self.peaks],
                    mode="markers",
                    marker=dict(size=10, color="red", symbol="x"),
                    hovertemplate="x: %{x}<br>y: %{y}<br>z: %{customdata:.2f}",
                    customdata=[self.data[y, x] for x, y in self.peaks],
                    hoverlabel=dict(namelength=0),
                    showlegend=False,
                )
            )

        if show_regions_of_interest:
            self._plot_regions_of_interest(fig)

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

    def _plot_regions_of_interest(self, fig: go.Figure):
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
    data: IntArray = np.array([])
    frames: list[RHEEDFrame] = []
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
            self.data = np.load(data_path)
            self.timestamps.append(self.extract_timestamp(data_path))
            if self.data.ndim == 2:
                self.data = self.data[np.newaxis, ...]
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
        self.data = np.stack(frames, axis=0)

    def extract_timestamp(self, data_path: Path) -> float:
        return datetime.strptime(
            data_path.name.split("_frame")[0],
            r"%Y%m%d_%H%M%S%f",
        ).timestamp()

    def extract_global_region_of_interest(
        self,
        threshold: float = 0.8,
    ) -> list[np.intp]:
        if self.data.size == 0:
            raise ValueError("No data loaded")

        first_frame = self.data[0]
        max_intensity = np.max(first_frame)
        mask = first_frame >= threshold * max_intensity

        y_indices, x_indices = np.where(mask)

        if len(y_indices) == 0 or len(x_indices) == 0:
            raise ValueError("No bright region detected")

        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        return [x_min, x_max, y_min, y_max]

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
                min(self.data.shape[2], ROI[1] + mx),
                max(0, ROI[2] - my),
                min(self.data.shape[1], ROI[3] + my),
            ]
        else:
            mt, mr, mb, ml = margins
            ROI = [
                max(0, ROI[0] - ml),
                min(self.data.shape[2], ROI[1] + mr),
                max(0, ROI[2] - mt),
                min(self.data.shape[1], ROI[3] + mb),
            ]
        self.data = self.data[:, ROI[2] : ROI[3], ROI[0] : ROI[1]]

    def generate_frames(self, sigma: int = 0, params: dict | None = None) -> None:
        self.frames = [
            RHEEDFrame(
                index=i,
                data=data,
                sigma=sigma,
                params=params,
                outdir=self.outdir,
            )
            for i, data in enumerate(self.data)
        ]

    def set_outdir(self, outdir: Path | str = ".") -> None:
        self.outdir = Path(outdir).absolute()
        self.outdir.mkdir(parents=True, exist_ok=True)

    def make_video(self, fps: int = 2) -> None:
        if self.data.size == 0:
            raise ValueError("No data loaded")

        height, width = self.data.shape[1:]

        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(self.outdir / "video.mp4"),
            fourcc,
            fps,
            (width, height),
        )

        for frame in self.frames:
            frame = cv2.cvtColor(frame.normalize().data, cv2.COLOR_GRAY2BGR)  # type: ignore
            video_writer.write(frame)

        video_writer.release()

    def get_peak_intensities(self, sigma: int = 0) -> list[list[np.float64]]:
        peak_intensities = [[] for _ in range(3)]
        for frame in self.frames:
            rheed_frame = frame.smooth(sigma) if sigma else frame
            intensities = rheed_frame.peak_intensities
            for j in range(3):
                intensity = intensities[j] if j < len(intensities) else None
                peak_intensities[j].append(intensity)
        return peak_intensities

    def get_sharpness(self) -> list[float]:
        return [frame.sharpness for frame in self.frames]

    def get_radial_profile(self):
        return [frame.radial_profile.max() for frame in self.frames]

    def plot_sharpness_time_series(
        self,
        data: list[float],
        figsize: tuple[int, int] = (12, 6),
    ):
        if self.data.size == 0:
            raise ValueError("No data loaded")

        if not self.timestamps:
            raise ValueError("No timestamps available")

        fig = go.Figure(layout=dict(width=figsize[0] * 100, height=figsize[1] * 100))

        maximum_variance = np.max(data)
        fig.add_trace(
            go.Scatter(
                x=self.timestamps,
                y=data / maximum_variance,
                mode="lines",
                name="Sharpness",
                hovertemplate="Frame: %{customdata}<br>Time: %{x:.1f}s<br>sharpness: %{y:.2f}",
                customdata=np.arange(len(self.timestamps)),
                hoverlabel=dict(namelength=0),
            )
        )

        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Sharpness",
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
            showlegend=True,
            legend=dict(
                yanchor="top",
                xanchor="right",
                x=1,
            ),
        )

        fig.show(config={"displayModeBar": False})

    def plot_radial_profile_time_series(
        self,
        data: list,
        figsize: tuple[int] = (12, 6),
    ):
        fig = go.Figure(layout=dict(width=figsize[0] * 100, height=figsize[1] * 100))

        maximum = np.max(data)
        fig.add_trace(
            go.Scatter(
                x=self.timestamps,
                y=data / maximum,
                mode="lines",
                name="Radial profile",
                hovertemplate="Frame: %{customdata}<br>Time: %{x:.1f}s<br>Radial profile: %{y:.2f}",
                customdata=np.arange(len(self.timestamps)),
                hoverlabel=dict(namelength=0),
            )
        )
        fig.update_layout(
            xaxis_title="Time [s]",
            yaxis_title="Peak max intensity",
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
            showlegend=True,
            legend=dict(
                yanchor="top",
                xanchor="right",
                x=1,
            ),
        )
        fig.show(config={"displayModeBar": False})

    def plot_intensity_time_series(
        self,
        data: list[list[np.float64]],
        figsize: tuple[int, int] = (12, 6),
    ) -> None:
        if self.data.size == 0:
            raise ValueError("No data loaded")

        if not self.timestamps:
            raise ValueError("No timestamps available")

        fig = go.Figure(layout=dict(width=figsize[0] * 100, height=figsize[1] * 100))

        colors = ["blue", "red", "lime"]
        labels = [f"Peak {i + 1}" for i in range(3)]

        for i in range(3):
            fig.add_trace(
                go.Scatter(
                    x=self.timestamps,
                    y=data[i],
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
            yaxis_title="Peak max intensity",
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
            showlegend=True,
            legend=dict(
                yanchor="top",
                xanchor="right",
                x=1,
            ),
        )

        fig.show(config={"displayModeBar": False})

    def compute_deposition_score(
        self,
        start_time: int = 100,
        end_time: int = 300,
    ) -> np.float64 | None:
        scores = []
        start = int(start_time * 2)
        end = int(end_time * 2)
        for frame in self.frames[start:end]:
            if frame.sharpness > 0.8:
                score = 1.0  # Spots
            elif frame.radial_profile[1] > 0.6:
                score = 0.83  # Streaks
            elif frame.radial_profile[2] > 0.5:
                score = 0.67  # Satellite Streaks
            elif frame.radial_profile[3] > 0.4:
                score = 0.5  # Modulated Streaks
            elif frame.radial_profile[4] > 0.3:
                score = 0.33  # Inclined Streaks
            else:
                score = 0.0  # Transmission Spots

            scores.append(score)

        return np.mean(scores) if scores else None

    def compute_decay_rate(self, intensities: IntArray) -> float:
        def exp_decay(t, I0, lambda_):
            return I0 * np.exp(-lambda_ * t)

        try:
            popt, _ = curve_fit(  # type: ignore
                exp_decay, self.timestamps, intensities, p0=(intensities[0], 0.01)
            )
            return popt[1]  # lambda_
        except RuntimeError:
            print("Curve fitting failed; returning NaN")
            return np.nan

    def compute_oscillation_score(self, intensities: IntArray) -> float:
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

    def analyze_quality(self, ROI_index: int) -> dict:
        intensities = [frame.peak_intensities[ROI_index] for frame in self.frames]
        intensities = np.array(intensities)
        return {
            "decay_rate": self.compute_decay_rate(intensities),
            "oscillation_score": self.compute_oscillation_score(intensities),
        }
