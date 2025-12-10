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

DIFFRACTION_BOTTOM_EDGE = 75  # pixels

LATTICE_PARAMETER = 20


class RHEEDFrame:
    """Rappresenta un singolo frame RHEED con metodi per l'analisi e la visualizzazione."""
    def __init__(
        self,
        index: int,
        data: IntArray,
        sigma: int = 0,
        params: dict | None = None,
        outdir: Path | str = Path("."),
    ) -> None:
        """Inizializza un frame RHEED con dati e parametri.
        
        Args:
            index: Numero/ID del frame
            data: Array numpy 2D con i dati dell'immagine
            sigma: Valore di smussatura gaussiana (0 = nessuno)
            params: Parametri personalizzati per l'analisi
            outdir: Cartella di output per salvare risultati
        """
        self.index = index
        normalized = data  # / np.max(data)
        self.data = (
            self._smoothen_data(normalized, sigma).astype(np.float64)
            if sigma
            else normalized
        )

        self.params = {
            "min_distance": 18,
            "threshold_rel": 0.5,
            "width": 12,
            "height": 60,
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
    def center(self) -> tuple[int, int]:
        return self.data.shape[1] // 2, self.data.shape[0] // 2

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

    def get_roi_center(self, x: int, y: int, width: int, height: int) -> tuple[int, int]:
        """Trova il centro della ROI considerando i picchi locali all'interno.
        
        Se ci sono 2 picchi vicini nella ROI, ritorna il centro tra i due.
        Altrimenti ritorna il punto originale.
        """
        # Estrai la regione candidata
        left = max(x - width // 2, 0)
        right = min(x + width // 2, self.data.shape[1])
        top = max(y - height // 2, 0)
        bottom = min(y + height // 2, DIFFRACTION_BOTTOM_EDGE)
        
        # Trova i picchi locali dentro questa regione con parametri sensibili
        roi_data = self.data[top:bottom, left:right]
        local_peaks = peak_local_max(
            roi_data,
            min_distance=2,
            threshold_rel=0.1,
        )
        
        # Converti coordinate relative a coordinate globali
        local_peaks_global = [(left + px, top + py) for py, px in local_peaks]
        
        # Se ci sono 2 o più picchi, usa il centro tra i due più alti
        if len(local_peaks_global) >= 2:
            # Prendi i due picchi più alti (maggiore intensità)
            peak_intensities = [(px, py, self.data[py, px]) for px, py in local_peaks_global]
            peak_intensities.sort(key=lambda p: p[2], reverse=True)
            
            x1, y1 = peak_intensities[0][:2]
            x2, y2 = peak_intensities[1][:2]
            
            # Ritorna il centro tra i due picchi
            return ((x1 + x2) // 2, (y1 + y2) // 2)
        
        # Altrimenti ritorna il punto originale
        return (x, y)

    def get_regions_of_interest(
        self,
        width: int | None = None,
        height: int | None = None,
        auto_center: bool = True,
    ):
        """Estrae le regioni di interesse attorno ai 3 picchi centrali.
        
        Args:
            width: Larghezza della ROI
            height: Altezza della ROI
            auto_center: Se True, centra la ROI tra 2 picchi se presenti
        """
        width = width or self.params["width"]
        height = height or self.params["height"]
        central_peaks = self.get_central_peaks()
        
        rois = []
        for x, y in sorted(central_peaks, key=lambda c: abs(c[0] - self.center[0])):
            # Se auto_center è abilitato, trova il vero centro della ROI
            if auto_center:
                x, y = self.get_roi_center(x, y, width, height)
            
            rois.append([
                max(y - height // 2, 0),
                min(y + height // 2, DIFFRACTION_BOTTOM_EDGE),
                x - width // 2,
                x + width // 2,
            ])
        
        return rois

    def get_central_peaks(self):
        center_x = self.center[0]
        peaks = [[], [], []]  # left, center, right
        for x, y in self.peaks:
            if abs(x - center_x) < LATTICE_PARAMETER:
                peaks[1].append((x, y))
            elif x < center_x - LATTICE_PARAMETER:
                peaks[0].append((x, y))
            elif x > center_x + LATTICE_PARAMETER:
                peaks[2].append((x, y))
        return [
            max(
                peaks[i],
                key=lambda p: self.data[p[1], p[0]],
                default=(0, 0),
            )  # type: ignore
            for i in range(3)
        ]

    def get_center_peak_cross_section(self):
        center_x = self.center[0]
        w = self.params["width"] // 2
        return self.data[
            :DIFFRACTION_BOTTOM_EDGE,
            center_x - w : center_x + w,
        ].sum(axis=1)

    def get_peak_ratio_sum(
        self,
        distance: int = LATTICE_PARAMETER // 2,
        prominence: float = 1.0,
    ) -> float:
        cross_section = self.get_center_peak_cross_section()
        peaks, _ = find_peaks(cross_section, distance=distance, prominence=prominence)
        peak0 = sorted(peaks, key=lambda peak: abs(peak - self.center[1]))[0]
        return sum(cross_section[peak] for peak in peaks) / cross_section[peak0] - 1

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
            # Disegna il rettangolo della ROI
            fig.add_shape(
                type="rect",
                x0=left,
                x1=right,
                y0=top,
                y1=bottom,
                line=dict(color=colors[i], width=2),
            )
            
            # Disegna il centro della ROI con un punto
            center_x = (left + right) / 2
            center_y = (top + bottom) / 2
            fig.add_trace(
                go.Scatter(
                    x=[center_x],
                    y=[center_y],
                    mode="markers",
                    marker=dict(size=8, color=colors[i], symbol="circle"),
                    showlegend=False,
                    hovertemplate=f"ROI {i+1} Center<br>x: %{{x}}<br>y: %{{y}}",
                )
            )
            
            # Etichetta della ROI
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

    def get_bg_subtracted_ROIs(
        self,
        bg_method: str = "annulus",      
        annulus_pad: int = 2,
    ) -> list[list[np.ndarray]]:
        """
        Return all ROI arrays with background subtracted (but not integrated).

        Output shape:
            list of length n_frames,
            each element is a list of n_rois arrays (each 2D).
        """
        if self.data.size == 0:
            raise ValueError("No data loaded")

        if not self.frames:
            self.generate_frames()

        all_frames = []   # list of list of ROI arrays per frame

        for frame in self.frames:
            frame_img = frame.data.astype(np.float64)
            frame_rois = []

            for (top, bottom, left, right) in frame.ROIs:
                roi = frame_img[top:bottom, left:right]
                h, w = roi.shape
                pad = min(annulus_pad, h//2, w//2)

                # ---- Background estimation ----
                if bg_method == "linear":
                    bg_plane = self.subtract_linear_background(roi, pad)
                    roi_corr = roi - bg_plane

                elif bg_method in ("annulus", "border"):
                    if pad > 0:
                        mask = np.zeros_like(roi, bool)
                        mask[:pad*4, :] = True # multiplied times four for dealing with rectangular shape
                        mask[-pad*4:, :] = True # multiplied times four for dealing with rectangular shape
                        mask[:, :pad] = True
                        mask[:, -pad:] = True
                        bg = np.median(roi[mask])
                    else:
                        bg = np.median(roi)
                    roi_corr = roi - bg

                elif bg_method == "global":
                    bg = np.median(frame_img)
                    roi_corr = roi - bg

                else:  # "none"
                    roi_corr = roi.copy()

                roi_corr[roi_corr < 0] = 0  # no negative intensities
                frame_rois.append(roi_corr)

            all_frames.append(frame_rois)

        return all_frames



    def integrate_and_normalize_ROIs(
        self,
        bg_method="annulus",
        annulus_pad=2,
        normalize="area",
        exposure_times=None,
        monitor_signal=None,
        relative_ref="first"
    ):
        # 1) Get background–subtracted ROI images
        bg_rois = self.get_bg_subtracted_ROIs(
            bg_method=bg_method,
            annulus_pad=annulus_pad
        )

        n_frames = len(bg_rois)
        n_rois = len(bg_rois[0])

        # 2) Integrate in one pass
        integrated = np.zeros((n_frames, n_rois), float)
        for i in range(n_frames):
            for j in range(n_rois):
                integrated[i, j] = bg_rois[i][j].sum()

        # 3) Normalization
        result = integrated.copy()

        if normalize == "area":
            for j, (top, bottom, left, right) in enumerate(self.frames[0].ROIs):
                area = (bottom - top) * (right - left)
                result[:, j] /= area

        elif normalize == "exposure":
            result = result / exposure_times[:, None]

        elif normalize == "monitor":
            result = result / monitor_signal[:, None]

        elif normalize == "relative":
            if relative_ref == "first":
                baseline = result[0, :]
            else:
                name, N = relative_ref
                baseline = result[:N].mean(axis=0)

            baseline[baseline == 0] = np.nan
            result = result / baseline

        return result



    def subtract_linear_background(self, roi_arr: np.ndarray, pad: int = 3) -> np.ndarray:
        """
        Estimate and subtract a linear 2D background plane using median values of ROI borders.

        Parameters
        ----------
        roi_arr : np.ndarray
            ROI image (2D array)
        pad : int
            Width in pixels of the "border" bands used to estimate background medians.

        Returns
        -------
        np.ndarray
            The ROI with the fitted background plane subtracted.
        """

        h, w = roi_arr.shape
        pad = min(pad, h // 2, w // 2)

        if pad == 0:
            # fallback: subtract global median
            return roi_arr - np.median(roi_arr)

        # --- Extract borders ---
        top = roi_arr[:pad, :]
        bottom = roi_arr[-pad:, :]
        left = roi_arr[:, :pad]
        right = roi_arr[:, -pad:]

        # --- Compute median of each border ---
        med_top = np.median(top)
        med_bottom = np.median(bottom)
        med_left = np.median(left)
        med_right = np.median(right)

        # --- Build system of equations for plane fitting ---
        # We define points at the center of each border
        pts = np.array([
            [w/2,        0,          1],   # top
            [w/2,        h,          1],   # bottom
            [0,          h/2,        1],   # left
            [w,          h/2,        1],   # right
        ])
        vals = np.array([med_top, med_bottom, med_left, med_right])

        # Solve least squares for plane: a*x + b*y + c
        coeffs, *_ = np.linalg.lstsq(pts, vals, rcond=None)
        a, b, c = coeffs

        # Build coordinate grid
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))

        # Compute fitted plane
        bg_plane = a * xx + b * yy + c

        

        return bg_plane

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
    
    def plot_integrated_ROIs(
        self,
        figsize: tuple[int, int] = (12, 6),
        annulus_pad: int = 3,
    ) -> None:
        """
        Plot intesities integrated over time for each ROI.
        """
        if self.data.size == 0:
            raise ValueError("No data loaded")

        if not self.timestamps:
            raise ValueError("No timestamps available")

        # Calcolo delle ROI integrate
        integrated = self.integrate_and_normalize_ROIs(annulus_pad=annulus_pad)
        num_ROIs = integrated.shape[1]

        # Colori generici (se più di 3 ROI si generano automaticamente)
        default_colors = ["blue", "red", "lime", "orange", "purple", "cyan"]
        colors = (default_colors * (num_ROIs // len(default_colors) + 1))[:num_ROIs]

        labels = [f"ROI {i+1}" for i in range(num_ROIs)]

        fig = go.Figure(
            layout=dict(
                width=figsize[0] * 100,
                height=figsize[1] * 100,
                title="Integrated ROI Intensities Over Time"
            )
        )

        for i in range(num_ROIs):
            fig.add_trace(
                go.Scatter(
                    x=self.timestamps,
                    y=integrated[:, i],
                    mode="lines",
                    name=labels[i],
                    line=dict(color=colors[i], width=2),
                    hovertemplate=(
                        "Frame: %{customdata}"
                        "<br>Time: %{x:.1f}s"
                        "<br>Integrated intensity: %{y}"
                    ),
                    customdata=np.arange(len(self.timestamps)),
                    hoverlabel=dict(namelength=0),
                )
            )

        fig.show()


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
