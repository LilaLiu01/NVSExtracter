import cv2
import numpy as np
from scipy.stats import entropy as shannon_entropy
from typing import Optional, List

# ----------------------------
# Spatial autocorrelation (grayscale)
# ----------------------------
def spatial_autocorr(frame_gray: np.ndarray) -> float:
    """
    Compute a simple normalized spatial autocorrelation scalar (center value of autocorr map).
    frame_gray: 2D uint8 or float image.
    Returns: normalized center autocorr value (float)
    """
    if frame_gray.dtype != np.float32:
        frame = frame_gray.astype(np.float32)
    else:
        frame = frame_gray.copy()

    # subtract mean to avoid DC dominance (optional)
    frame -= np.mean(frame)

    f = np.fft.fft2(frame)
    power = np.abs(f) ** 2
    corr = np.fft.ifft2(power).real
    corr = np.fft.fftshift(corr)
    # avoid division by zero
    maxv = corr.max() if corr.max() != 0 else 1.0
    corr /= maxv
    center = corr[corr.shape[0] // 2, corr.shape[1] // 2]
    return float(center)


# ----------------------------
# Spatial autocorrelation per channel (RGB or LAB)
# ----------------------------
def spatial_autocorr_channels(frame_bgr: np.ndarray, color_space: str = "RGB") -> List[float]:
    """
    Compute spatial autocorrelation center value per channel.
    color_space: 'RGB' or 'Lab'
    Returns: list of channel autocorrs (len=3)
    """
    if color_space == "Lab":
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
    else:
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    autocorrs = []
    for c in range(frame.shape[2]):
        chan = frame[:, :, c].astype(np.float32)
        chan -= np.mean(chan)
        f = np.fft.fft2(chan)
        power = np.abs(f) ** 2
        corr = np.fft.ifft2(power).real
        corr = np.fft.fftshift(corr)
        maxv = corr.max() if corr.max() != 0 else 1.0
        corr /= maxv
        center_val = corr[corr.shape[0] // 2, corr.shape[1] // 2]
        autocorrs.append(float(center_val))
    return autocorrs


# ----------------------------
# Temporal autocorrelation (pixel-level)
# ----------------------------
def temporal_autocorr(prev_frame: Optional[np.ndarray], curr_frame: np.ndarray) -> float:
    """
    Pearson correlation between flattened previous and current frames (grayscale).
    Returns nan if prev_frame is None.
    """
    if prev_frame is None:
        return float("nan")

    # ensure same dtype and shape
    p = prev_frame
    c = curr_frame
    if p.shape != c.shape:
        # resize curr to prev shape
        c = cv2.resize(c, (p.shape[1], p.shape[0]))

    p_flat = p.flatten().astype(np.float32)
    c_flat = c.flatten().astype(np.float32)

    # handle constant images
    if np.std(p_flat) == 0 or np.std(c_flat) == 0:
        return float("nan")

    corr = np.corrcoef(p_flat, c_flat)[0, 1]
    return float(corr)


# ----------------------------
# Optical flow & motion autocorrelation
# ----------------------------
def compute_optical_flow(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Compute dense optical flow (Farneback) between two grayscale frames.
    Inputs should be single-channel images. Converts to float32 internally.
    Returns: flow HxWx2 float32 array (u,v)
    """
    # ensure same size
    if prev_gray.shape != curr_gray.shape:
        curr_gray = cv2.resize(curr_gray, (prev_gray.shape[1], prev_gray.shape[0]))

    # convert to float32 normalized [0,1]
    if prev_gray.dtype != np.float32:
        p = prev_gray.astype(np.float32) / 255.0
    else:
        p = prev_gray
    if curr_gray.dtype != np.float32:
        c = curr_gray.astype(np.float32) / 255.0
    else:
        c = curr_gray

    flow = cv2.calcOpticalFlowFarneback(
        p, c,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )
    return flow.astype(np.float32)


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """
    Compute magnitude of dense flow (HxW).
    """
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag


def motion_autocorr_from_flows(prev_flow: Optional[np.ndarray], curr_flow: np.ndarray) -> float:
    """
    Compute correlation between previous flow and current flow.
    Returns nan if prev_flow is None.
    Approach: flatten u and v components separately and average their Pearson correlations.
    """
    if prev_flow is None:
        return float("nan")

    # Ensure same shape
    if prev_flow.shape != curr_flow.shape:
        curr_flow = cv2.resize(curr_flow, (prev_flow.shape[1], prev_flow.shape[0])).reshape(prev_flow.shape)

    prev_flat_u = prev_flow[..., 0].flatten().astype(np.float32)
    curr_flat_u = curr_flow[..., 0].flatten().astype(np.float32)
    prev_flat_v = prev_flow[..., 1].flatten().astype(np.float32)
    curr_flat_v = curr_flow[..., 1].flatten().astype(np.float32)

    # safe guards for constant arrays
    corr_u = np.nan
    corr_v = np.nan
    if np.std(prev_flat_u) != 0 and np.std(curr_flat_u) != 0:
        corr_u = np.corrcoef(prev_flat_u, curr_flat_u)[0, 1]
    if np.std(prev_flat_v) != 0 and np.std(curr_flat_v) != 0:
        corr_v = np.corrcoef(prev_flat_v, curr_flat_v)[0, 1]

    # average available correlations
    corrs = [c for c in (corr_u, corr_v) if not np.isnan(c)]
    if len(corrs) == 0:
        return float("nan")
    return float(np.mean(corrs))


# A convenience wrapper for motion magnitude autocorrelation over time series:
def motion_magnitude_autocorr(prev_mag: Optional[np.ndarray], curr_mag: np.ndarray) -> float:
    """
    Correlate flattened magnitudes between two consecutive frames.
    Returns nan if prev_mag is None.
    """
    if prev_mag is None:
        return float("nan")
    p = prev_mag.flatten().astype(np.float32)
    c = curr_mag.flatten().astype(np.float32)
    if np.std(p) == 0 or np.std(c) == 0:
        return float("nan")
    return float(np.corrcoef(p, c)[0, 1])


# ----------------------------
# Frame entropy (grayscale)
# ----------------------------
def frame_entropy(frame_gray: np.ndarray, bins: int = 256) -> float:
    """
    Shannon entropy of grayscale histogram (base 2).
    """
    # ensure uint8 domain [0,255]
    if frame_gray.dtype != np.uint8:
        frame = (255.0 * (frame_gray - frame_gray.min()) / max((frame_gray.ptp(), 1e-6))).astype(np.uint8)
    else:
        frame = frame_gray
    hist, _ = np.histogram(frame, bins=bins, range=(0, 255), density=True)
    # small smoothing to avoid log(0)
    hist = hist + 1e-12
    return float(shannon_entropy(hist, base=2))


# ----------------------------
# Small helper: safe grayscale conversion
# ----------------------------
def to_gray(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to single-channel uint8 grayscale (useful for other functions).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return gray
