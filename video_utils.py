import cv2

def stream_video(video_path, max_frames=None, sample_rate=1):
    """
    Stream frames from a video file efficiently.

    Args:
        video_path (str): Path to the video file.
        max_frames (int, optional): Maximum number of frames to yield.
        sample_rate (int): Yield every Nth frame (1 = all frames).

    Yields:
        np.ndarray: Video frames (BGR format).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_idx >= max_frames:
            break
        if frame_idx % sample_rate == 0:
            yield frame
        frame_idx += 1

    cap.release()