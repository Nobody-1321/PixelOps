"""
Hue wheel histogram visualization.

This module provides a polar hue histogram renderer using OpenCV.
"""

import cv2
import numpy as np

def hue_histogram_polar(
	image_bgr: np.ndarray,
	size: int = 500,
	ring_width: int | None = None,
	hist_alpha: float = 1.0,
	ring_alpha: float = 1.0,
	background_color: tuple[int, int, int] = (115, 115, 115),
	histogram_color: tuple[int, int, int] = (115, 115, 115),
	sat_threshold: int = 5,
) -> np.ndarray:
	"""
	Build a polar hue histogram with a color-wheel ring background.

	Parameters
	----------
	image_bgr : np.ndarray
		Input image in BGR format (uint8, shape HxWx3).
	size : int, default=500
		Output square canvas size in pixels.
	ring_width : int | None, default=None
		Width of the outer color-wheel ring. If None, an automatic width
		based on size is used.
	hist_alpha : float, default=1.0
		Opacity used when blending the hue-colored histogram base.
	ring_alpha : float, default=1.0
		Opacity used for the outer color-wheel ring.
	background_color : tuple[int, int, int], default=(115, 115, 115)
		BGR background color for the canvas.
	histogram_color : tuple[int, int, int], default=(115, 115, 115)
		BGR color used to draw the inward histogram fill.
	sat_threshold : int, default=5
		Saturation threshold used to ignore near-gray pixels.

	Returns
	-------
	np.ndarray
		Rendered BGR canvas with the polar hue histogram.
	"""
	if not isinstance(image_bgr, np.ndarray):
		raise TypeError("image_bgr must be a numpy array.")
	if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
		raise ValueError("image_bgr must have shape (H, W, 3) in BGR format.")
	if image_bgr.dtype != np.uint8:
		raise ValueError("image_bgr must be uint8.")
	if size < 32:
		raise ValueError("size must be >= 32.")

	hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
	hue = hsv[:, :, 0]
	sat = hsv[:, :, 1]

	# Keep only sufficiently saturated pixels for a cleaner hue histogram.
	mask = sat > sat_threshold
	hue_filtered = hue[mask]

	if hue_filtered.size == 0:
		hist = np.zeros(180, dtype=np.float32)
	else:
		hist = cv2.calcHist([hue_filtered], [0], None, [180], [0, 180]).flatten()

	hist = cv2.GaussianBlur(hist.reshape(-1, 1), (1, 9), 2).flatten()
	hist = hist / (hist.max() + 1e-6)

	canvas = np.full((size, size, 3), background_color, dtype=np.uint8)
	overlay_ring = np.zeros_like(canvas)
	overlay_hist_base = canvas.copy()

	center = (size // 2, size // 2)
	max_radius = size // 2 - 10

	if ring_width is None:
		ring_width = max(30, int(size * 0.12))
	ring_width = int(np.clip(ring_width, 1, max_radius - 1))
	hist_outer_radius = max_radius - ring_width

	for i in range(180):
		angle1 = (i / 180.0) * 360
		angle2 = ((i + 1) / 180.0) * 360

		color_hsv = np.uint8([[[i, 255, 255]]])
		color_bgr = tuple(int(x) for x in cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0, 0])

		cv2.ellipse(
			overlay_ring,
			center,
			(max_radius, max_radius),
			0,
			angle1,
			angle2,
			color_bgr,
			-1,
		)

		cv2.ellipse(
			overlay_hist_base,
			center,
			(hist_outer_radius, hist_outer_radius),
			0,
			angle1,
			angle2,
			color_bgr,
			-1,
		)

	cv2.circle(overlay_ring, center, hist_outer_radius, (0, 0, 0), -1)

	hist_alpha = float(np.clip(hist_alpha, 0.0, 1.0))
	canvas = cv2.addWeighted(canvas, 1.0 - hist_alpha, overlay_hist_base, hist_alpha, 0)

	ring_alpha = float(np.clip(ring_alpha, 0.0, 1.0))
	ring_mask = np.any(overlay_ring > 0, axis=2)
	if ring_alpha >= 0.999:
		canvas[ring_mask] = overlay_ring[ring_mask]
	elif ring_alpha > 0.0:
		ring_blend = cv2.addWeighted(canvas, 1.0 - ring_alpha, overlay_ring, ring_alpha, 0)
		canvas[ring_mask] = ring_blend[ring_mask]

	for i in range(180):
		angle1 = (i / 180.0) * 360
		angle2 = ((i + 1) / 180.0) * 360

		value = hist[i]
		inner_radius = int(hist_outer_radius * (1 - value))

		cv2.ellipse(
			canvas,
			center,
			(inner_radius, inner_radius),
			0,
			angle1,
			angle2,
			histogram_color,
			-1,
		)

	cv2.circle(canvas, center, max_radius, (255, 255, 255), 1)

	return canvas
