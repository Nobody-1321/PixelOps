import cv2
import numpy as np

def white_balance_custom(
    img,
    strength=1.0,        # 0 = sin efecto, 1 = completo
    clip_percent=0.0,    # 0 = sin recorte, ej: 5 para ignorar extremos
    bias=(1.0, 1.0, 1.0) # ajuste manual (B, G, R)
):
    img = img.astype(np.float32)

    result = np.zeros_like(img)

    for c in range(3):
        channel = img[:, :, c]

        # --- 1. Opcional: recorte de outliers ---
        if clip_percent > 0:
            low = np.percentile(channel, clip_percent)
            high = np.percentile(channel, 100 - clip_percent)
            channel = np.clip(channel, low, high)

        # --- 2. Promedio del canal ---
        avg = np.mean(channel)

        # --- 3. Objetivo gris ---
        target = np.mean(img)

        # --- 4. Factor de corrección ---
        scale = target / (avg + 1e-6)

        # --- 5. Mezcla con original (control de intensidad) ---
        scale = 1.0 + (scale - 1.0) * strength

        # --- 6. Aplicar bias manual ---
        scale *= bias[c]

        result[:, :, c] = img[:, :, c] * scale

    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


# Uso ejemplo
input_path = "./data/img/studio_logoNO.webp"
img = cv2.imread(input_path)

balanced = white_balance_custom(
    img,
    strength=0.1,
    clip_percent=1,
    bias=(1.0, 1.0, 1.1)  # un poco más cálido (más rojo)
)

cv2.imwrite("output.jpg", balanced)
