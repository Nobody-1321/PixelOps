import cv2
import numpy as np
from numba import njit

# =====================================================================
# ETAPA 1: CONSTRUCCIÓN DEL EDGE TANGENT FLOW (ETF)
# =====================================================================

@njit(fastmath=True)
def update_etf_kernel(etf, grad_mag, r):
    """
    Optimización nativa con Numba para el suavizado no lineal del campo ETF (Eq. 1).
    """
    h, w, _ = etf.shape
    new_etf = np.zeros_like(etf)
    
    for y in range(h):
        for x in range(w):
            t_cur = etf[y, x]
            g_cur = grad_mag[y, x]
            
            # Acumuladores para el promedio ponderado
            t_accum = np.zeros(2, dtype=np.float32)
            
            # Límites del vecindario de radio r
            y_min = max(0, y - r)
            y_max = min(h - 1, y + r)
            x_min = max(0, x - r)
            x_max = min(w - 1, x + r)
            
            for ny in range(y_min, y_max + 1):
                for nx in range(x_min, x_max + 1):
                    t_nbr = etf[ny, nx]
                    g_nbr = grad_mag[ny, nx]
                    
                    # 1. Peso de Magnitud w_m (Eq. 3) -> eta = 1
                    # tanh(g_nbr - g_cur) escalado al rango [0, 1]
                    w_m = 0.5 * (1.0 + np.tanh(g_nbr - g_cur))
                    
                    # 2. Peso de Dirección w_d (Eq. 4)
                    dot_product = t_cur[0] * t_nbr[0] + t_cur[1] * t_nbr[1]
                    w_d = abs(dot_product)
                    
                    # 3. Función de Signo phi (Eq. 5)
                    phi = 1.0 if dot_product >= 0.0 else -1.0
                    
                    # Peso total (w_s es filtro de caja constante = 1 denotado por el radio)
                    weight = w_m * w_d
                    
                    t_accum[0] += phi * t_nbr[0] * weight
                    t_accum[1] += phi * t_nbr[1] * weight
            
            # Normalizar el nuevo vector
            norm = np.sqrt(t_accum[0]**2 + t_accum[1]**2)
            if norm > 0.0:
                new_etf[y, x] = t_accum / norm
            else:
                new_etf[y, x] = t_cur
                
    return new_etf

def compute_etf(img_gray, r=5, iterations=3):
    """
    Inicializa y refina el campo de vectores tangentes de borde (ETF).
    """
    # 1. Calcular gradientes espaciales con Sobel
    sobel_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    
    # Magnitud del gradiente normalizada en el rango [0, 1]
    grad_mag = cv2.magnitude(sobel_x, sobel_y)
    max_mag = grad_mag.max()
    if max_mag > 0:
        grad_mag /= max_mag
        
    # 2. Vectores tangentes iniciales (perpendiculares al gradiente en sentido antihorario)
    h, w = img_gray.shape
    etf = np.zeros((h, w, 2), dtype=np.float32)
    etf[:, :, 0] = -sobel_y
    etf[:, :, 1] = sobel_x
    
    # Normalizar campo inicial
    norms = np.sqrt(etf[:, :, 0]**2 + etf[:, :, 1]**2)
    mask = norms > 0
    etf[mask] /= norms[mask][:, np.newaxis]
    
    # 3. Refinar iterativamente el campo usando el kernel no lineal
    for _ in range(iterations):
        etf = update_etf_kernel(etf, grad_mag, r)
        
    return etf

# =====================================================================
# ETAPA 2: FILTRADO ANISOTRÓPICO FDoG (FLOW-BASED DIFFERENCE OF GAUSSIANS)
# =====================================================================

@njit(fastmath=True)
def fdog_kernel(img_gray, etf, sigma_c, sigma_s, sigma_m, rho):
    """
    Aplica el filtro FDoG integrando numéricamente sobre las curvas del flujo (Eq. 6 y 9).
    """
    h, w = img_gray.shape
    H = np.zeros((h, w), dtype=np.float32)
    
    # Definición de las ventanas de integración basadas en las sigmas (3 * sigma)
    S_len = int(ceil(3.0 * sigma_m))
    T_len = int(ceil(3.0 * sigma_s))
    
    # Precomputar pesos gaussianos unidimensionales
    gauss_c = np.zeros(T_len + 1, dtype=np.float32)
    gauss_s = np.zeros(T_len + 1, dtype=np.float32)
    gauss_m = np.zeros(S_len + 1, dtype=np.float32)
    
    for t in range(T_len + 1):
        gauss_c[t] = exp(-0.5 * (t / sigma_c)**2) / (sqrt(2.0 * np.pi) * sigma_c)
        gauss_s[t] = exp(-0.5 * (t / sigma_s)**2) / (sqrt(2.0 * np.pi) * sigma_s)
    for s in range(S_len + 1):
        gauss_m[s] = exp(-0.5 * (s / sigma_m)**2) / (sqrt(2.0 * np.pi) * sigma_m)

    for y in range(h):
        for x in range(w):
            sum_h = 0.0
            w_h_sum = 0.0
            
            # --- Integración Longitudinal (Paso s a lo largo del flujo ETF) ---
            for s in range(-S_len, S_len + 1):
                # Desplazamiento iterativo de Euler bidireccional
                cx, cy = float(x), float(y)
                step = float(s)
                
                # Avanzar de manera adaptativa a lo largo de la curva de flujo
                # En Numba implementamos pasos discretos para encontrar c_x(s)
                remaining = abs(step)
                while remaining > 0.0:
                    ix, iy = int(round(cx)), int(round(cy))
                    if ix < 0 or ix >= w or iy < 0 or iy >= h:
                        break
                    t_vec = etf[iy, ix]
                    direction = 1.0 if step >= 0 else -1.0
                    cx += direction * t_vec[0]
                    cy += direction * t_vec[1]
                    remaining -= 1.0
                    
                ix, iy = int(round(cx)), int(round(cy))
                if ix < 0 or ix >= w or iy < 0 or iy >= h:
                    continue
                
                # Tangente en la posición de la curva c_x(s)
                t_local = etf[iy, ix]
                # Dirección perpendicular (Gradiente local)
                g_local = np.array([-t_local[1], t_local[0]], dtype=np.float32)
                
                # --- Integración Transversal (Paso t perpendicular al flujo) ---
                sum_f = 0.0
                for t in range(-T_len, T_len + 1):
                    lx = cx + t * g_local[0]
                    ly = cy + t * g_local[1]
                    
                    ilx, ily = int(round(lx)), int(round(ly))
                    if ilx < 0 or ilx >= w or ily < 0 or ily >= h:
                        val = 255.0  # Asumir fondo blanco si sale de la imagen
                    else:
                        val = float(img_gray[ily, ilx])
                        
                    # Aplicar núcleo DoG 1D (Eq. 7)
                    w_c = gauss_c[abs(t)]
                    w_s = gauss_s[abs(t)]
                    sum_f += val * (w_c - rho * w_s)
                
                # Ponderación longitudinal (Eq. 9)
                w_m = gauss_m[abs(s)]
                sum_h += sum_f * w_m
                w_h_sum += w_m
                
            if w_h_sum > 0:
                H[y, x] = sum_h / w_h_sum
            else:
                H[y, x] = 0.0
                
    return H

# Auxiliares de matemáticas nativas requeridas por Numba
from math import exp, sqrt, ceil

def apply_fdog(img_gray, etf, sigma_m=3.0, sigma_c=1.0, rho=0.99, tau=0.5):
    """
    Calcula las respuestas de la acumulación FDoG y binariza la salida (Eq. 10).
    """
    sigma_s = 1.6 * sigma_c
    H = fdog_kernel(img_gray.astype(np.float32), etf, sigma_c, sigma_s, sigma_m, rho)
    
    # Binarización basada en la función hiperbólica tanh (Eq. 10)
    output = np.ones_like(img_gray, dtype=np.uint8) * 255
    # Condición: H(x) < 0 e 1 + tanh(H(x)) < tau
    # Para trabajar con intensidades de imagen, adaptamos la métrica del umbral
    condition = (H < 0) & ((1.0 + np.tanh(H)) < tau)
    output[condition] = 0
    
    return output

# =====================================================================
# PIPELINE ITERATIVO COMPLETO (SECCIÓN 3.2)
# =====================================================================

def coherent_line_drawing(img_bgr, etf_r=5, fdog_iter=3, sigma_m=3.0, sigma_c=4.0, rho=0.99, tau=0.5):
    """
    Pipeline principal que procesa la imagen y ejecuta el bucle de retroalimentación iterativo.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    print("[1/3] Calculando campo de direcciones ETF...")
    etf = compute_etf(gray, r=etf_r, iterations=3)
    
    print(f"[2/3] Ejecutando bucle iterativo FDoG ({fdog_iter} pasadas)...")
    current_img = gray.copy()
    
    for i in range(fdog_iter):
        print(f"      -> Procesando iteración {i + 1}...")
        # Aplicar opcionalmente un leve desenfoque gaussiano antes de la iteración (Sección 3.2)
        if i > 0:
            current_img = cv2.GaussianBlur(current_img, (3, 3), 0)
            
        line_map = apply_fdog(current_img, etf, sigma_m, sigma_c, rho, tau)
        
        # Superponer los bordes negros detectados sobre la imagen de trabajo (Sección 3.2)
        if i < fdog_iter - 1:
            current_img = np.where(line_map == 0, 0, gray)
            
    return line_map

# =====================================================================
# PARTE DE PRUEBA Y EJECUCIÓN
# =====================================================================
if __name__ == "__main__":
    # Cambia 'tu_imagen.jpg' por la ruta de la fotografía que desees transformar
    input_path = "./data/img/cerezo.png"
    
    img = cv2.imread(input_path)
    if img is None:
        # Si no hay imagen, creamos una de prueba (un círculo con ruido)
        print(f"No se encontró '{input_path}'. Generando imagen sintética de prueba...")
        img = np.ones((400, 400, 3), dtype=np.uint8) * 200
        cv2.circle(img, (200, 200), 100, (50, 50, 50), -1)
        # Añadir ruido gaussiano sutil
        noise = np.random.normal(0, 15, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = cv2.resize(img, (400, 400))  # Redimensionar para pruebas rápidas

    # Ejecutar algoritmo con los valores estándar sugeridos en el artículo
    # r=5, sigma_m=3.0, sigma_c=1.0, tau=0.5, 3 iteraciones
    result_lines = coherent_line_drawing(img, etf_r=1, fdog_iter=1, sigma_m=5.0, sigma_c=0.5, tau=0.1)
    
    # Mostrar resultados
    cv2.imshow("Imagen Original", img)
    cv2.imshow("Resultado: Coherent Line Drawing", result_lines)
    
    print("\n¡Listo! Presiona cualquier tecla sobre las ventanas para cerrar.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()