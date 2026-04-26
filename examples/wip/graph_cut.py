import numpy as np
import cv2
import maxflow

# =========================
# CONFIG
# =========================

COLOR_0 = np.array([0, 0, 255])   # rojo (BGR)
COLOR_1 = np.array([255, 0, 0])   # azul (BGR)

LAMBDA = 20.0

# =========================
# DISTANCIA DE COLOR
# =========================

def color_dist(a, b):
    return np.sum((a - b) ** 2)

# =========================
# GRAPH CUT
# =========================

def segment(image):

    H, W, _ = image.shape

    g = maxflow.Graph[float]()
    nodes = g.add_nodes(H * W)

    def idx(y, x):
        return y * W + x

    # =====================
    # DATA TERM
    # =====================
    for y in range(H):
        for x in range(W):

            pixel = image[y, x]

            d0 = color_dist(pixel, COLOR_0)
            d1 = color_dist(pixel, COLOR_1)

            g.add_tedge(idx(y, x), d0, d1)

    # =====================
    # SMOOTHNESS
    # =====================
    for y in range(H):
        for x in range(W):

            for dy, dx in [(1,0), (0,1)]:
                ny, nx = y + dy, x + dx

                if ny < H and nx < W:

                    diff = color_dist(image[y,x], image[ny,nx]) + 1e-5
                    w = LAMBDA * (1.0 / diff)

                    g.add_edge(idx(y,x), idx(ny,nx), w, w)

    print("[INFO] Running maxflow...")
    flow = g.maxflow()
    print(f"[INFO] Energy (flow): {flow:.2f}")

    # =====================
    # RESULTADO
    # =====================
    labels = np.zeros((H, W), dtype=np.uint8)

    for y in range(H):
        for x in range(W):
            labels[y, x] = g.get_segment(idx(y, x))

    return labels

# =========================
# VISUALIZACIÓN
# =========================

def visualize(image, labels):

    seg = np.zeros_like(image)

    seg[labels == 0] = [0, 0, 255]
    seg[labels == 1] = [255, 0, 0]

    cv2.imshow("Original", image)
    cv2.imshow("Labels", labels * 255)
    cv2.imshow("Segmented", seg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    #input_path = "./data/img/scarlettNO.webp"
    #input_path = "./data/img/cerezoNO.png"
    #input_path = "./data/img/jokerNO.webp"
    input_path = "./data/img/guacaNO.jpg"
    #input_path = "./data/img/Wilhelm_SchirmerNO.jpeg"
    #input_path = "./data/img/Rabbit in a Field of Purple Flowers.png"
    #input_path = "./data/img/marilyn-monroeNO.jpg"
    #input_path = "./data/img/mar_rayoNO.jpeg"
    #input_path = "./data/img/dmonNO.jpg" 
    #input_path = "./data/img/noche_estrelladaNO.jpg"
    #input_path = "./data/img/mujerIANO.webp"
    #input_path = "./data/img/womanIA2NO.webp"
    #input_path = "./data/img/white_dressNO.jpeg"
    #input_path = "./data/img/vintageNO.jpg"
    #input_path = "./data/img/VAN-GOGHNO.jpg"
    #input_path = "./data/img/cuartoNO.jpeg"
    #input_path = "./data/img/barNO.jpeg"
    #input_path = "./data/img/armadura_doradaNO.jpeg"
    #input_path = "./data/img/027NO.webp"
    #input_path = "./cdf_color_transfer_lab_result.jpg"

    img = cv2.imread(input_path)

    if img is None:
        print("No se pudo cargar la imagen")
        exit()

    from pixelops.filtering import mean_shift_filter    
    import pixelops as pix  
    img = mean_shift_filter(img, hs=15, hr=15.0, max_iter=25, eps=1.0)
    img = pix.normalize_to_uint8(img)


    labels = segment(img)

    print("[INFO] Pixels clase 0:", np.sum(labels == 0))
    print("[INFO] Pixels clase 1:", np.sum(labels == 1))

    visualize(img, labels)