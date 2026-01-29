# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:07:59 2025

@author: bara.fall
"""

# -*- coding: utf-8 -*-
"""
Pipeline complet : Prétraitement + sélection ROI + fractale FULL COVERAGE
Affichage des carrés (rouge/vert) pour chaque taille de boîte
Auteur : bara.fall + ChatGPT
Date : 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, morphology, img_as_ubyte
from skimage.util import img_as_bool
from skimage.filters import threshold_otsu
import os

# ------------------------------------------------------------
# 1️⃣ CHARGEMENT DE L'IMAGE
# ------------------------------------------------------------

image_path = r"C:\Users\bara.fall\Desktop\Manip\M925-6.6K-2.jpg"
img = io.imread(image_path)

# Conversion en niveaux de gris
if len(img.shape) == 3:
    gray = color.rgb2gray(img)
else:
    gray = img.astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min())

# ------------------------------------------------------------
# 2️⃣ ÉGALISATION DE CONTRASTE (CLAHE)
# ------------------------------------------------------------

gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)

plt.figure(figsize=(8,6))
plt.imshow(gray_eq, cmap='gray')
plt.title("Image après CLAHE")
plt.axis('off')
plt.show()

# ------------------------------------------------------------
# 3️⃣ DÉTECTION DE CONTOURS (CANNY)
# ------------------------------------------------------------

gray_8bit = img_as_ubyte(gray_eq)
edges = cv2.Canny(gray_8bit, 40, 120)

plt.figure(figsize=(8,6))
plt.imshow(edges, cmap='gray')
plt.title("Contours détectés")
plt.axis('off')
plt.show()

# ------------------------------------------------------------
# 4️⃣ FERMETURE MORPHOLOGIQUE
# ------------------------------------------------------------

kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
filled = morphology.remove_small_holes(closed.astype(bool), 1)
clean = morphology.remove_small_objects(filled, 1)

plt.figure(figsize=(8,6))
plt.imshow(clean, cmap='gray')
plt.title("Contours fermés et nettoyés")
plt.axis('off')
plt.show()

# ============================================================
# 5️⃣ SÉLECTION DE RÉGION MANUELLE (ROI)
# ============================================================

def select_region_manual(image, max_display=1200, window_name="Sélection d'une zone"):
    h, w = image.shape
    scale_display = min(1.0, max_display / max(w, h))

    disp = cv2.resize(img_as_ubyte(image), (int(w * scale_display), int(h * scale_display)))
    disp_color = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    roi_pts = []
    selecting = [False]

    def draw_rectangle(event, x, y, flags, param):
        img_temp = disp_color.copy()
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_pts[:] = [(x, y)]
            selecting[0] = True

        elif event == cv2.EVENT_MOUSEMOVE and selecting[0]:
            cv2.rectangle(img_temp, roi_pts[0], (x, y), (0, 0, 255), 2)
            cv2.imshow(window_name, img_temp)

        elif event == cv2.EVENT_LBUTTONUP:
            roi_pts.append((x, y))
            selecting[0] = False
            cv2.rectangle(img_temp, roi_pts[0], roi_pts[1], (0, 0, 255), 2)
            cv2.imshow(window_name, img_temp)

    print("\n🖱️ Clique & glisse pour sélectionner l'agrégat.")
    print("✔ ENTER pour valider, ESC pour annuler.\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_rectangle)
    cv2.imshow(window_name, disp_color)

    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    if len(roi_pts) < 2 or key == 27:
        raise ValueError("⚠️ Sélection annulée.")

    (x1, y1), (x2, y2) = roi_pts
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    x_full, y_full = int(x1 / scale_display), int(y1 / scale_display)
    w_full, h_full = int((x2 - x1) / scale_display), int((y2 - y1) / scale_display)

    print(f"→ ROI = ({x_full}, {y_full}) → ({x_full + w_full}, {y_full + h_full})")

    return image[y_full:y_full + h_full, x_full:x_full + w_full]

# ------------------------------------------------------------
# 6️⃣ FONCTION FRACTALE FULL-COVERAGE FIABLE
# ------------------------------------------------------------

def fractal_dimension_full_coverage(image, show_steps=True, save_figures=False, output_dir="Fractal_Steps"):
    """
    Méthode box-counting FULL COVERAGE (fiable)
    - Toutes les boîtes couvrent l'image, même si elles débordent
    - Version scientifiquement correcte du box-counting
    """

    image = img_as_bool(image)
    h, w = image.shape

    # tailles de boîtes = puissances de 2
    sizes = 2**np.arange(int(np.log2(min(h, w))), 1, -1)

    N_list = []
    log_inv = []
    log_N = []

    print("\n🔹 Box-counting FULL COVERAGE (fiable)")
    print("-------------------------------------------------------------")
    print(f"{'Taille S':>10s} | {'Boîtes objet (N)':>18s}")
    print("-------------------------------------------------------------")

    if save_figures and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for S in sizes:
        S = int(S)
        count = 0

        img_vis = np.stack([image]*3, axis=-1).astype(float)

        # FULL COVERAGE : on parcourt toute l'image, même si la boîte déborde
        for y in range(0, h, S):
            for x in range(0, w, S):

                y2 = min(y + S, h)
                x2 = min(x + S, w)

                box = image[y:y2, x:x2]

                # contient l'objet ?
                if np.any(~box):
                    count += 1
                    color = [0, 1, 0]   # vert
                else:
                    color = [1, 0, 0]   # rouge

                # tracer la boîte partielle si nécessaire
                img_vis[y:y+1, x:x2] = color
                img_vis[y2-1:y2, x:x2] = color
                img_vis[y:y2, x:x+1] = color
                img_vis[y:y2, x2-1:x2] = color

        N_list.append(count)
        log_inv.append(np.log(1/S))
        log_N.append(np.log(count))

        print(f"{S:>10d} | {count:>18d}")

        if show_steps:
            plt.figure(figsize=(5,5))
            plt.imshow(img_vis)
            plt.title(f"Taille S = {S} px — N = {count}")
            plt.axis('off')
            plt.show()

        if save_figures:
            plt.imsave(os.path.join(output_dir, f"fullcov_S{S}.png"), img_vis)

    # estimation de la pente log(N) vs log(1/S)
    log_inv = np.array(log_inv)
    log_N = np.array(log_N)

    coeffs = np.polyfit(log_inv, log_N, 1)
    D = coeffs[0]

    plt.figure(figsize=(6,4))
    plt.plot(log_inv, log_N, 'o-', label=f"D ≈ {D:.3f}")
    plt.xlabel("log(1/S)")
    plt.ylabel("log(N)")
    plt.grid(True)
    plt.legend()
    plt.title("Dimension fractale (Full Coverage)")
    plt.show()

    print(f"\n📏 Dimension fractale estimée : D = {D:.3f}\n")
    return D

# ============================================================
# 7️⃣ PIPELINE FINAL
# ============================================================

try:
    selected_region = select_region_manual(clean)

    plt.figure(figsize=(5,5))
    plt.imshow(selected_region, cmap='gray')
    plt.title("Zone sélectionnée")
    plt.axis('off')
    plt.show()

    th = threshold_otsu(selected_region)
    binary = selected_region > th

    plt.figure(figsize=(5,5))
    plt.imshow(binary, cmap='gray')
    plt.title("Image binaire (Otsu)")
    plt.axis('off')
    plt.show()

    # Méthode FRACTALE FIABLE
    D = fractal_dimension_full_coverage(binary, show_steps=True)

    print(f"📏 Dimension fractale finale : D = {D:.3f}")

except ValueError as e:
    print(e)
