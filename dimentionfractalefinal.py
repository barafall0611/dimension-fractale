# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 11:06:23 2025

@author: bara.fall
"""




import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, morphology, img_as_ubyte
from skimage.util import img_as_bool
import os
from skimage.filters import threshold_otsu

# ------------------------------------------------------------
# 1️⃣ CHARGEMENT DE L'IMAGE
# ------------------------------------------------------------
image_path = r"C:\Users\bara.fall\Desktop\Manip\M925-6.6K-2.jpg"
img = io.imread(image_path)


if len(img.shape) == 3:
    gray = color.rgb2gray(img)
else:
    gray = img.astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min())

# ------------------------------------------------------------
# 2️⃣ ÉGALISATION DU CONTRASTE (CLAHE)
# ------------------------------------------------------------
gray_eq = exposure.equalize_adapthist(gray, clip_limit=0.03)

plt.figure(figsize=(8,6))
plt.imshow(gray_eq, cmap='gray')
plt.title("Image après égalisation adaptative du contraste (CLAHE)")
plt.axis('off')
plt.show()


# ------------------------------------------------------------
# 3️⃣ DÉTECTION DES CONTOURS (Canny)
# ------------------------------------------------------------
# Conversion en format 8 bits pour OpenCV
gray_8bit = img_as_ubyte(gray_eq)

# Détection de contours
edges = cv2.Canny(gray_8bit, 40, 120)  # seuils à ajuster selon contraste

plt.figure(figsize=(8,6))
plt.imshow(edges, cmap='gray')
plt.title("Contours détectés (Canny)")
plt.axis('off')
plt.show()

# ------------------------------------------------------------
# 4️⃣ FERMETURE ET NETTOYAGE MORPHOLOGIQUE
# ------------------------------------------------------------
# Dilatation et fermeture pour connecter les contours

kernel = np.ones((3,3), np.uint8)
closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
filled = morphology.remove_small_holes(closed.astype(bool),1)
clean = morphology.remove_small_objects(filled,1)

plt.figure(figsize=(8,6))
plt.imshow(clean, cmap='gray')
plt.title("Contours fermés et zones nettoyées")
plt.axis('off')
plt.show()


# ============================================================
# 5️⃣ Sélection manuelle d'une zone (agrégat)
# ============================================================
def select_region_manual(image, max_display=1200, window_name="Sélection d'une zone"):
    """
    Sélection manuelle d'une région d'intérêt (ROI) avec tracé rouge en temps réel.
    """
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

    print("\n🖱️ Clique et fais glisser pour sélectionner une zone (rectangle rouge en direct).")
    print("✅ Relâche pour valider, puis appuie sur 'ENTER' pour confirmer ou 'ESC' pour annuler.\n")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, draw_rectangle)
    cv2.imshow(window_name, disp_color)

    # Attendre la validation de l'utilisateur
    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    if len(roi_pts) < 2 or key == 27:  # 27 = ESC
        raise ValueError("⚠️ Aucune sélection détectée ou annulée.")

    (x1, y1), (x2, y2) = roi_pts
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    # Conversion coordonnées vers image originale
    x_full, y_full = int(x1 / scale_display), int(y1 / scale_display)
    w_full, h_full = int((x2 - x1) / scale_display), int((y2 - y1) / scale_display)

    print(f"✅ Zone sélectionnée : ({x_full}, {y_full}) → ({x_full + w_full}, {y_full + h_full})")

    selected_region = image[y_full:y_full + h_full, x_full:x_full + w_full]
    return selected_region



def fractal_dimension(image, show_steps=True, save_figures=False, output_dir="Fractal_Steps"):
    image = img_as_bool(image)
    assert image.ndim == 2

    h, w = image.shape
    print(image.shape)
    n = int(2**np.ceil(np.log2(max(h, w))))
    print(n)

    square = np.ones((n, n), dtype=bool)
    y0 = (n - h) // 2
    x0 = (n - w) // 2
    square[y0:y0+h, x0:x0+w] = image
    image = square

    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    N_total_list, N_objet_list, taux_list = [], [], []

    print("\n🔹 Début du comptage fractal (box-counting coloré)")
    print("-------------------------------------------------------------")
    print(f"{'Taille (px)':>10s} | {'Boîtes totales':>15s} | {'Boîtes objet (N)':>20s} | {'Taux couverture (%)':>20s}")
    print("-------------------------------------------------------------")

    for S in sizes:
        S = int(S)
        count_obj = 0
        nb_x = n // S
        nb_y = n // S
        N_total = nb_x * nb_y

        img_vis = np.stack([image]*3, axis=-1).astype(float)

        for y in range(0, n, S):
            for x in range(0, n, S):
                box = ~image[y:y+S, x:x+S]     
                
                if np.any(box):
                    count_obj += 1
                    color = [0, 1, 0]   # vert
                else:
                    color = [1, 0, 0]   # rouge
        
                # tracer les bords du carré
                img_vis[y:y+1, x:x+S] = color
                img_vis[y+S-1:y+S, x:x+S] = color
                img_vis[y:y+S, x:x+1] = color
                img_vis[y:y+S, x+S-1:x+S] = color
               
                taux = 100.0 * count_obj / N_total

        N_total_list.append(N_total)
        N_objet_list.append(count_obj)
        taux_list.append(taux)

        print(f"{S:>10d} | {N_total:>15d} | {count_obj:>20d} | {taux:>19.2f}%")
        
        if show_steps:
            plt.figure(figsize=(5,5))
            plt.imshow(img_vis, cmap='gray')
            plt.title(f"Taille = {S}px — N_objet = {count_obj} / {N_total} ({taux:.2f}%)")
            plt.axis('off')
            plt.show()

        if save_figures:
            plt.imsave(os.path.join(output_dir, f"grille_{S}px.png"), img_vis)

    print("----------------------------------------------------------------------------")
    

    # --- Dimension fractale normale ---
    sizes = np.array(sizes, dtype=float)
    N_obj = np.array(N_objet_list, dtype=float)
    mask = N_obj > 0
    coeffs = np.polyfit(np.log(1.0/sizes[mask]), np.log(N_obj[mask]), 1)
    D = coeffs[0]
    print(np.log(1.0/sizes), np.log(N_obj))
    plt.figure(figsize=(6,4))
    plt.plot(np.log(1.0/sizes), np.log(N_obj), 'o-', label=f'D ≈ {D:.3f}')
    plt.xlabel("log(1 / taille de boîte)")
    plt.ylabel("log(N_objet)")
    plt.title("Estimation de la dimension fractale (Box-Counting)")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"\n📏 Dimension fractale estimée : D = {D:.3f}\n")

    # --- NOUVELLE COURBE (taux < 50%) ---
    
    taux_array = np.array(taux_list)
    sizes_array = sizes
    N_obj_array = N_obj

    mask2 = (taux_array < 50) & (N_obj_array > 0)

    plt.figure(figsize=(6,4))
    plt.plot(np.log(1.0 / sizes_array[mask2]),
             np.log(N_obj_array[mask2]),
             'o-', label="Points filtrés (taux < 50%)")
    plt.xlabel("log(1 / taille de boîte)")
    plt.ylabel("log(N_objet)")

    plt.title("Courbe filtrée (taux < 50%)")
    plt.grid(True)
    plt.legend()
    plt.show()

    coeffs_f = np.polyfit(np.log(1.0 / sizes_array[mask2]),
                          np.log(N_obj_array[mask2]), 1)
    D_filtered = coeffs_f[0]

    print(f"📏 Nouvelle dimension fractale estimée (taux < 50%) : D_filtered = {D_filtered:.3f}\n")

    return D, D_filtered

    

try:
    # 1️⃣ Sélection de la zone
    selected_region = select_region_manual(clean)
    plt.figure(figsize=(5,5))
    plt.imshow(selected_region, cmap='gray')
    plt.title("Zone sélectionnée pour analyse fractale")
    plt.axis('off')
    plt.show()

    # 2️⃣ Binarisation automatique (Otsu)
  
    th = threshold_otsu(selected_region)
    binary = selected_region > th
    plt.figure(figsize=(5,5))
    plt.imshow(binary, cmap='gray')
    plt.title("Image binaire (Otsu)")
    plt.axis('off')
    plt.show()

    # 3️⃣ Calcul et visualisation des étapes fractales
    D, D_filtered = fractal_dimension(binary, show_steps=True)
    
    print(f"\n📏 Dimension fractale estimée : D = {D:.3f}")
    print(f"📏 Dimension fractale filtrée (<50%) : D_filtered = {D_filtered:.3f}")

except ValueError as e:
    print(e)
