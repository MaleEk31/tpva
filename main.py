import cv2
import numpy as np

# Charger les images

img1 = cv2.imread('house1.pgm', 0)
img2 = cv2.imread('house2.pgm', 0)
h, w = img1.shape[:2]
corners = np.zeros((h, w))
# Convertir l'image en float32
img1 = np.float32(img1)
for y in range(1, h-1):
    for x in range(1, w-1):
        E = []
        for u, v in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            # Calculer la différence entre les pixels
            diff = np.sum((img1[y, x] - img1[y+v, x+u])**2)
            E.append(diff)
        corners[y, x] = np.min(E)
seuil=0.1
corners[corners < seuil*np.max(corners)] = 0
corners = cv2.dilate(corners, np.ones((3,3)))
# Convertir l'image en couleur
img_color = cv2.cvtColor(img1.astype(np.uint8), cv2.COLOR_GRAY2BGR)
# Créer un masque binaire pour les coins détectés
ret, mask = cv2.threshold(corners.astype(np.uint8), 0, 255, cv2.THRESH_BINARY)
# Affecter les pixels rouges aux positions des coins détectés
img_color[mask > 0] = [0, 0, 255]
# Afficher l'image avec les coins détectés
cv2.imshow("Corners detected", img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Définir la fonction de différence
def diff(img1, p1, img2, p2):
    return np.abs(int(img1[p1[1], p1[0]]) - int(img2[p2[1], p2[0]]))

def get_neighbors(point, max_x, max_y):
    """Retourne la liste des voisins d'un point"""
    x, y = point
    neighbors = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            if dx == 0 and dy == 0:
                continue
            neighbor_x, neighbor_y = x + dx, y + dy
            if in_bounds(neighbor_x, neighbor_y, max_x, max_y):
                neighbors.append((neighbor_x, neighbor_y))
    return neighbors


def in_bounds(x, y, max_x, max_y):
    """Vérifie si un point est dans les limites d'une image"""
    return 0 <= x < max_x and 0 <= y < max_y


def get_weight(point1, point2, img):
    """Retourne le poids entre deux points"""
    intensity_diff = abs(img[point1[1], point1[0]] - img[point2[1], point2[0]])
    if intensity_diff > 0:
        return 1.0 / intensity_diff
    else:
        return float('inf')
# Initialiser la matrice de correspondance
corr = np.zeros((h, w, h, w))

# Répéter jusqu'à convergence
diff_thresh = 10 # Seuil de différence pour la convergence
max_iter = 100 # Nombre maximal d'itérations
for i in range(max_iter):
    # Pour chaque paire de coins correspondants
    for y1 in range(h):
        for x1 in range(w):
            if corners[y1, x1] > 0:
                for y2 in range(h):
                    for x2 in range(w):
                        if corners[y2, x2] > 0:
                            # Calculer la différence entre les coins correspondants
                            d = diff(img1, (x1, y1), img2, (x2, y2))
                            # Mettre à jour la valeur de chaque coin correspondant
                            corr[y1, x1, y2, x2] = (1 + d) / (1 + np.sum(corr[y1, x1, :, :]))

    # Vérifier la convergence
    if np.max(np.abs(corr - np.roll(corr, 1, axis=2))) < diff_thresh:
        break

# Créer une image de résultat qui montre les correspondances entre les coins des deux images
result = np.zeros((h, w * 2, 3), dtype=np.uint8)
result[:, :w, 0] = img1
result[:, w:, 1] = img2
for y1 in range(h):
    for x1 in range(w):
        if corners[y1, x1] > 0:
            # Trouver le coin correspondant dans l'image 2
            x2, y2 = np.unravel_index(np.argmax(corr[y1, x1, :, :]), (h, w))
            if corners[y2, x2] > 0:
                # Dessiner une ligne entre les coins correspondants
                cv2.line(result, (x1, y1), (x2 + w, y2), (0, 255, 0), 1)
# Afficher l'image de résultat
cv2.imshow('Correspondences', result)
cv2.waitKey(0)
cv2.destroyAllWindows()