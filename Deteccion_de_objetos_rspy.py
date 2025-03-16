import os
import cv2
import sys
import numpy as np
import time 

import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()  # -Oculta la ventana principal

# -Funcion para seleccionar la carpeta desde el administrador de archivos 
def seleccionar_carpeta(): 
    carpeta_base = filedialog.askdirectory(title='Selecciona una carpeta')
    # -Calcula el total de imagenes en cada carpeta 
    n_img_color = len(os.listdir(f'{carpeta_base}/color'))
    n_img_depth = len(os.listdir(f'{carpeta_base}/depth'))
    
    # -Verifica que la cantidad de imagenes sea la misma
    if(n_img_color != n_img_depth):
        print('La cantidad de imagenes en las carpetas color y depth son distintas') 
        sys.exit(1)

    # -Regresa el numero total de imagenes y el path de la carpeta 
    return n_img_color, carpeta_base

# -Funcion para cargar las imagenes 
def cargar_imagen(x,y,carpeta_base, k):
    # Cargar imágenes
    depth_path = os.path.join(carpeta_base, f'depth/prueba_{k}_color.png')
    color_path = os.path.join(carpeta_base, f'color/prueba_{k}_color.png')

    img_color = cv2.imread (color_path)
    img_depth = cv2.imread (depth_path, cv2.IMREAD_UNCHANGED)
    if img_color is None or img_depth is None:
        print(f"Imagen >{k}< no encontrada")
        sys.exit(1)
    
    # -Redimensionar imágenes
    img_color = cv2.resize(img_color, (x, y))
    img_depth = cv2.resize(img_depth, (x, y))

    return img_color,img_depth

# -Funcion para ajustar la resolucion de las imagenes 
def resolucion_imagen(ANCHO_X,ALTO_Y):
    # -Calculo de matrices 3x3
    x_s = ANCHO_X//3
    y_s = ALTO_Y//3
    segmentacion_coords = [
        (0, y_s,0,x_s),        # Matriz 1
        (0, y_s, x_s, x_s*2),      # Matriz 2
        (0, y_s, x_s*2, x_s*3),      # Matriz 3 
        (y_s, y_s*2, 0, x_s),      # Matriz 4
        (y_s, y_s*2, x_s, x_s*2),    # Matriz 5
        (y_s, y_s*2, x_s*2, x_s*3),    # Matriz 6
        (y_s*2, y_s*3, 0, x_s),     # Matriz 7
        (y_s*2, y_s*3, x_s, x_s*2),   # Matriz 8
        (y_s*2, y_s*3, x_s*2, x_s*3)    # Matriz 9
        ]    
    return segmentacion_coords

def main():

    fps = 30
    umbral = 15000
    umbral_prox = 40000
    frames_cache = np.zeros((fps, 3, 3), dtype=np.float32)
    temp_inicial = time.time()
    # -Seleccion de carpeta
    n_imagenes, carpeta_base = seleccionar_carpeta()

    # -Resolucion recomendada en raspberry 320x240
    # -Originalmente estaba como 640x480
    ANCHO_X, ALTO_Y = 320, 240
    segmentacion_coords = resolucion_imagen(ANCHO_X,ALTO_Y)    

    for k in range(n_imagenes):
        # -Cargar imágenes
        img_color, img_depth = cargar_imagen(ANCHO_X,ALTO_Y,carpeta_base,k)
        # -Aplicando efecto blur a la imagen de profundidad 
        img_depth = cv2.GaussianBlur(img_depth,(7,7),0)

        # -Inversion de la imagen de profundidad        
        img_inv = 65535 - img_depth 

        # -Selección del umbral
        # -Selecciona los valores que pasan el umbral
        u = img_inv >= 63500 
        # -Convierte la matriz de umbral al tipo de matriz original 
        # -Filtra los datos que no pasan el umbral
        img_filtrada = img_inv * u.astype(np.uint16)
        # -Segmenta la imagen
        segments = [img_filtrada[y1:y2, x1:x2] for (y1, y2, x1, x2) in segmentacion_coords] 
        # -Calcula el promedio
        promedios = np.array([np.mean(seg) for seg in segments]).reshape(3, 3)

        # -Actualiza el frames_cache        
        frames_cache[k % fps] = promedios

        # -Calcula el promedio corregido
        promedios_corregidos = np.mean(frames_cache, axis=0)

        # -Detección del objeto mediante el umbral 
        PU = promedios_corregidos >= umbral
        PUP = promedios_corregidos >= umbral_prox

        # -Dibuja los rectángulos en la imagen
        for row in range(PU.shape[0]):
            for col in range(PU.shape[1]):
                y1, y2, x1, x2 = segmentacion_coords[row * 3 + col]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                # -Calcula la distancia de los objetos 
                distancia = img_depth[cy, cx] / 1000
                distancia = f"{distancia:.2f} m"
                if PU[row, col] == 1: 
                    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if PUP[row, col] == 1: 
                    cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 0, 255), 2)                    
                
                cv2.putText(img_color,distancia, (x1, y1 + 15 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    # Mostrar la distancia en metros dentro del cuadro                
                    
        # -Muestra la imagen
        cv2.imshow('Deteccion de Objetos', img_color)
        cv2.waitKey(1)
        
    cv2.destroyAllWindows()
    temp_final = time.time()
    print(f"(seg): {temp_final - temp_inicial:.2f} segundos")
    
    return 0

if __name__ == '__main__':
    main()