# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 11:42:38 2025

@author: celia
"""

#!/usr/bin/env python3
import json
import base64
from pathlib import Path
from typing import List, Tuple, Dict, Any
from PIL import Image, ImageDraw

def cargar_imagen(labelme: Dict[str, Any], base_dir: Path = Path(".")) -> Image.Image:
    """
    Intenta cargar la imagen desde imagePath; si no existe o falla, usa imageData (base64).
    Valida contra imageWidth / imageHeight si están presentes.
    """
    img = None
    image_path = labelme.get("imagePath")
    if image_path:
        ruta = (base_dir / image_path).expanduser()
        if ruta.exists():
            img = Image.open(ruta).convert("RGBA")

    if img is None:
        image_data = labelme.get("imageData")
        if not image_data:
            raise FileNotFoundError(
                "No se pudo cargar la imagen: ni 'imagePath' válido ni 'imageData' presente."
            )
        # Labelme guarda la imagen como base64 sin encabezado data:...
        try:
            raw = base64.b64decode(image_data)
        except Exception as e:
            raise ValueError(f"imageData no es un base64 válido: {e}")
        img = Image.open(io.BytesIO(raw)).convert("RGBA")  # type: ignore

    # Verificación opcional de tamaño
    w, h = img.size
    iw = labelme.get("imageWidth")
    ih = labelme.get("imageHeight")
    if isinstance(iw, int) and isinstance(ih, int) and (w != iw or h != ih):
        # Si difiere, redimensionar para calzar con las coordenadas
        img = img.resize((iw, ih))
    return img

def extraer_poligonos(labelme: Dict[str, Any]) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """
    Devuelve lista de (label, puntos) para cada shape tipo 'polygon'.
    """
    shapes = labelme.get("shapes", [])
    polys = []
    for s in shapes:
        if s.get("shape_type") == "polygon" and "points" in s:
            label = s.get("label", "sin_etiqueta")
            pts = [(float(x), float(y)) for x, y in s["points"]]
            polys.append((label, pts))
    if not polys:
        raise ValueError("No se encontraron shapes de tipo 'polygon' en el JSON.")
    return polys

def dibujar_poligonos(
    img: Image.Image,
    polys: List[Tuple[str, List[Tuple[float, float]]]],
    colores_por_label: Dict[str, Tuple[int,int,int,int]] | None = None,
    borde=(255, 0, 0, 255),
    relleno=(255, 0, 0, 80),
    width: int = 3,
    dibujar_texto: bool = True,
) -> Image.Image:
    """
    Dibuja cada polígono con (borde, relleno). Permite color por etiqueta.
    """
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    for label, pts in polys:
        color_borde = borde
        color_relleno = relleno
        if colores_por_label and label in colores_por_label:
            rgb = colores_por_label[label]
            color_borde = rgb
            # Relleno semitransparente del mismo color
            color_relleno = (rgb[0], rgb[1], rgb[2], 80) if len(rgb) == 4 else (*rgb, 80)

        # Cerrar polígono si hace falta
        if pts[0] != pts[-1]:
            pts_cerrado = pts + [pts[0]]
        else:
            pts_cerrado = pts

        draw.polygon(pts, outline=color_borde, fill=color_relleno)
        draw.line(pts_cerrado, fill=color_borde, width=width, joint="curve")

        if dibujar_texto:
            # Coloca la etiqueta cerca del primer punto
            x0, y0 = pts[0]
            draw.text((x0 + 3, y0 + 3), label, fill=color_borde)

    return Image.alpha_composite(img, overlay)

def guardar(img: Image.Image, salida: Path) -> Path:
    salida = salida.with_suffix(".png")
    img.save(salida, format="PNG")
    return salida

def main():
    import argparse, io  # io usado en cargar_imagen si se usa imageData
    parser = argparse.ArgumentParser(description="Dibuja polígonos de un JSON Labelme sobre una imagen.")
    parser.add_argument("json_path", type=Path, help="Ruta al archivo JSON de Labelme.")
    parser.add_argument("-o", "--output", type=Path, default=None, help="PNG de salida (opcional).")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Directorio base para resolver imagePath.")
    args = parser.parse_args()

    # Cargar JSON
    with open(args.json_path, "r", encoding="utf-8") as f:
        labelme = json.load(f)

    # Parche local para que cargar_imagen use io (porque lo importamos en main)
    globals()["io"] = __import__("io")

    img = cargar_imagen(labelme, base_dir=args.base_dir)
    polys = extraer_poligonos(labelme)

    # Puedes mapear colores por etiqueta si quieres distinguir
    colores = {
        # "rafaga": (0, 200, 255, 255),  # cian, por ejemplo
    }

    out_img = dibujar_poligonos(
        img,
        polys,
        colores_por_label=colores,
        borde=(255, 0, 0, 255),
        relleno=(255, 0, 0, 80),
        width=3,
        dibujar_texto=True,
    )

    # Nombre por defecto
    out_path = args.output
    if out_path is None:
        base = args.json_path.with_suffix("").name
        out_path = args.json_path.with_name(f"{base}_overlay.png")

    final_path = guardar(out_img, out_path)
    print(f"Guardado: {final_path}")

if __name__ == "__main__":
    import io
    from datetime import datetime

    # Carpeta donde está este script (funciona bien en Spyder con runfile)
    try:
        base_dir = Path(__file__).parent
    except NameError:
        # Por si se ejecuta en un entorno donde __file__ no está definido:
        base_dir = Path.cwd()

    print(f"Trabajando en: {base_dir}")

    # Busca todos los JSON de Labelme en la misma carpeta
    json_files = sorted(base_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError("No se encontraron archivos .json en la carpeta del script.")

    for json_path in json_files:
        try:
            print(f"\nProcesando: {json_path.name}")
            with open(json_path, "r", encoding="utf-8") as f:
                labelme = json.load(f)

            # Cargar imagen usando imagePath del JSON o imageData
            img = cargar_imagen(labelme, base_dir=base_dir)

            # Extraer polígonos
            polys = extraer_poligonos(labelme)

            # Dibujo (puedes personalizar colores por etiqueta en 'colores')
            colores = {
                # "rafaga": (0, 200, 255, 255),
            }
            out_img = dibujar_poligonos(
                img,
                polys,
                colores_por_label=colores,
                borde=(255, 0, 0, 255),
                relleno=(255, 0, 0, 80),
                width=3,
                dibujar_texto=True,
            )

            # Nombre de salida junto al JSON: <nombreJSON>_overlay.png
            out_path = json_path.with_name(f"{json_path.stem}_overlay.png")
            guardar(out_img, out_path)
            print(f"✅ Guardado: {out_path}")

        except Exception as e:
            print(f"⚠️ Error con {json_path.name}: {e}")
