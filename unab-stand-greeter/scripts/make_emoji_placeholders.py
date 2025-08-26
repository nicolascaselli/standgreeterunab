# scripts/make_emoji_placeholders.py
# -*- coding: utf-8 -*-
"""
Genera PNGs placeholder para emojis del UI (wave, thumbs_up, mute) sin depender
de fuentes con soporte emoji. Usa texto ASCII grande centrado.
Compatible con Pillow >= 10 (usa textbbox).
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path("src/assets/emoji")
OUT.mkdir(parents=True, exist_ok=True)

def load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    Intenta cargar una fuente del sistema con buena cobertura.
    En macOS suele existir Arial Unicode. Si falla, usa default.
    """
    candidates = [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for p in candidates:
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()

def center_text(draw: ImageDraw.ImageDraw, img_w: int, img_h: int, text: str, font: ImageFont.FreeTypeFont):
    """
    Calcula la posición (x,y) para centrar `text` dentro de (img_w, img_h) usando textbbox().
    """
    # bbox = (left, top, right, bottom)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (img_w - tw) // 2
    y = (img_h - th) // 2
    return x, y

def make_icon(name: str, bg_rgba: tuple, label: str, fg_rgba: tuple = (255, 255, 255, 255)):
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Círculo de color
    pad = 8
    d.ellipse((pad, pad, size - pad, size - pad), fill=bg_rgba)

    # Texto centrado (sombra suave)
    font = load_font(110)  # ajusta tamaño si quieres más grande/pequeño
    x, y = center_text(d, size, size, label, font)
    d.text((x + 2, y + 2), label, fill=(0, 0, 0, 120), font=font)
    d.text((x, y), label, fill=fg_rgba, font=font)

    out_path = OUT / f"{name}.png"
    img.save(out_path)
    print(f"OK -> {out_path}")

if __name__ == "__main__":
    # Etiquetas ASCII para no depender de fuentes con emoji
    make_icon("wave",      (255, 193,   7, 255), "HI")   # 👋
    make_icon("thumbs_up", ( 76, 175,  80, 255), "OK")   # 👍
    make_icon("mute",      (244,  67,  54, 255), "M")    # 🔇

    print("Listo. Colocados en src/assets/emoji/")
