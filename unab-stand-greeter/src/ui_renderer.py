# -*- coding: utf-8 -*-
"""
Render de UI sobre frames (OpenCV + PIL):
- Logo UNAB
- Textos con sombra ligera
- Panel QR
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import qrcode

def _hex_to_bgr(hx: str):
    hx = hx.lstrip('#')
    return (int(hx[4:6],16), int(hx[2:4],16), int(hx[0:2],16))

class UIRenderer:
    def __init__(self, width, height, brand_primary="#A00321", text_color="#FFFFFF",
                 logo_path=None):
        self.w, self.h = width, height
        self.primary_bgr = _hex_to_bgr(brand_primary)
        self.text_bgr = _hex_to_bgr(text_color)
        self.logo = None
        if logo_path:
            self.logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # respeta alpha
            if self.logo is not None:
                # Escalar logo a ~12% de altura
                target_h = int(self.h * 0.12)
                scale = target_h / self.logo.shape[0]
                self.logo = cv2.resize(self.logo, (int(self.logo.shape[1]*scale), target_h))

        # Fuentes (intenta cargar del sistema; fallback a default)
        try:
            self.font_big = ImageFont.truetype("Arial.ttf", size=int(self.h*0.07))
            self.font_mid = ImageFont.truetype("Arial.ttf", size=int(self.h*0.04))
        except:
            self.font_big = ImageFont.load_default()
            self.font_mid = ImageFont.load_default()

    def _pil_overlay(self, frame_bgr):
        return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")

    def _to_bgr(self, pil_img):
        return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    def _draw_text_center(self, img: Image.Image, text, y, font, fill=(255,255,255,255)):
        draw = ImageDraw.Draw(img)
        W, _ = img.size
        bbox = draw.textbbox((0,0), text, font=font)
        text_w = bbox[2]-bbox[0]
        x = (W - text_w)//2
        # sombra
        draw.text((x+2, y+2), text, font=font, fill=(0,0,0,160))
        draw.text((x, y), text, font=font, fill=fill)
        return img

    def _overlay_logo(self, base: Image.Image, margin=20):
        if self.logo is None:
            return base
        overlay = base.copy()
        lh, lw = self.logo.shape[:2]
        x = base.width - lw - margin
        y = margin
        # composición con alpha si 4 canales
        logo = self.logo
        if logo.shape[2] == 4:
            # convertir a PIL
            logo_rgba = Image.fromarray(cv2.cvtColor(logo, cv2.COLOR_BGRA2RGBA))
            overlay.paste(logo_rgba, (x, y), logo_rgba)
        else:
            # sin alpha
            logo_rgb = cv2.cvtColor(logo, cv2.COLOR_BGR2RGB)
            logo_pil = Image.fromarray(logo_rgb)
            overlay.paste(logo_pil, (x, y))
        return overlay

    def _make_qr(self, url, size_ratio=0.35):
        size = int(min(self.w, self.h) * size_ratio)
        qr = qrcode.QRCode(version=2, box_size=10, border=2)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGBA")
        return img.resize((size, size))

    def render_idle(self, frame_bgr, text_idle, text_prompt=None):
        base = self._pil_overlay(frame_bgr)
        base = self._overlay_logo(base)
        y = int(self.h * 0.35)
        base = self._draw_text_center(base, text_idle, y, self.font_big)
        if text_prompt:
            base = self._draw_text_center(base, text_prompt, y + int(self.h*0.09),
                                          self.font_mid, fill=(255,255,255,220))
        return self._to_bgr(base)

    def render_greeting(self, frame_bgr, lines):
        base = self._pil_overlay(frame_bgr)
        base = self._overlay_logo(base)
        y = int(self.h * 0.30)
        for i, line in enumerate(lines):
            base = self._draw_text_center(base, line, y + i*int(self.h*0.09),
                                          self.font_big if i == 0 else self.font_mid)
        return self._to_bgr(base)

    def render_qr_panel(self, frame_bgr, url, caption="Escanea el QR para saber más"):
        base = self._pil_overlay(frame_bgr)
        base = self._overlay_logo(base)
        qr = self._make_qr(url)
        # Panel semitransparente
        overlay = Image.new("RGBA", base.size, (0,0,0,0))
        panel_h = int(self.h * 0.6)
        panel_w = int(self.w * 0.5)
        px = (self.w - panel_w)//2
        py = (self.h - panel_h)//2
        panel = Image.new("RGBA", (panel_w, panel_h), (0,0,0,140))
        overlay.paste(panel, (px, py))
        # Pegar QR
        qx = px + (panel_w - qr.size[0])//2
        qy = py + int(panel_h*0.18)
        overlay.paste(qr, (qx, qy), qr)
        # Texto
        draw = ImageDraw.Draw(overlay)
        # Título
        self._draw_text_center(overlay, "Información de la carrera", py + int(panel_h*0.07), self.font_mid)
        self._draw_text_center(overlay, caption, qy + qr.size[1] + int(self.h*0.04), self.font_mid)
        out = Image.alpha_composite(base, overlay)
        return self._to_bgr(out)
