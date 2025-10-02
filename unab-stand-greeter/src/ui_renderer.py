# src/ui_renderer.py
# -*- coding: utf-8 -*-
import cv2
import qrcode
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import platform
import re

# --- Conjuntos útiles para detectar emojis que usamos (básicos + tonos) ---
SKIN_TONES = {chr(cp) for cp in range(0x1F3FB, 0x1F3FF + 1)}  # 🏻 🏼 🏽 🏾 🏿
# Base que necesitamos cubrir sí o sí en tus textos:
BASE_EMOJIS = {"👋", "👍", "✌"}  # usaremos ✌ + tono como token compuesto
# Regex simple para capturar tokens (✌ + (tono)?), (👍 + (tono)?), (👋 + (tono)?)
EMOJI_TOKEN_RE = re.compile(
    "("
    r"\u270C[\U0001F3FB-\U0001F3FF]?|"      # ✌ + tono opcional
    r"\U0001F44D[\U0001F3FB-\U0001F3FF]?|"  # 👍 + tono opcional
    r"\U0001F44B[\U0001F3FB-\U0001F3FF]?|"  # 👋 + tono opcional
    r"\u2728\uFE0F?"                        # ✨ (sparkles) + VS16 opcional
    ")"
)

class UIRenderer:
    def __init__(self, width, height, brand_primary, text_color, assets, show_qr_panel=False, logo_height_ratio=None):
        """
        :param width: ancho lógico del canvas
        :param height: alto lógico del canvas
        :param brand_primary: color primario HEX (ej. "#A00321")
        :param text_color: color de texto HEX (ej. "#FFFFFF")
        :param assets: dict con rutas y fuentes:
            - "logo": Path al PNG con alfa del logo
            - "font_path": Path a la fuente principal (TTF)
            - "emoji_font_path": Path a la fuente de emojis (TTF/TTC)
            - "logo_height_ratio": (opcional) fracción del alto de la ventana
        :param show_qr_panel: bool
        :param logo_height_ratio: opcional (prioridad sobre assets["logo_height_ratio"])
        """
        self.width = int(width)
        self.height = int(height)
        self.brand_primary_bgr = self._hex_to_bgr(brand_primary)
        self.text_color_rgb = tuple(reversed(self._hex_to_bgr(text_color)))
        self.show_qr_panel = bool(show_qr_panel)
        self.qr_cache = {}
        self.system = platform.system()

        # --- Parámetro de escala proporcional del logo ---
        ratio_from_assets = (assets or {}).get("logo_height_ratio", None)
        self.logo_height_ratio = float(
            logo_height_ratio if logo_height_ratio is not None else (ratio_from_assets if ratio_from_assets is not None else 0.20)
        )
        self.logo_height_ratio = max(0.05, min(0.5, self.logo_height_ratio))

        # Logo RAW (sin escalar); se escala por frame (alto proporcional)
        self._logo_raw = self._load_logo_raw((assets or {}).get("logo"))
        self._logo_scaled = None
        self._logo_cache_h = None  # cache por altura del frame

        # Fuentes
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.emoji_font_large = None
        self.emoji_font_medium = None

        font_path = (assets or {}).get("font_path")
        if font_path and Path(font_path).exists():
            try:
                self.font_large = ImageFont.truetype(str(font_path), 68)
                self.font_medium = ImageFont.truetype(str(font_path), 42)
                self.font_small = ImageFont.truetype(str(font_path), 24)
                print(f"INFO: Fuente principal cargada desde {font_path}")
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo cargar la fuente principal: {e}")

        emoji_font_path = (assets or {}).get("emoji_font_path")
        if emoji_font_path and Path(emoji_font_path).exists():
            self._load_emoji_fonts(emoji_font_path)
        else:
            print("ADVERTENCIA: No se encontró la fuente de emojis")

    # ---------------------- Fuentes de Emoji ----------------------

    def _load_emoji_fonts(self, emoji_font_path):
        print(f"INFO: Intentando cargar fuente de emojis desde {emoji_font_path} en {self.system}")
        if self.system == "Darwin":
            font_sizes_large = [68, 64, 60, 56, 48]
            font_sizes_medium = [42, 40, 36, 32, 28]
            font_index = 0
        elif self.system == "Windows":
            font_sizes_large = [72, 68, 64, 60, 56]
            font_sizes_medium = [44, 42, 40, 36, 32]
            font_index = 0
        else:
            font_sizes_large = [68, 64, 60, 56, 48, 40]
            font_sizes_medium = [42, 40, 36, 32, 28, 24]
            font_index = 0

        for size in font_sizes_large:
            try:
                if str(emoji_font_path).endswith('.ttc'):
                    self.emoji_font_large = ImageFont.truetype(str(emoji_font_path), size, index=font_index)
                else:
                    self.emoji_font_large = ImageFont.truetype(str(emoji_font_path), size)
                print(f"INFO: Fuente de emojis grande cargada con tamaño {size}")
                break
            except Exception as e:
                print(f"DEBUG: Tamaño grande {size} falló: {e}")
                continue

        for size in font_sizes_medium:
            try:
                if str(emoji_font_path).endswith('.ttc'):
                    self.emoji_font_medium = ImageFont.truetype(str(emoji_font_path), size, index=font_index)
                else:
                    self.emoji_font_medium = ImageFont.truetype(str(emoji_font_path), size)
                print(f"INFO: Fuente de emojis mediana cargada con tamaño {size}")
                break
            except Exception as e:
                print(f"DEBUG: Tamaño mediano {size} falló: {e}")
                continue

        if self.emoji_font_large is None and self.emoji_font_medium is None:
            print("ADVERTENCIA: No se pudo cargar ninguna fuente de emojis")
            self._try_fallback_emoji_fonts()
        else:
            self._test_emoji_font()

    def _try_fallback_emoji_fonts(self):
        print("INFO: Intentando fuentes de emojis alternativas del sistema...")
        if self.system == "Windows":
            fallback_fonts = [
                "C:/Windows/Fonts/segmdl2.ttf",
                "C:/Windows/Fonts/arial.ttf"
            ]
        elif self.system == "Darwin":
            fallback_fonts = [
                "/System/Library/Fonts/Helvetica.ttc"
            ]
        else:
            fallback_fonts = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/TTF/DejaVuSans.ttf"
            ]
        for font_path in fallback_fonts:
            if Path(font_path).exists():
                try:
                    self.emoji_font_large = ImageFont.truetype(font_path, 68)
                    self.emoji_font_medium = ImageFont.truetype(font_path, 42)
                    print(f"INFO: Usando fuente fallback: {font_path}")
                    return
                except Exception:
                    continue
        print("ADVERTENCIA: No se pudo cargar ninguna fuente alternativa")

    def _test_emoji_font(self):
        test_font = self.emoji_font_large or self.emoji_font_medium
        if test_font:
            try:
                img = Image.new('RGB', (100, 100), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), "👋", font=test_font)  # prueba
                print(f"INFO: Fuente de emojis verificada correctamente en {self.system}")
            except Exception as e:
                print(f"ADVERTENCIA: Error al verificar fuente de emojis: {e}")

    # ---------------------- Logo dinámico ----------------------

    def _load_logo_raw(self, logo_path):
        if not logo_path:
            return None
        p = Path(logo_path)
        if not p.exists():
            return None
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    def _get_scaled_logo(self, frame_h):
        if self._logo_raw is None:
            return None
        target_h = max(1, int(float(frame_h) * float(self.logo_height_ratio)))
        if self._logo_scaled is not None and self._logo_cache_h == target_h:
            return self._logo_scaled
        h, w = self._logo_raw.shape[:2]
        scale = target_h / float(h)
        target_w = max(1, int(w * scale))
        self._logo_scaled = cv2.resize(self._logo_raw, (target_w, target_h), interpolation=cv2.INTER_AREA)
        self._logo_cache_h = target_h
        return self._logo_scaled

    # ---------------------- Texto con emojis (tokenizado) ----------------------

    def _tokenize_text(self, text: str):
        """
        Divide 'text' en lista de tokens: strings que son o no emojis.
        Soporta: '✌', '✌' + tono, '👍' (+ tono), '👋' (+ tono).
        """
        tokens = []
        i = 0
        n = len(text)
        while i < n:
            # intento: ¿coincide uno de nuestros emojis con tono?
            m = EMOJI_TOKEN_RE.match(text, i)
            if m:
                token = m.group(0)
                tokens.append(("emoji", token))
                i += len(token)
                continue
            # si no, carácter normal
            tokens.append(("text", text[i]))
            i += 1
        return tokens

    def _draw_text_with_emojis(self, draw, text, start_xy, font, emoji_font, fill):
        """
        Dibuja 'text' mezclando fuente principal y fuente de emojis (si existe),
        soportando tokens con tono de piel.
        """
        if not text:
            return

        x, y = start_xy
        tokens = self._tokenize_text(str(text))

        for kind, token in tokens:
            if kind == "emoji" and emoji_font is not None:
                # Algunos PIL en Windows renderizan más bajo; ajustamos un pelín
                y_shift = -4 if self.system == "Windows" else 0
                try:
                    draw.text((x, y + y_shift), token, font=emoji_font)
                    # medir ancho del token con emoji_font
                    try:
                        bbox = emoji_font.getbbox(token)
                        adv = bbox[2] - bbox[0]
                    except Exception:
                        adv = int(emoji_font.getlength(token)) if hasattr(emoji_font, "getlength") else 40
                    x += max(adv, 20)
                    continue
                except Exception:
                    # si falla el emoji_font, cae a texto normal
                    pass

            # Texto normal (o fallback)
            if font is not None:
                draw.text((x, y), token, font=font, fill=fill)
                try:
                    bbox = font.getbbox(token)
                    adv = bbox[2] - bbox[0]
                except Exception:
                    adv = int(font.getlength(token)) if hasattr(font, "getlength") else 10
                x += max(adv, 6)
            else:
                draw.text((x, y), token, fill=fill)
                x += 10

    # ---------------------- Render base + QR ----------------------

    def _render_base(self, img, *lines):
        """Renderiza líneas y luego superpone el logo proporcional."""
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        text_color = self.text_color_rgb
        shadow_color = (0, 0, 0)

        # Línea principal
        if len(lines) > 0 and lines[0]:
            main_text = str(lines[0])
            if self.font_large:
                # sombra
                self._draw_text_with_emojis(draw, main_text, (42, 32), self.font_large, self.emoji_font_large, shadow_color)
                # texto
                self._draw_text_with_emojis(draw, main_text, (40, 30), self.font_large, self.emoji_font_large, text_color)

        # Subtítulo
        if len(lines) > 1 and lines[1]:
            sub_text = str(lines[1])
            if self.font_medium:
                self._draw_text_with_emojis(draw, sub_text, (42, self.height - 78), self.font_medium, self.emoji_font_medium, shadow_color)
                self._draw_text_with_emojis(draw, sub_text, (40, self.height - 80), self.font_medium, self.emoji_font_medium, text_color)

        img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Logo
        logo = self._get_scaled_logo(img_with_text.shape[0])
        if logo is not None:
            self._overlay_rgba(img_with_text, logo, self.width - logo.shape[1] - 30, 20)

        return img_with_text

    def render_idle(self, img, main_text, sub_text):
        return self._render_base(img, main_text, sub_text)

    def render_greeting(self, img, lines):
        return self._render_base(img, *lines)

    def render_qr_panel(self, img, url):
        if url not in self.qr_cache:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4
            )
            qr.add_data(url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
            self.qr_cache[url] = cv2.cvtColor(np.array(qr_img), cv2.COLOR_RGB2BGR)

        qr_code_img = self.qr_cache[url]
        qr_h, qr_w = qr_code_img.shape[:2]

        panel_w, panel_h = int(self.width * 0.8), int(self.height * 0.7)
        panel_x, panel_y = (self.width - panel_w) // 2, (self.height - panel_h) // 2

        overlay = np.full((panel_h, panel_w, 3), (250, 250, 250), dtype=np.uint8)
        qr_x, qr_y = (panel_w - qr_w) // 2, (panel_h - qr_h) // 2
        overlay[qr_y:qr_y + qr_h, qr_x:qr_x + qr_w] = qr_code_img

        cv2.addWeighted(
            overlay, 0.9,
            img[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w], 0.1,
            0,
            img[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w]
        )

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        title_text = "¡Conoce más sobre la carrera!"
        if self.font_medium:
            try:
                bbox = self.font_medium.getbbox(title_text)
                text_w = bbox[2] - bbox[0]
            except Exception:
                text_w = int(self.font_medium.getlength(title_text)) if hasattr(self.font_medium, 'getlength') else 600
            draw.text(((self.width - text_w) // 2, panel_y + 30), title_text, font=self.font_medium, fill=(0, 0, 0))

        img2 = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        logo = self._get_scaled_logo(img2.shape[0])
        if logo is not None:
            self._overlay_rgba(img2, logo, self.width - logo.shape[1] - 30, 20)

        return img2

    # ---------------------- Utilidades varias ----------------------

    def _overlay_rgba(self, bg, fg, x, y):
        if fg is None:
            return
        h, w = fg.shape[:2]
        if x >= bg.shape[1] or y >= bg.shape[0] or x + w <= 0 or y + h <= 0:
            return

        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(bg.shape[1], x + w)
        y_end = min(bg.shape[0], y + h)

        fg_x_start = max(0, -x)
        fg_y_start = max(0, -y)
        fg_x_end = fg_x_start + (x_end - x_start)
        fg_y_end = fg_y_start + (y_end - y_start)

        if fg.shape[2] == 4:  # con alfa
            alpha = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end, 3:4] / 255.0
            bg_roi = bg[y_start:y_end, x_start:x_end]
            fg_roi = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end, :3]
            bg[y_start:y_end, x_start:x_end] = (alpha * fg_roi + (1 - alpha) * bg_roi).astype(bg.dtype)
        else:
            bg[y_start:y_end, x_start:x_end] = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end]

    def _hex_to_bgr(self, hx):
        hx = hx.strip().lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
