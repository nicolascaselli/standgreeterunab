# src/ui_renderer.py
import cv2
import qrcode
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UIRenderer:
    def __init__(self, width, height, brand_primary, text_color, assets, show_qr_panel=False):
        self.width = width
        self.height = height
        self.brand_primary_bgr = self._hex_to_bgr(brand_primary)
        self.text_color_rgb = tuple(reversed(self._hex_to_bgr(text_color)))
        self.show_qr_panel = show_qr_panel
        self.qr_cache = {}

        # Cargar fuentes
        self.logo = self._load_logo(assets.get("logo"))
        self.emoji_font = assets.get("emoji_font")
        self.main_font = None
        # Inicializar atributos de fuente de emojis
        self.emoji_font_large = None
        self.emoji_font_medium = None

        font_path = assets.get("font_path")
        if font_path and font_path.exists():
            try:
                # Tamaños de fuente para diferentes textos
                self.font_large = ImageFont.truetype(str(font_path), 68)
                self.font_medium = ImageFont.truetype(str(font_path), 42)
                self.font_small = ImageFont.truetype(str(font_path), 24)
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo cargar la fuente principal: {e}")
                self.font_large = self.font_medium = self.font_small = None

        # Ajustar el tamaño de la fuente de emojis para que coincida con la fuente principal
        if self.emoji_font:
            self.emoji_font_large = self.emoji_font.font_variant(size=58)
            self.emoji_font_medium = self.emoji_font.font_variant(size=36)

    def _load_logo(self, logo_path):
        if not logo_path: return None
        logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if logo is None: return None
        # Redimensionar logo a una altura fija, manteniendo la proporción
        h, w = logo.shape[:2]
        new_h = int(self.height * 0.08)
        scale = new_h / h
        new_w = int(w * scale)
        return cv2.resize(logo, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _draw_text_with_emojis(self, draw, text, start_xy, font, emoji_font, fill):
        """Dibuja texto, cambiando a la fuente de emojis cuando es necesario."""
        if not font or not emoji_font:  # Fallback si las fuentes no cargaron
            draw.text(start_xy, text, font=font, fill=fill)
            return

        x, y = start_xy
        for char in text:
            if char in emoji_font.get_variation_names():
                # Usar la fuente de emojis y ajustar la posición verticalmente
                draw.text((x, y - 10), char, font=emoji_font, fill=fill, embedded_color=True)
                # Usamos getbbox en lugar del obsoleto getsize
                bbox = emoji_font.getbbox(char)
                x += bbox[2] - bbox[0]  # Ancho del caracter
            else:
                # Usar la fuente de texto normal
                draw.text((x, y), char, font=font, fill=fill)
                bbox = font.getbbox(char)
                x += bbox[2] - bbox[0]

    def _render_base(self, img, main_text, sub_text=""):
        # 1. (Opcional) Ya no dibujamos el banner rojo. La línea está comentada/eliminada.
        # cv2.rectangle(img, (0, 0), (self.width, int(self.height * 0.12)), self.brand_primary_bgr, -1)

        # 2. Convertimos la imagen de la cámara a formato Pillow para dibujar el texto.
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Dibujamos el texto principal con soporte para emojis.
        # Usamos un color de texto que contraste con el fondo del video (ej. blanco con sombra negra).
        text_color_with_alpha = self.text_color_rgb + (255,)
        if self.font_large and self.emoji_font_large:
            # Dibujar una sombra para mejorar la legibilidad
            shadow_color = (0, 0, 0, 255)
            self._draw_text_with_emojis(draw, main_text, (42, 32), self.font_large, self.emoji_font_large, shadow_color)
            # Dibujar el texto principal
            self._draw_text_with_emojis(draw, main_text, (40, 30), self.font_large, self.emoji_font_large, text_color_with_alpha)

        # Dibujamos el subtítulo.
        if sub_text and self.font_medium:
            # Sombra para el subtítulo
            draw.text((42, self.height - 78), sub_text, font=self.font_medium, fill=(0, 0, 0))
            # Texto del subtítulo
            draw.text((40, self.height - 80), sub_text, font=self.font_medium, fill=self.text_color_rgb)

        # 3. Convertimos la imagen con texto de vuelta a formato OpenCV.
        img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # 4. Superponemos el logo AL FINAL, para que quede encima de todo.
        if self.logo is not None:
            # Usamos la función _overlay_rgba que ya tienes para manejar la transparencia.
            self._overlay_rgba(img_with_text, self.logo, self.width - self.logo.shape[1] - 30, 20)

        return img_with_text

    def render_idle(self, img, main_text, sub_text):
        return self._render_base(img, main_text, sub_text)

    def render_greeting(self, img, lines):
        main_text = lines[0] if lines else ""
        sub_text = lines[1] if len(lines) > 1 else ""

        # Convierte a Pillow para dibujar texto
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Dibuja texto principal
        if self.font_large:
            draw.text((40, 30), main_text, font=self.font_large, fill=self.text_color_rgb)

        # Dibuja subtítulo con soporte para emojis
        if sub_text and self.font_medium and self.emoji_font_medium:
            self._draw_text_with_emojis(draw, sub_text, (40, self.height - 80), self.font_medium,
                                        self.emoji_font_medium, self.text_color_rgb)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def render_qr_panel(self, img, url):
        # ... (El resto de la clase no necesita cambios)
        if url not in self.qr_cache:
            qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=4)
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

        cv2.addWeighted(overlay, 0.9, img[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w], 0.1, 0,
                        img[panel_y:panel_y + panel_h, panel_x:panel_x + panel_w])

        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        title_text = "¡Conoce más sobre la carrera!"
        bbox = self.font_medium.getbbox(title_text)
        text_w = bbox[2] - bbox[0]
        draw.text(((self.width - text_w) // 2, panel_y + 30), title_text, font=self.font_medium, fill=(0, 0, 0))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _overlay_rgba(self, bg, fg, x, y):
        if fg is None: return
        h, w = fg.shape[:2]

        # Asegurarse de que la imagen de fondo tenga 3 canales (BGR)
        if bg.shape[2] == 4:
            bg = bg[:, :, :3]

        # Asegurarse de que la imagen superpuesta tenga 4 canales (BGRA)
        if fg.shape[2] == 3:
            # Si no tiene canal alfa, no se puede hacer la superposición transparente
            return

        alpha = fg[:, :, 3] / 255.0
        bg_roi = bg[y:y+h, x:x+w]

        for c in range(0, 3):
            bg_roi[:, :, c] = (alpha * fg[:, :, c] + (1 - alpha) * bg_roi[:, :, c])

    def _hex_to_bgr(self, hx):
        hx = hx.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))