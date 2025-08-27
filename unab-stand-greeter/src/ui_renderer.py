# src/ui_renderer.py
import cv2
import qrcode
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import emoji
import platform


class UIRenderer:
    def __init__(self, width, height, brand_primary, text_color, assets, show_qr_panel=False):
        self.width = width
        self.height = height
        self.brand_primary_bgr = self._hex_to_bgr(brand_primary)
        self.text_color_rgb = tuple(reversed(self._hex_to_bgr(text_color)))
        self.show_qr_panel = show_qr_panel
        self.qr_cache = {}
        self.system = platform.system()

        # Cargar fuentes
        self.logo = self._load_logo(assets.get("logo"))

        # Inicializar fuentes como None por defecto
        self.font_large = None
        self.font_medium = None
        self.font_small = None
        self.emoji_font_large = None
        self.emoji_font_medium = None

        # Cargar fuente principal
        font_path = assets.get("font_path")
        if font_path and font_path.exists():
            try:
                self.font_large = ImageFont.truetype(str(font_path), 68)
                self.font_medium = ImageFont.truetype(str(font_path), 42)
                self.font_small = ImageFont.truetype(str(font_path), 24)
                print(f"INFO: Fuente principal cargada desde {font_path}")
            except Exception as e:
                print(f"ADVERTENCIA: No se pudo cargar la fuente principal: {e}")

        # Cargar fuente de emojis con manejo específico por SO
        emoji_font_path = assets.get("emoji_font_path")
        if emoji_font_path and emoji_font_path.exists():
            self._load_emoji_fonts(emoji_font_path)
        else:
            print("ADVERTENCIA: No se encontró la fuente de emojis")

    def _load_emoji_fonts(self, emoji_font_path):
        """Carga fuentes de emojis con configuraciones específicas por SO."""
        print(f"INFO: Intentando cargar fuente de emojis desde {emoji_font_path} en {self.system}")

        # Configuraciones por sistema operativo
        if self.system == "Darwin":  # macOS
            # macOS maneja mejor las fuentes TTC y requiere tamaños específicos
            font_sizes_large = [68, 64, 60, 56, 48]
            font_sizes_medium = [42, 40, 36, 32, 28]
            font_index = 0  # Para fuentes TTC, usar el primer índice
        elif self.system == "Windows":
            # Windows con Segoe UI Emoji o NotoColorEmoji
            font_sizes_large = [72, 68, 64, 60, 56]
            font_sizes_medium = [44, 42, 40, 36, 32]
            font_index = 0
        else:  # Linux y otros
            font_sizes_large = [68, 64, 60, 56, 48, 40]
            font_sizes_medium = [42, 40, 36, 32, 28, 24]
            font_index = 0

        # Intentar cargar fuente grande
        for size in font_sizes_large:
            try:
                if str(emoji_font_path).endswith('.ttc'):
                    # Para fuentes TTC (principalmente macOS)
                    self.emoji_font_large = ImageFont.truetype(str(emoji_font_path), size, index=font_index)
                else:
                    # Para fuentes TTF regulares
                    self.emoji_font_large = ImageFont.truetype(str(emoji_font_path), size)
                print(f"INFO: Fuente de emojis grande cargada con tamaño {size}")
                break
            except Exception as e:
                print(f"DEBUG: Tamaño grande {size} falló: {e}")
                continue

        # Intentar cargar fuente mediana
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

        # Verificar que al menos una fuente se cargó
        if self.emoji_font_large is None and self.emoji_font_medium is None:
            print("ADVERTENCIA: No se pudo cargar ninguna fuente de emojis")
            self._try_fallback_emoji_fonts()
        else:
            # Verificar que la fuente funciona probando un emoji
            self._test_emoji_font()

    def _try_fallback_emoji_fonts(self):
        """Intenta cargar fuentes de emojis alternativas del sistema."""
        print("INFO: Intentando fuentes de emojis alternativas del sistema...")

        fallback_fonts = []

        if self.system == "Windows":
            fallback_fonts = [
                "C:/Windows/Fonts/segmdl2.ttf",  # Segoe MDL2 Assets
                "arial.ttf"  # Arial como último recurso
            ]
        elif self.system == "Darwin":
            fallback_fonts = [
                "/System/Library/Fonts/Helvetica.ttc"
            ]
        else:  # Linux
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
                except:
                    continue

        print("ADVERTENCIA: No se pudo cargar ninguna fuente alternativa")

    def _test_emoji_font(self):
        """Verifica que la fuente de emojis funciona correctamente."""
        test_font = self.emoji_font_large or self.emoji_font_medium
        if test_font:
            try:
                test_img = Image.new('RGB', (100, 100), (255, 255, 255))
                test_draw = ImageDraw.Draw(test_img)
                test_draw.text((10, 10), "👋", font=test_font, embedded_color=True)
                print(f"INFO: Fuente de emojis verificada correctamente en {self.system}")
            except Exception as e:
                print(f"ADVERTENCIA: Error al verificar fuente de emojis en {self.system}: {e}")

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
        """Dibuja texto con manejo específico de emojis por SO."""
        if not text:
            return

        x, y = start_xy

        for i, char in enumerate(text):
            if emoji.is_emoji(char) and emoji_font:
                # Dibujar emoji con configuraciones específicas por SO
                try:
                    if self.system == "Windows":
                        # En Windows, a veces necesitamos ajustar la posición Y
                        emoji_y = y - 5
                    else:
                        emoji_y = y

                    draw.text((x, emoji_y), char, font=emoji_font, embedded_color=True)

                    # Obtener ancho del emoji
                    try:
                        bbox = emoji_font.getbbox(char)
                        char_width = bbox[2] - bbox[0]
                    except:
                        # Fallback si getbbox falla
                        char_width = emoji_font.getlength(char) if hasattr(emoji_font, 'getlength') else 30

                    x += max(char_width, 20)  # Mínimo 20px de ancho

                except Exception as e:
                    print(f"ERROR: Error dibujando emoji '{char}' en {self.system}: {e}")
                    # Fallback: mostrar texto alternativo
                    if font:
                        fallback_text = "[emoji]"
                        draw.text((x, y), fallback_text, font=font, fill=fill)
                        x += font.getlength(fallback_text) if hasattr(font, 'getlength') else 50
                    else:
                        x += 30
            else:
                # Dibujar carácter normal
                if font:
                    draw.text((x, y), char, font=font, fill=fill)
                    try:
                        bbox = font.getbbox(char)
                        char_width = bbox[2] - bbox[0]
                    except:
                        char_width = font.getlength(char) if hasattr(font, 'getlength') else 10
                    x += char_width

    def _render_base(self, img, *lines):
        """Renderiza las líneas de texto base con soporte para emojis."""
        # Convertir imagen a PIL en modo RGB (sin alfa por ahora)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Colores para texto y sombra
        text_color = self.text_color_rgb
        shadow_color = (0, 0, 0)

        print(f"DEBUG: Renderizando {len(lines)} líneas de texto")

        # Línea principal (grande)
        if len(lines) > 0 and lines[0]:
            main_text = str(lines[0])
            print(f"DEBUG: Línea principal: '{main_text}'")
            # Sombra
            if self.font_large:
                self._draw_text_with_emojis(draw, main_text, (42, 32), self.font_large, self.emoji_font_large, shadow_color)
                # Texto principal
                self._draw_text_with_emojis(draw, main_text, (40, 30), self.font_large, self.emoji_font_large, text_color)

        # Subtítulo (mediano)
        if len(lines) > 1 and lines[1]:
            sub_text = str(lines[1])
            print(f"DEBUG: Subtítulo: '{sub_text}'")
            # Sombra
            if self.font_medium:
                self._draw_text_with_emojis(draw, sub_text, (42, self.height - 78), self.font_medium, self.emoji_font_medium, shadow_color)
                # Texto principal
                self._draw_text_with_emojis(draw, sub_text, (40, self.height - 80), self.font_medium, self.emoji_font_medium, text_color)

        # Convertir de vuelta a OpenCV
        img_with_text = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Superponer logo
        if self.logo is not None:
            self._overlay_rgba(img_with_text, self.logo, self.width - self.logo.shape[1] - 30, 20)

        return img_with_text

    def render_idle(self, img, main_text, sub_text):
        return self._render_base(img, main_text, sub_text)

    def render_greeting(self, img, lines):
        return self._render_base(img, *lines)

    def render_qr_panel(self, img, url):
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

        # Usar Pillow para el texto del título (sin emojis en este caso)
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        title_text = "¡Conoce más sobre la carrera!"
        if self.font_medium:
            bbox = self.font_medium.getbbox(title_text)
            text_w = bbox[2] - bbox[0]
            draw.text(((self.width - text_w) // 2, panel_y + 30), title_text, font=self.font_medium, fill=(0, 0, 0))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _overlay_rgba(self, bg, fg, x, y):
        if fg is None: return
        h, w = fg.shape[:2]

        # Asegurarse de que las coordenadas estén dentro de los límites
        if x >= bg.shape[1] or y >= bg.shape[0] or x + w <= 0 or y + h <= 0:
            return

        # Recortar si es necesario
        x_start = max(0, x)
        y_start = max(0, y)
        x_end = min(bg.shape[1], x + w)
        y_end = min(bg.shape[0], y + h)

        # Calcular offsets para la imagen superpuesta
        fg_x_start = max(0, -x)
        fg_y_start = max(0, -y)
        fg_x_end = fg_x_start + (x_end - x_start)
        fg_y_end = fg_y_start + (y_end - y_start)

        if fg.shape[2] == 4:  # Si tiene canal alfa
            alpha = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end, 3] / 255.0
            bg_roi = bg[y_start:y_end, x_start:x_end]
            fg_roi = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end, :3]

            for c in range(3):
                bg_roi[:, :, c] = (alpha * fg_roi[:, :, c] + (1 - alpha) * bg_roi[:, :, c])
        else:
            # Sin canal alfa, copia directa
            bg[y_start:y_end, x_start:x_end] = fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end]

    def _hex_to_bgr(self, hx):
        hx = hx.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))