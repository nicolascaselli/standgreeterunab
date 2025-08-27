# src/ui_renderer.py
import cv2
import qrcode
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class UIRenderer:
    def __init__(self, width, height, brand_primary="#A00321", text_color="#FFFFFF", assets=None,
                 show_qr_panel=False):
        self.width = width
        self.height = height
        self.brand_primary = brand_primary
        self.text_color = text_color
        self.assets = assets or {}
        self.show_qr_panel = show_qr_panel
        self._qr_cache_url = None
        self._qr_np = None
        self._text_color_bgr = self._hex_to_bgr(self.text_color)
        self._primary_bgr = self._hex_to_bgr(self.brand_primary)
        self._logo = None

        # Cargar fuente TTF para texto con tildes
        self.font_regular = None
        self.font_bold = None
        font_path = self.assets.get("font_path")
        if font_path and font_path.exists():
            try:
                self.font_regular = ImageFont.truetype(str(font_path), 42)
                self.font_bold = ImageFont.truetype(str(font_path), 48)  # Usar un tamaño mayor para "negrita"
            except IOError as e:
                print(f"ADVERTENCIA: No se pudo cargar la fuente TTF: {e}")

        logo_path = self.assets.get("logo")
        if logo_path:
            lg = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
            if lg is not None:
                max_w = int(self.width * 0.18)
                if lg.shape[1] > max_w:
                    scale = max_w / lg.shape[1]
                    lg = cv2.resize(lg, (int(lg.shape[1] * scale), int(lg.shape[0] * scale)), cv2.INTER_AREA)
                self._logo = lg

    def render_idle(self, frame, line1, line2):
        img = frame
        h, w = img.shape[:2]
        cy = int(h * 0.42)

        # Usar el nuevo método para dibujar texto
        img = self._put_centered_pil(img, line1, (cy - 10), font=self.font_bold)
        img = self._put_centered_pil(img, line2, (cy + 50), font=self.font_regular)

        if self._logo is not None:
            lh, lw = self._logo.shape[:2]
            margin = 30
            x = w - lw - margin
            y = margin
            self._overlay_rgba(img, self._logo, x, y)
        return img

    def render_greeting(self, frame, lines):
        img = frame
        h, _ = img.shape[:2]
        base = int(h * 0.40)
        for i, text in enumerate(lines[:3]):
            font = self.font_bold if i == 0 else self.font_regular
            img = self._put_centered_pil(img, text, base + i * 55, font=font)
        return img

    def render_qr_panel(self, frame, url):
        img = frame
        if self._qr_cache_url != url:
            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
            self._qr_np = np.array(qr_img)[:, :, ::-1]
            self._qr_cache_url = url

        qr_bgr = self._qr_np.copy()
        max_side = int(self.height * 0.42)
        hq, wq = qr_bgr.shape[:2]
        scale = min(max_side / max(hq, wq), 1.0)
        if scale != 1.0:
            qr_bgr = cv2.resize(qr_bgr, (int(wq * scale), int(hq * scale)), interpolation=cv2.INTER_AREA)
            hq, wq = qr_bgr.shape[:2]

        x = (self.width - wq) // 2
        y = (self.height - hq) // 2

        if self.show_qr_panel:
            pad = 20
            x0, y0, x1, y1 = x - pad, y - pad, x + wq + pad, y + hq + pad
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(img, (x0, y0), (x1, y1), self._primary_bgr, 3, cv2.LINE_AA)

        img[y:y + hq, x:x + wq] = qr_bgr
        img = self._put_centered_pil(img, "Escanea el código", y + hq + 40, font=self.font_regular)
        return img

    # ---------- Helpers ----------
    def _pil_to_cv2(self, pil_img):
        return np.array(pil_img)[:, :, ::-1]

    def _cv2_to_pil(self, cv2_img):
        return Image.fromarray(cv2_img[:, :, ::-1])

    def _put_centered_pil(self, img_cv2, text, cy, font, color=None):
        if not text or not font:
            # Si no hay fuente, recurrir al método antiguo (sin tildes)
            return self._put_centered_cv2(img_cv2, text, cy)

        color = color or self._text_color_bgr
        pil_color = (color[2], color[1], color[0])  # BGR a RGB

        img_pil = self._cv2_to_pil(img_cv2)
        draw = ImageDraw.Draw(img_pil)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        x = (self.width - tw) // 2
        y = cy - th // 2

        # Sombra
        draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
        # Texto
        draw.text((x, y), text, font=font, fill=pil_color)

        return self._pil_to_cv2(img_pil)

    def _put_centered_cv2(self, img, text, cy, scale=1.0, color=None, weight=2):
        """Fallback a cv2.putText si la fuente TTF no está disponible."""
        if not text: return img
        color = color or self._text_color_bgr
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font_face, scale, weight)
        x = (img.shape[1] - tw) // 2
        cv2.putText(img, text, (x, cy), font_face, scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, cy), font_face, scale, color, weight, cv2.LINE_AA)
        return img

    def _overlay_rgba(self, bg_bgr, fg_bgra, x, y):
        # (Sin cambios, este método ya funciona bien)
        if fg_bgra is None: return bg_bgr
        H, W = bg_bgr.shape[:2];
        h, w = fg_bgra.shape[:2]
        if x >= W or y >= H: return bg_bgr
        x2, y2 = min(x + w, W), min(y + h, H)
        x0_fg, y0_fg = max(0, -x), max(0, -y)
        x0_bg, y0_bg = max(0, x), max(0, y)
        w_eff, h_eff = x2 - x0_bg, y2 - y0_bg
        if w_eff <= 0 or h_eff <= 0: return bg_bgr
        fg_crop = fg_bgra[y0_fg:y0_fg + h_eff, x0_fg:x0_fg + w_eff]
        bg_roi = bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff]
        if fg_crop.shape[2] == 4:
            alpha = fg_crop[:, :, 3:4] / 255.0
            blended = ((1 - alpha) * bg_roi + alpha * fg_crop[:, :, :3]).astype(bg_roi.dtype)
            bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff] = blended
        else:
            bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff] = fg_crop[:, :, :3]
        return bg_bgr

    def _hex_to_bgr(self, hx):
        hx = hx.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))