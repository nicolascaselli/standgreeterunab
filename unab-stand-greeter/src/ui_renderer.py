# ui_renderer.py
import cv2
import qrcode
import numpy as np

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
        logo_path = (self.assets.get("logo") if self.assets else None)
        if logo_path:
            lg = cv2.imread(str(logo_path))
            if lg is not None:
                # Redimensionar logo (ancho 18% pantalla)
                max_w = int(self.width * 0.18)
                if lg.shape[1] > max_w:
                    scale = max_w / lg.shape[1]
                    lg = cv2.resize(lg, (int(lg.shape[1] * scale), int(lg.shape[0] * scale)), cv2.INTER_AREA)
                self._logo = lg

    def render_idle(self, frame, line1, line2):
        """Pantalla idle con dos líneas centradas y logo opcional arriba."""
        img = frame
        h, w = img.shape[:2]
        cy = int(h * 0.42)
        self._put_centered(img, line1, (cy - 10), scale=1.1, weight=2)
        self._put_centered(img, line2, (cy + 50), scale=0.9)
        if self._logo is not None:
            lh, lw = self._logo.shape[:2]
            x = (w - lw) // 2
            y = int(h * 0.08)
            img[y:y + lh, x:x + lw] = self._logo
        return img

    def render_greeting(self, frame, lines):
        """Pantalla de saludo; lines es lista de 1-3 líneas."""
        img = frame
        h, _ = img.shape[:2]
        base = int(h * 0.40)
        for i, text in enumerate(lines[:3]):
            self._put_centered(img, text, base + i * 55, scale=1.0 if i == 0 else 0.9)
        return img

    def render_qr_panel(self, frame, url):
        """Muestra el QR centrado (sin panel decorativo salvo que show_qr_panel=True)."""
        img = frame
        # Cache / generar
        if self._qr_cache_url != url:
            qr = qrcode.QRCode(border=1)
            qr.add_data(url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
            self._qr_np = np.array(qr_img)[:, :, ::-1]  # RGB->BGR
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
            x0, y0 = x - pad, y - pad
            x1, y1 = x + wq + pad, y + hq + pad
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(img, (x0, y0), (x1, y1), self._primary_bgr, 3, cv2.LINE_AA)

        img[y:y + hq, x:x + wq] = qr_bgr
        self._put_centered(img, "Escanea el c\u00F3digo", y + hq + 40, scale=0.9)
        return img

    # ---------- Helpers ----------
    def _put_centered(self, img, text, cy, scale=1.0, color=None, weight=2):
        if not text:
            return
        color = color or self._text_color_bgr
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, weight)
        x = (img.shape[1] - tw) // 2
        y = cy
        # Sombra
        cv2.putText(img, text, (x, y), font, scale, (0, 0, 0), weight + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, scale, color, weight, cv2.LINE_AA)

    def _hex_to_bgr(self, hx):
        hx = hx.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))