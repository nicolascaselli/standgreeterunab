import cv2
import numpy as np
from PIL import Image, ImageDraw
import qrcode


def hex_to_bgr(hex_color):
    """Convierte #RRGGBB a (B, G, R)."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


# ui_renderer.py
class UIRenderer:
    def __init__(self, width, height, brand_primary="#A00321", text_color="#FFFFFF", assets=None,
                 show_qr_panel=False):
        self.width = width
        self.height = height
        self.brand_primary = brand_primary
        self.text_color = text_color
        self.assets = assets or {}
        self.show_qr_panel = show_qr_panel
        # ... resto init (cache QR, fuentes, etc.)

    def render_qr_panel(self, frame, url):
        """
        Muestra sólo el QR sin panel de fondo (si show_qr_panel es False).
        """
        # Generar / obtener QR (suponiendo cache previa)
        if not hasattr(self, "_qr_cache_url") or self._qr_cache_url != url:
            import qrcode
            qr = qrcode.QRCode(border=1)  # border pequeño sólo del propio QR
            qr.add_data(url)
            qr.make(fit=True)
            qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
            import numpy as np
            self._qr_np = np.array(qr_img)[:, :, ::-1]  # a BGR
            self._qr_cache_url = url

        qr_bgr = self._qr_np
        # Redimensionar si se requiere
        max_side = int(self.height * 0.42)
        h, w = qr_bgr.shape[:2]
        scale = min(max_side / max(h, w), 1.0)
        if scale != 1.0:
            qr_bgr = cv2.resize(qr_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = qr_bgr.shape[:2]

        # Calcular posición (centrado inferior)
        margin = 30
        x = (self.width - w) // 2
        y = (self.height - h) // 2

        if self.show_qr_panel:
            # (Opcional) panel si se reactiva
            pad = 20
            x0, y0 = x - pad, y - pad
            x1, y1 = x + w + pad, y + h + pad
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), self._hex_to_bgr(self.brand_primary), 3)

        # Pegar QR (sin fondo adicional)
        frame[y:y + h, x:x + w] = qr_bgr

        # (Opcional) texto descriptivo (sin fondo)
        label = "Escanea el c\u00F3digo"
        cv2.putText(frame, label, (x + 4, y + h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        return frame

    def _hex_to_bgr(self, hx):
        hx = hx.lstrip("#")
        return (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))