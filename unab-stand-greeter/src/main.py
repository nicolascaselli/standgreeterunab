# -*- coding: utf-8 -*-
"""
UNAB Stand Greeter (con HUD de reconocimientos):
- Detección de saludo (hola) y pulgar arriba con panel en vivo
- UI + TTS + QR + Logo
Controles:
  q: salir | m: mute | f: fullscreen | r: reset
"""
import cv2, time, yaml, numpy as np
from pathlib import Path
import os
import urllib.request
import ssl
from PIL import ImageFont, Image, ImageDraw
import platform

# --- Definir rutas base para que funcione independientemente del CWD ---
# Ruta al directorio del script actual (src)
SCRIPT_DIR = Path(__file__).resolve().parent
# Ruta al directorio raíz del proyecto (un nivel arriba de src)
PROJECT_ROOT = SCRIPT_DIR.parent

from detector import Detector
from wave_gesture import WaveDetector
from thumbs_gesture import ThumbsUpDetector
from ui_renderer import UIRenderer
from tts import TTS
from metrics import Metrics
import re


STATE_IDLE, STATE_GREETING, STATE_WAIT_THUMBS, STATE_SHOW_QR, STATE_COOLDOWN = range(5)


def draw_landmarks_basic(img, pose, width, height):
    """Dibuja hombros y muñecas detectadas."""
    # Asegura que la imagen tenga un layout de memoria compatible con OpenCV
    img = np.ascontiguousarray(img, dtype=np.uint8)

    key_color = (50, 200, 255)
    for k in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist"):
        x, y = pose.get(k, (None, None))
        if x is None: continue
        cx, cy = int(x * width), int(y * height)
        cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, key_color, -1, cv2.LINE_AA)
    # Devuelve la imagen modificada
    return img

# ---------- Utilidades HUD / overlay ----------
def overlay_alpha(img, overlay, x, y, alpha=0.6):
    """Pega un rectángulo BGR con alpha sobre img."""
    h, w = overlay.shape[:2]
    H, W = img.shape[:2]
    if x >= W or y >= H:
        return img
    x2, y2 = min(x + w, W), min(y + h, H)
    roi = img[y:y2, x:x2]
    ov = overlay[:(y2 - y), :(x2 - x)]
    cv2.addWeighted(ov, alpha, roi, 1 - alpha, 0, roi)
    img[y:y2, x:x2] = roi
    return img


def draw_text(img, text, org, font, color=(255, 255, 255)):
    """Dibuja texto con tildes usando Pillow sobre una imagen de OpenCV."""
    # Convierte la imagen de OpenCV (BGR) a una imagen de Pillow (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)

    # Dibuja la sombra del texto
    draw.text((org[0] + 1, org[1] + 1), text, font=font, fill=(0, 0, 0))
    # Dibuja el texto principal
    draw.text(org, text, font=font, fill=color)

    # Convierte la imagen de Pillow de vuelta a formato OpenCV y la retorna
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_bool_dot(img, center, ok):
    color = (40, 180, 60) if ok else (40, 40, 200)
    cv2.circle(img, center, 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, center, 6, color, -1, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, ratio, color=(40, 180, 60)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
    fill = int(max(0.0, min(1.0, ratio)) * (w - 4))
    cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill, y + h - 2), color, -1, cv2.LINE_AA)


# ---------- Emoji PNG helper (ya lo tenías) ----------
def _read_rgba(pathlike):
    p = Path(pathlike)
    if not p.exists():
        p2 = Path.cwd() / p
        if p2.exists():
            p = p2
        else:
            return None
    return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)


def _resize(img, width=None, height=None):
    if img is None: return None
    h, w = img.shape[:2]
    if width is None and height is None: return img
    if width is None:
        scale = height / float(h);
        width = int(w * scale)
    elif height is None:
        scale = width / float(w);
        height = int(h * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def overlay_rgba(bg_bgr, fg_bgra, x, y):
    if fg_bgra is None: return bg_bgr
    H, W = bg_bgr.shape[:2];
    h, w = fg_bgra.shape[:2]
    if x >= W or y >= H: return bg_bgr
    x2, y2 = min(x + w, W), min(y + h, H)
    if x2 <= 0 or y2 <= 0: return bg_bgr
    x0_fg, y0_fg = max(0, -x), max(0, -y)
    x0_bg, y0_bg = max(0, x), max(0, y)
    w_eff, h_eff = x2 - x0_bg, y2 - y0_bg
    fg_crop = fg_bgra[y0_fg:y0_fg + h_eff, x0_fg:x0_fg + w_eff]
    bg_roi = bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff]
    if fg_crop.shape[2] == 4:
        alpha = fg_crop[:, :, 3:4] / 255.0
        bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff] = ((1 - alpha) * bg_roi + alpha * fg_crop[:, :, :3]).astype(
            bg_roi.dtype)
    else:
        bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff] = fg_crop[:, :, :3]
    return bg_bgr


def get_pose_item(pose, key):
    return pose.get(key, (None, None))


def compute_panel_xy(anchor, panel_w, panel_h, width, height, margin_xy):
    """Devuelve (x,y) de la esquina superior izquierda del panel según el anchor y margen."""
    ax = (anchor or "bl").lower()
    mx, my = margin_xy
    # X
    if "l" in ax:
        x = mx
    elif "r" in ax:
        x = width - panel_w - mx
    else:  # 'c' center
        x = (width - panel_w) // 2
    # Y
    if "t" in ax:
        y = my
    elif "b" in ax:
        y = height - panel_h - my
    else:  # 'c' center (tc/bc ya cubiertos con t/b)
        y = (height - panel_h) // 2
    return x, y


def setup_emoji_font():
    """Verifica y carga la fuente de emojis según el sistema operativo."""
    system = platform.system()

    # Rutas de fuentes según el sistema operativo
    if system == "Darwin":  # macOS
        system_fonts = [
            "/System/Library/Fonts/Apple Color Emoji.ttc",
            "/Library/Fonts/Apple Color Emoji.ttc"
        ]
    elif system == "Windows":
        system_fonts = [
            "C:/Windows/Fonts/seguiemj.ttf",  # Segoe UI Emoji
            "C:/Windows/Fonts/NotoColorEmoji.ttf"
        ]
    else:  # Linux y otros
        system_fonts = [
            "/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
            "/usr/share/fonts/noto-color-emoji/NotoColorEmoji.ttf",
            "/system/fonts/NotoColorEmoji.ttf"
        ]

    # Intentar usar fuentes del sistema primero
    for font_path in system_fonts:
        if Path(font_path).exists():
            print(f"INFO: Usando fuente de emojis del sistema: {font_path}")
            return Path(font_path)

    print(f"INFO: No se encontraron fuentes de emojis del sistema en {system}")

    # Fallback a NotoColorEmoji descargada
    font_path = PROJECT_ROOT / "src/assets/fonts/NotoColorEmoji.ttf"
    font_url = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"

    if not font_path.exists():
        print("INFO: Intentando descargar NotoColorEmoji...")
        try:
            font_path.parent.mkdir(parents=True, exist_ok=True)

            # Crear un contexto SSL que no verifique certificados
            context = ssl._create_unverified_context()

            # Agregar headers para evitar bloqueos
            req = urllib.request.Request(font_url)
            req.add_header('User-Agent', 'Mozilla/5.0 (compatible; emoji-downloader/1.0)')

            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                if response.status == 200:
                    with open(font_path, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"INFO: Fuente descargada y guardada en '{font_path}'")
                else:
                    print(f"ERROR: HTTP {response.status} al descargar la fuente")
                    return None

        except Exception as e:
            print(f"ADVERTENCIA: No se pudo descargar la fuente de emojis: {e}")
            return None
    else:
        print(f"INFO: Fuente de emojis encontrada en '{font_path}'")

    # Verificar que el archivo descargado es válido
    if font_path.exists() and font_path.stat().st_size > 1000:  # Al menos 1KB
        return font_path
    else:
        print("ERROR: La fuente descargada parece estar corrupta")
        return None


def main():
    # Buscar config.yaml en la raíz del proyecto y en el directorio src
    config_path_root = PROJECT_ROOT / "config.yaml"
    config_path_src = SCRIPT_DIR / "config.yaml"

    if config_path_root.exists():
        config_path = config_path_root
    elif config_path_src.exists():
        config_path = config_path_src
    else:
        # Si no se encuentra, lanza un error claro.
        raise FileNotFoundError(
            f"No se pudo encontrar 'config.yaml'. Se buscó en: \n- {config_path_root}\n- {config_path_src}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_index = cfg.get("camera_index", 0)
    width, height = cfg.get("resolution", [1280, 720])
    fullscreen = cfg.get("fullscreen", True)
    debug_cfg = cfg.get("debug", {}) or {}
    SHOW_PANEL = bool(debug_cfg.get("show_panel", True))
    DRAW_LANDMARKS = bool(debug_cfg.get("draw_landmarks", True))

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    det = Detector()
    wave_cfg = cfg.get("wave", {})
    wave_left = WaveDetector(wave_cfg.get("window_seconds", 1.5),
                             wave_cfg.get("amplitude_threshold", 0.04),
                             wave_cfg.get("min_peaks", 3))
    wave_right = WaveDetector(wave_cfg.get("window_seconds", 1.5),
                              wave_cfg.get("amplitude_threshold", 0.04),
                              wave_cfg.get("min_peaks", 3))

    thumbs_cfg = cfg.get("thumbs", {})
    thumbs = ThumbsUpDetector(thumbs_cfg.get("tip_above_wrist_margin", 0.02),
                              thumbs_cfg.get("other_fingers_folded", True))

    tts_cfg = cfg.get("tts", {})
    tts = TTS(enabled=tts_cfg.get("enabled", True),
              rate=tts_cfg.get("rate", 175),
              volume=float(tts_cfg.get("volume", 1.0)))
    metrics = Metrics(cfg.get("metrics", {}).get("csv_path", "metrics.csv"))

    # --- Carga de fuente y assets ---
    emoji_font_path = setup_emoji_font()

    # Cargar fuente principal para textos con tildes
    font_path_abs = PROJECT_ROOT / "src/assets/fonts/Roboto-Regular.ttf"
    main_font_panel = None
    if font_path_abs.exists():
        try:
            # Usamos un tamaño más pequeño para el panel de depuración
            main_font_panel = ImageFont.truetype(str(font_path_abs), 16)
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo cargar la fuente principal para el panel: {e}")

    brand = cfg.get("brand", {})
    assets_cfg = cfg.get("assets", {})

    logo_path_str = assets_cfg.get("logo_path")
    logo_path_abs = PROJECT_ROOT / logo_path_str if logo_path_str else None

    ui_assets = {
        "logo": logo_path_abs,
        "emoji_font_path": emoji_font_path,
        "font_path": font_path_abs  # Pasar la ruta de la fuente a UIRenderer
    }

    ui = UIRenderer(width, height,
                    brand_primary=brand.get("primary", "#A00321"),
                    text_color=brand.get("text", "#FFFFFF"),
                    assets=ui_assets,
                    show_qr_panel=False)

    # --- Carga de textos y configuración ---
    idle_text = cfg.get("idle_text", "Acércate y salúdanos")
    greeting_lines = [line for line in cfg.get("greeting_texts", ["¡Hola! 👋", "Bienvenido/a a UNAB"])]
    prompt_wave_text = cfg.get("prompt_wave_text", "Levanta tu mano y saluda 👋")

    # Debug: verificar que los emojis estén en los textos
    print(f"DEBUG: Textos con emojis:")
    print(f"  - greeting_lines[0]: '{greeting_lines[0]}'")
    print(f"  - prompt_wave_text: '{prompt_wave_text}'")

    qr_url = (cfg.get("qr", {}) or {}).get("url", "https://www.unab.cl/carreras/mallas/ing_civil_informatica.pdf")
    wave_cd = cfg.get("cooldowns", {}).get("wave_seconds", 6)
    thumbs_cd = cfg.get("cooldowns", {}).get("thumbs_seconds", 6)

    # Íconos PNG (opcional, como fallback o para otros usos)
    icon_big_h, icon_small_h = int(height * 0.14), int(height * 0.08)
    wave_icon = _resize(_read_rgba(PROJECT_ROOT / "src/assets/emoji/wave.png"), height=icon_big_h)
    thumb_icon = _resize(_read_rgba(PROJECT_ROOT / "src/assets/emoji/thumbs_up.png"), height=icon_big_h)
    mute_icon = _resize(_read_rgba(PROJECT_ROOT / "src/assets/emoji/mute.png"), height=icon_small_h)

    win = "UNAB Greeter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    win_cfg = (cfg.get("window") or {})
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        # aplica tamaño y posición si vienen en config
        w, h = win_cfg.get("size", [width, height])
        cv2.resizeWindow(win, int(w), int(h))
        x, y = win_cfg.get("position", [50, 50])
        try:
            cv2.moveWindow(win, int(x), int(y))
        except Exception:
            pass

        # (Opcional) siempre encima en Windows
        if win_cfg.get("always_on_top", False):
            try:
                import platform
                if platform.system() == "Windows":
                    import win32gui, win32con
                    hwnd = win32gui.FindWindow(None, win)  # por título
                    win32gui.SetWindowPos(
                        hwnd, win32con.HWND_TOPMOST, int(x), int(y), int(w), int(h), 0
                    )
            except Exception as e:
                print("No pude poner la ventana 'always on top':", e)

    state = STATE_IDLE
    last_trigger_t = 0
    muted = not tts_cfg.get("enabled", True)
    wave_count = 0
    thumbs_up_count = 0

    greet_tts = "Hola! Bienvenido a Ingeniería Civil Informática de la UNAB. Si quieres saber más, muéstranos pulgar arriba."
    qr_tts = "Perfecto. Escanea el código QR para conocer la carrera y sus proyectos."

    while True:
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        out = det.process(frame)
        pose = out.get("pose", {})
        hands = out.get("hands", [])
        now = time.time()
        elapsed = now - last_trigger_t

        key = cv2.waitKey(1) & 0xFF
        if key == ord(cfg["keys"].get("quit", "q")): break
        if key == ord(cfg["keys"].get("mute", "m")):
            muted = not muted;
            tts.set_enabled(not muted)
            metrics.log("mute_toggled", f"{not muted}")
        if key == ord(cfg["keys"].get("fullscreen_toggle", "f")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(
                win,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            )
            # Reaplicar tamaño/posición al volver a ventana
            if not fullscreen:
                w, h = win_cfg.get("size", [width, height])
                cv2.resizeWindow(win, int(w), int(h))
                x, y = win_cfg.get("position", [50, 50])
                try:
                    cv2.moveWindow(win, int(x), int(y))
                except Exception:
                    pass
                if win_cfg.get("always_on_top", False):
                    try:
                        if platform.system() == "Windows":
                            import win32gui, win32con
                            hwnd = win32gui.FindWindow(None, win)
                            win32gui.SetWindowPos(
                                hwnd, win32con.HWND_TOPMOST, int(x), int(y), int(w), int(h), 0
                            )
                    except Exception as e:
                        print("No pude poner la ventana 'always on top':", e)

        if key == ord(cfg["keys"].get("reset", "r")):
            state = STATE_IDLE;
            metrics.log("state_reset")

        # --- detección base por estado + render principal ---
        view = frame.copy()  # Usar una copia para no modificar el frame original

        # Pose básica
        ls = pose.get("left_shoulder", (None, None))
        rs = pose.get("right_shoulder", (None, None))
        lw = pose.get("left_wrist", (None, None))
        rw = pose.get("right_wrist", (None, None))
        shoulders_y = [y for (_, y) in (ls, rs) if y is not None]
        person_present = bool(shoulders_y)

        # actualizamos los dos detectores de hola para depurar métricas
        wave_left_trigger = False
        wave_right_trigger = False
        if shoulders_y:
            min_sh_y = min(shoulders_y)
            if lw[0] is not None:
                wave_left_trigger = wave_left.update(lw[0], lw[1], min_sh_y, now)
            if rw[0] is not None:
                wave_right_trigger = wave_right.update(rw[0], rw[1], min_sh_y, now)

        # estado UI
        if state == STATE_IDLE:
            view = ui.render_idle(view, idle_text, prompt_wave_text)
            if wave_left_trigger or wave_right_trigger:
                state = STATE_GREETING
                last_trigger_t = now
                wave_count += 1  # Incrementar contador de saludos
                metrics.log("wave_detected", "left" if wave_left_trigger else "right")
                if not muted: tts.speak_async(greet_tts)

        elif state == STATE_GREETING:
            view = ui.render_greeting(view, greeting_lines)
            if elapsed > 1.2:
                state = STATE_WAIT_THUMBS
                last_trigger_t = now
                metrics.log("state", "WAIT_THUMBS")

        elif state == STATE_WAIT_THUMBS:
            thumbs_text = "Muéstranos Pulgar arriba 👍 para ver más"
            print(f"DEBUG: thumbs_text: '{thumbs_text}'")
            view = ui.render_greeting(view, [greeting_lines[0], thumbs_text])
            thumbs_up = False
            thumbs_dbg_list = []
            for hand in hands:
                dbg = thumbs.debug(hand)
                thumbs_dbg_list.append(dbg)
                if dbg["thumb_up"] and (dbg["folded_all"] or not thumbs.require_folded):
                    thumbs_up = True
                    break

            if thumbs_up:
                state = STATE_SHOW_QR;
                last_trigger_t = now
                thumbs_up_count += 1  # Incrementar contador de pulgares
                metrics.log("thumbs_up", "detected")
                if not muted: tts.speak_async(qr_tts)

            if elapsed > 12:
                state = STATE_IDLE;
                metrics.log("timeout_wait_thumbs")

        elif state == STATE_SHOW_QR:
            view = ui.render_qr_panel(view, qr_url)
            if elapsed > 4.5:
                state = STATE_COOLDOWN;
                last_trigger_t = now;
                metrics.log("qr_shown")

        elif state == STATE_COOLDOWN:
            view = ui.render_idle(view, "¡Gracias! ✨", "Acércate y salúdanos")
            if elapsed > max(wave_cd, thumbs_cd):
                state = STATE_IDLE;
                metrics.log("cooldown_end")

        # --- HUD: landmarks + panel ---
        if DRAW_LANDMARKS:
            view = draw_landmarks_basic(view, pose, width, height)

        if SHOW_PANEL:
            # config del panel
            pconf = (cfg.get("debug", {}) or {}).get("panel", {}) or {}
            anchor = pconf.get("anchor", "bl")
            margin = pconf.get("margin", [20, 20])
            panel_w = int(width * float(pconf.get("width_ratio", 0.36)))
            panel_h = int(height * float(pconf.get("height_ratio", 0.32)))  # Aumentado para los contadores
            alpha = float(pconf.get("alpha", 0.65))

            # recolectar métricas para HUD
            left_dbg = wave_left.get_debug()
            right_dbg = wave_right.get_debug()
            state_name = {0: "IDLE", 1: "GREETING", 2: "WAIT_THUMBS", 3: "SHOW_QR", 4: "COOLDOWN"}[state]

            # fondo panel y posición
            panel = np.full((panel_h, panel_w, 3), (25, 25, 25), dtype=np.uint8)
            x0, y0 = compute_panel_xy(anchor, panel_w, panel_h, width, height, margin)
            view = overlay_alpha(view, panel, x0, y0, alpha=alpha)

            # offsets internos para texto/gráficos
            tx = x0 + 10
            ty = y0 + 10

            if main_font_panel:
                # Título
                view = draw_text(view, "Reconocimiento (en vivo)", (tx, ty + 2), main_font_panel, color=(255, 255, 255))
                # Persona
                draw_bool_dot(view, (tx + 10, ty + 40), person_present)
                view = draw_text(view, f"Persona detectada: {'Si' if person_present else 'No'}", (tx + 30, ty + 36), main_font_panel)
                # hola - izquierda
                lratio = left_dbg["peaks"] / max(1, wave_left.min_peaks)
                draw_bool_dot(view, (tx + 10, ty + 70), left_dbg["hand_raised"])
                view = draw_text(view, f"hola (izq): picos {left_dbg['peaks']}/{wave_left.min_peaks}  x.ptp={left_dbg['x_ptp']:.3f}", (tx + 30, ty + 66), main_font_panel)
                draw_progress_bar(view, tx + 30, ty + 90, int(panel_w * 0.28), 10, lratio)
                # hola - derecha
                rratio = right_dbg["peaks"] / max(1, wave_right.min_peaks)
                draw_bool_dot(view, (tx + 10, ty + 120), right_dbg["hand_raised"])
                view = draw_text(view, f"hola (der): picos {right_dbg['peaks']}/{wave_right.min_peaks}  x.ptp={right_dbg['x_ptp']:.3f}", (tx + 30, ty + 116), main_font_panel)
                draw_progress_bar(view, tx + 30, ty + 140, int(panel_w * 0.28), 10, rratio)
                # Pulgar arriba
                any_thumb = any(thumbs.debug(h)["thumb_up"] and (thumbs.debug(h)["folded_all"] or not thumbs.require_folded) for h in hands)
                draw_bool_dot(view, (tx + 10, ty + 170), any_thumb)
                view = draw_text(view, f"Pulgar arriba: {'Si' if any_thumb else 'No'}", (tx + 30, ty + 166), main_font_panel)
                # Estado + Mute
                view = draw_text(view, f"Estado: {state_name}    Mute: {'ON' if muted else 'OFF'}", (tx + 10, ty + 195), main_font_panel, color=(220, 220, 220))
                # Contadores
                counters_text = f"Saludos: {wave_count} | Pulgares: {thumbs_up_count}"
                view = draw_text(view, counters_text, (tx + 10, ty + 220), main_font_panel, color=(220, 220, 220))
            else:
                # Fallback si la fuente no cargó (no mostrará tildes)
                cv2.putText(view, "Panel no disponible: Falta fuente TTF", (tx, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Indicador mute PNG
        if muted and mute_icon is not None:
            view = overlay_rgba(view, mute_icon, 20, 20)

        cv2.imshow(win, view)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass
    main()
