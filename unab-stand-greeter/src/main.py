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
from PIL import ImageFont
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

def remove_emojis(text):
    # Elimina emojis y caracteres fuera del rango Unicode latino básico
    return re.sub(r'[^\w\s.,;:¡!¿?áéíóúÁÉÍÓÚñÑüÜ/-]', '', text)
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


def draw_text(img, text, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def draw_bool_dot(img, center, ok):
    color = (40, 180, 60) if ok else (40, 40, 200)
    cv2.circle(img, center, 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, center, 6, color, -1, cv2.LINE_AA)


def draw_progress_bar(img, x, y, w, h, ratio, color=(40, 180, 60)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
    fill = int(max(0.0, min(1.0, ratio)) * (w - 4))
    cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill, y + h - 2), color, -1, cv2.LINE_AA)


def draw_landmarks_basic(img, pose, width, height):
    """Dibuja hombros y muñecas detectadas."""
    key_color = (50, 200, 255)
    for k in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist"):
        x, y = pose.get(k, (None, None))
        if x is None: continue
        cx, cy = int(x * width), int(y * height)
        cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, key_color, -1, cv2.LINE_AA)


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
    """Verifica y descarga la fuente de emojis si no existe, luego la carga."""
    # La ruta ahora se construye a partir de la raíz del proyecto
    font_path = PROJECT_ROOT / "src/assets/fonts/NotoColorEmoji-Regular.ttf"
    font_url = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"

    if not font_path.exists():
        print("INFO: Fuente de emojis no encontrada. Intentando descargar...")
        try:
            font_path.parent.mkdir(parents=True, exist_ok=True)
            # Crear un contexto SSL que no verifique certificados
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(font_url, context=context) as response, open(font_path, 'wb') as out_file:
                out_file.write(response.read())
            print(f"INFO: Fuente descargada y guardada en '{font_path}'")
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo descargar la fuente de emojis: {e}")
            return None

    try:
        return ImageFont.truetype(str(font_path), 38)
    except IOError:
        print(f"ADVERTENCIA: No se pudo cargar la fuente desde '{font_path}'. Los emojis no se mostrarán.")
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
    emoji_font = setup_emoji_font()
    brand = cfg.get("brand", {})
    assets_cfg = cfg.get("assets", {})

    # Construir rutas absolutas para todos los assets
    logo_path_str = assets_cfg.get("logo_path")
    logo_path_abs = PROJECT_ROOT / logo_path_str if logo_path_str else None

    # Crear un diccionario de assets con rutas absolutas
    ui_assets = {
        "logo": logo_path_abs,
        "emoji_font": emoji_font
    }

    # Pasar el diccionario de assets a UIRenderer
    ui = UIRenderer(width, height,
                    brand_primary=brand.get("primary", "#A00321"),
                    text_color=brand.get("text", "#FFFFFF"),
                    assets=ui_assets,
                    show_qr_panel=False)  # asegurarse False

    # --- Carga de textos y configuración (sin cambios) ---
    idle_text = remove_emojis(cfg.get("idle_text", "Acércate y salúdanos"))
    prompt_wave_text = remove_emojis(cfg.get("prompt_wave_text", "Levanta tu mano y saluda"))
    greeting_lines = [remove_emojis(line) for line in cfg.get("greeting_texts", ["¡Hola!", "Bienvenido/a a UNAB"])]

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
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    state = STATE_IDLE
    last_trigger_t = 0
    muted = not tts_cfg.get("enabled", True)
    wave_count = 0
    thumbs_up_count = 0

    greet_tts = "Hola! Bienvenido a Ingeniería Civil Informática de la UNAB. Si quieres saber más, muéstranos pulgar arriba."  # Emoji quitado
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
            tts.set_enabled(not muted if False else not muted)  # pequeña defensa
            tts.set_enabled(not muted);
            metrics.log("mute_toggled", f"{not muted}")
        if key == ord(cfg["keys"].get("fullscreen_toggle", "f")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
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
            view = ui.render_greeting(view, [greeting_lines[0], "Muéstranos Pulgar arriba para ver más"])  # Emoji quitado

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
            view = ui.render_idle(view, "¡Gracias!", "Acércate y salúdanos")  # Emoji quitado
            if elapsed > max(wave_cd, thumbs_cd):
                state = STATE_IDLE;
                metrics.log("cooldown_end")

        # --- HUD: landmarks + panel ---
        if DRAW_LANDMARKS:
            draw_landmarks_basic(view, pose, width, height)

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

            # Título
            draw_text(view, "Reconocimiento (en vivo)", (tx, ty + 12), scale=0.7, color=(255, 255, 255))

            # Persona
            draw_bool_dot(view, (tx + 10, ty + 40), person_present)
            draw_text(view, f"Persona detectada: {'Si' if person_present else 'No'}", (tx + 30, ty + 46), 0.6)

            # hola - izquierda
            lratio = left_dbg["peaks"] / max(1, wave_left.min_peaks)
            draw_bool_dot(view, (tx + 10, ty + 70), left_dbg["hand_raised"])
            draw_text(view,
                      f"hola (izq): picos {left_dbg['peaks']}/{wave_left.min_peaks}  x.ptp={left_dbg['x_ptp']:.3f}",
                      (tx + 30, ty + 76), 0.55)
            draw_progress_bar(view, tx + 30, ty + 90, int(panel_w * 0.28), 10, lratio)

            # hola - derecha
            rratio = right_dbg["peaks"] / max(1, wave_right.min_peaks)
            draw_bool_dot(view, (tx + 10, ty + 120), right_dbg["hand_raised"])
            draw_text(view,
                      f"hola (der): picos {right_dbg['peaks']}/{wave_right.min_peaks}  x.ptp={right_dbg['x_ptp']:.3f}",
                      (tx + 30, ty + 126), 0.55)
            draw_progress_bar(view, tx + 30, ty + 140, int(panel_w * 0.28), 10, rratio)

            # Pulgar arriba
            any_thumb = False
            if hands:
                for hand in hands:
                    dbg = thumbs.debug(hand)
                    any_thumb = any_thumb or (dbg["thumb_up"] and (dbg["folded_all"] or not thumbs.require_folded))
            draw_bool_dot(view, (tx + 10, ty + 170), any_thumb)
            draw_text(view, f"Pulgar arriba: {'Si' if any_thumb else 'No'}", (tx + 30, ty + 176), 0.6)

            # Estado + Mute
            draw_text(view, f"Estado: {state_name}    Mute: {'ON' if muted else 'OFF'}", (tx + 10, ty + 205), 0.6,
                      color=(220, 220, 220))

            # Contadores
            counters_text = f"Saludos: {wave_count} | Pulgares: {thumbs_up_count}"
            draw_text(view, counters_text, (tx + 10, ty + 230), 0.6, color=(220, 220, 220))

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