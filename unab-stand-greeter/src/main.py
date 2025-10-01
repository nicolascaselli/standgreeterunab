# -*- coding: utf-8 -*-
"""
UNAB Stand Greeter (con HUD de reconocimientos + manos dibujadas):
- Detección de saludo (hola) y pulgar arriba con panel en vivo
- UI + TTS + QR + Logo
- ASK_GAME: elegir entre QR (👍) o Jugar (✋ mano abierta)
Controles:
  q: salir | m: mute | f: fullscreen | r: reset
"""
import cv2, time, yaml, numpy as np
from pathlib import Path
import urllib.request, ssl, platform
from PIL import ImageFont, Image, ImageDraw
from maze import MazeGame

# --- Paths base ---
SCRIPT_DIR = Path(__file__).resolve().parent      # src
PROJECT_ROOT = SCRIPT_DIR.parent                  # raíz

from detector import Detector
from wave_gesture import WaveDetector
from thumbs_gesture import ThumbsUpDetector
from ui_renderer import UIRenderer
from tts import TTS
from metrics import Metrics

# ---------------- Estados ----------------
STATE_IDLE, STATE_GREETING, STATE_WAIT_THUMBS, STATE_SHOW_QR, STATE_COOLDOWN = range(5)
STATE_ASK_GAME = 5   # QR o Juego
STATE_GAME     = 6   # Juego del laberinto

# ---------------- Landmarks mano (MediaPipe) ----------------
WRIST = 0
INDEX_PIP, INDEX_TIP   = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP     = 14, 16
PINKY_PIP, PINKY_TIP   = 18, 20

# Intentar usar conexiones oficiales de MediaPipe; si no están, usar fallback fijo
try:
    import mediapipe as mp
    HAND_CONNS = list(mp.solutions.hands.HAND_CONNECTIONS)
except Exception:
    HAND_CONNS = [
        (0,1),(1,2),(2,3),(3,4),      # pulgar
        (0,5),(5,6),(6,7),(7,8),      # índice
        (5,9),(9,10),(10,11),(11,12), # medio
        (9,13),(13,14),(14,15),(15,16),# anular
        (13,17),(17,18),(18,19),(19,20),# meñique
        (0,17)                        # muñeca a meñique base
    ]

def get_index_tip_px(hands, width, height):
    """Devuelve (x_px, y_px) del dedo índice de la PRIMERA mano válida; o None."""
    INDEX_TIP = 8
    for h in hands:
        if INDEX_TIP in h:
            x, y = h[INDEX_TIP]
            return int(x * width), int(y * height)
    return None

# ---------------- Utilidades de mano ----------------
def is_hand_open(hand_landmarks: dict, margin: float = 0.01, min_extended: int = 3) -> bool:
    """Mano abierta si 3+ dedos (sin pulgar) tienen TIP.y < PIP.y - margin (coords normalizadas)."""
    if not hand_landmarks:
        return False
    count = 0
    pairs = [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
             (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]
    for tip, pip in pairs:
        if tip in hand_landmarks and pip in hand_landmarks:
            _, tipy = hand_landmarks[tip]
            _, pipy = hand_landmarks[pip]
            if tipy < (pipy - margin):
                count += 1
    return count >= min_extended

def _hand_to_indexed_dict(hand):
    """
    Normaliza una mano a dict {0..20: (x,y)}:
    - Si ya es dict con claves int, lo retorna.
    - Si es lista/tupla de 21 elementos (x,y), la mapea.
    - Si es objeto MediaPipe con .landmark, lo mapea.
    - Si no se reconoce, retorna None.
    """
    if hand is None:
        return None
    if isinstance(hand, dict) and all(isinstance(k, int) for k in hand.keys()):
        return hand
    if isinstance(hand, (list, tuple)) and len(hand) == 21:
        try:
            d = {i: (float(pt[0]), float(pt[1])) for i, pt in enumerate(hand)}
            return d
        except Exception:
            pass
    lm = getattr(hand, "landmark", None)
    if lm is not None and len(lm) == 21:
        try:
            d = {i: (float(lm[i].x), float(lm[i].y)) for i in range(21)}
            return d
        except Exception:
            pass
    return None

def normalize_hands(hands_list):
    """Aplica _hand_to_indexed_dict a cada mano; filtra None."""
    out = []
    for h in hands_list or []:
        d = _hand_to_indexed_dict(h)
        if d:
            out.append(d)
    return out

# ---------------- Dibujo landmarks pose y mano ----------------
def draw_landmarks_basic(img, pose, width, height):
    """Hombros y muñecas (pose)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    key_color = (50, 200, 255)
    for k in ("left_shoulder", "right_shoulder", "left_wrist", "right_wrist"):
        x, y = pose.get(k, (None, None))
        if x is None: continue
        cx, cy = int(x * width), int(y * height)
        cv2.circle(img, (cx, cy), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 5, key_color, -1, cv2.LINE_AA)
    return img

def draw_hands_landmarks(img, hands, width, height, color_pts=(0,255,255), color_conn=(255,120,0)):
    """Dibuja 21 puntos y conexiones por mano."""
    if not hands:
        return img
    for hand in hands:
        for a, b in HAND_CONNS:
            if a in hand and b in hand:
                ax, ay = hand[a]; bx, by = hand[b]
                ax, ay = int(ax*width), int(ay*height)
                bx, by = int(bx*width), int(by*height)
                cv2.line(img, (ax, ay), (bx, by), color_conn, 2, cv2.LINE_AA)
        for i, (x, y) in hand.items():
            cx, cy = int(x*width), int(y*height)
            r = 4 if i != INDEX_TIP else 6
            cv2.circle(img, (cx, cy), r, (0,0,0), -1, cv2.LINE_AA)
            cv2.circle(img, (cx, cy), r-1, color_pts, -1, cv2.LINE_AA)
    return img

# ---------- HUD / overlay ----------
def overlay_alpha(img, overlay, x, y, alpha=0.6):
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
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text((org[0] + 1, org[1] + 1), text, font=font, fill=(0, 0, 0))
    draw.text(org, text, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

def draw_bool_dot(img, center, ok):
    color = (40, 180, 60) if ok else (40, 40, 200)
    cv2.circle(img, center, 7, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(img, center, 6, color, -1, cv2.LINE_AA)

def draw_progress_bar(img, x, y, w, h, ratio, color=(40, 180, 60)):
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2, cv2.LINE_AA)
    fill = int(max(0.0, min(1.0, ratio)) * (w - 4))
    cv2.rectangle(img, (x + 2, y + 2), (x + 2 + fill, y + h - 2), color, -1, cv2.LINE_AA)

# ---------- Emoji PNG helpers ----------
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
        scale = height / float(h); width = int(w * scale)
    elif height is None:
        scale = width / float(w); height = int(h * scale)
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

def overlay_rgba(bg_bgr, fg_bgra, x, y):
    if fg_bgra is None: return bg_bgr
    H, W = bg_bgr.shape[:2]; h, w = fg_bgra.shape[:2]
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
        bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff] = ((1 - alpha) * bg_roi + alpha * fg_crop[:, :, :3]).astype(bg_roi.dtype)
    else:
        bg_bgr[y0_bg:y0_bg + h_eff, x0_bg:x0_bg + w_eff] = fg_crop[:, :, :3]
    return bg_bgr

def compute_panel_xy(anchor, panel_w, panel_h, width, height, margin_xy):
    ax = (anchor or "bl").lower()
    mx, my = margin_xy
    if "l" in ax: x = mx
    elif "r" in ax: x = width - panel_w - mx
    else: x = (width - panel_w) // 2
    if "t" in ax: y = my
    elif "b" in ax: y = height - panel_h - my
    else: y = (height - panel_h) // 2
    return x, y

def setup_emoji_font():
    system = platform.system()
    if system == "Darwin":
        system_fonts = ["/System/Library/Fonts/Apple Color Emoji.ttc", "/Library/Fonts/Apple Color Emoji.ttc"]
    elif system == "Windows":
        system_fonts = ["C:/Windows/Fonts/seguiemj.ttf", "C:/Windows/Fonts/NotoColorEmoji.ttf"]
    else:
        system_fonts = ["/usr/share/fonts/truetype/noto-color-emoji/NotoColorEmoji.ttf",
                        "/usr/share/fonts/noto-color-emoji/NotoColorEmoji.ttf",
                        "/system/fonts/NotoColorEmoji.ttf"]
    for font_path in system_fonts:
        if Path(font_path).exists():
            print(f"INFO: Usando fuente de emojis del sistema: {font_path}")
            return Path(font_path)

    print(f"INFO: No se encontraron fuentes de emojis del sistema en {system}")
    font_path = PROJECT_ROOT / "src/assets/fonts/NotoColorEmoji.ttf"
    font_url = "https://github.com/googlefonts/noto-emoji/raw/main/fonts/NotoColorEmoji.ttf"
    if not font_path.exists():
        print("INFO: Intentando descargar NotoColorEmoji...")
        try:
            font_path.parent.mkdir(parents=True, exist_ok=True)
            context = ssl._create_unverified_context()
            req = urllib.request.Request(font_url)
            req.add_header('User-Agent', 'Mozilla/5.0 (compatible; emoji-downloader/1.0)')
            with urllib.request.urlopen(req, context=context, timeout=30) as response:
                if response.status == 200:
                    with open(font_path, 'wb') as out_file:
                        out_file.write(response.read())
                    print(f"INFO: Fuente descargada y guardada en '{font_path}'")
                else:
                    print(f"ERROR: HTTP {response.status} al descargar la fuente"); return None
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo descargar la fuente de emojis: {e}"); return None
    else:
        print(f"INFO: Fuente de emojis encontrada en '{font_path}'")
    if font_path.exists() and font_path.stat().st_size > 1000:
        return font_path
    print("ERROR: La fuente descargada parece estar corrupta")
    return None

def main():
    # Cargar config
    config_path_root = PROJECT_ROOT / "config.yaml"
    config_path_src  = SCRIPT_DIR / "config.yaml"
    if config_path_root.exists(): config_path = config_path_root
    elif config_path_src.exists(): config_path = config_path_src
    else:
        raise FileNotFoundError(f"No se pudo encontrar 'config.yaml'.\n- {config_path_root}\n- {config_path_src}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_index = cfg.get("camera_index", 0)

    # --- resoluciones y mínimos (canvas UI grande tipo TV) ---
    target_w, target_h = cfg.get("resolution", [1600, 900])              # por defecto grande
    min_w, min_h = (cfg.get("min_resolution") or [1280, 720])            # mínimo de seguridad
    width  = max(int(target_w), int(min_w))
    height = max(int(target_h), int(min_h))

    # pedir mínimo a la cámara (si driver lo respeta)
    camera_cfg = (cfg.get("camera") or {})
    cam_min = camera_cfg.get("min_resolution", [min_w, min_h])
    cam_req_w, cam_req_h = int(cam_min[0]), int(cam_min[1])
    # espejo: hace más natural el control con la mano frente a la TV
    mirror_preview = bool(camera_cfg.get("mirror", True))

    fullscreen = cfg.get("fullscreen", False)  # por defecto NO fullscreen
    debug_cfg = cfg.get("debug", {}) or {}
    SHOW_PANEL = bool(debug_cfg.get("show_panel", True))
    DRAW_LANDMARKS = bool(debug_cfg.get("draw_landmarks", True))

    # Captura
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_req_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_req_h)

    # Detectores
    det = Detector()
    wave_cfg = cfg.get("wave", {})
    wave_left  = WaveDetector(wave_cfg.get("window_seconds", 1.5),
                              wave_cfg.get("amplitude_threshold", 0.04),
                              wave_cfg.get("min_peaks", 3))
    wave_right = WaveDetector(wave_cfg.get("window_seconds", 1.5),
                              wave_cfg.get("amplitude_threshold", 0.04),
                              wave_cfg.get("min_peaks", 3))

    thumbs_cfg = cfg.get("thumbs", {})
    thumbs = ThumbsUpDetector(thumbs_cfg.get("tip_above_wrist_margin", 0.02),
                              thumbs_cfg.get("other_fingers_folded", True))

    # TTS y métricas
    tts_cfg = cfg.get("tts", {})
    tts = TTS(enabled=tts_cfg.get("enabled", True),
              rate=tts_cfg.get("rate", 175),
              volume=float(tts_cfg.get("volume", 1.0)))
    metrics = Metrics(cfg.get("metrics", {}).get("csv_path", "metrics.csv"))

    # Fuentes/Assets
    emoji_font_path = setup_emoji_font()
    font_path_abs = PROJECT_ROOT / "src/assets/fonts/Roboto-Regular.ttf"
    main_font_panel = None
    if font_path_abs.exists():
        try: main_font_panel = ImageFont.truetype(str(font_path_abs), 16)
        except Exception as e: print(f"ADVERTENCIA: No se pudo cargar la fuente principal para el panel: {e}")

    brand = cfg.get("brand", {})
    assets_cfg = cfg.get("assets", {})
    logo_path_str = assets_cfg.get("logo_path")
    logo_path_abs = PROJECT_ROOT / logo_path_str if logo_path_str else None

    ui_assets = {"logo": logo_path_abs, "emoji_font_path": emoji_font_path, "font_path": font_path_abs}
    ui = UIRenderer(width, height,
                    brand_primary=brand.get("primary", "#A00321"),
                    text_color=brand.get("text", "#FFFFFF"),
                    assets=ui_assets,
                    show_qr_panel=False)

    # Textos
    idle_text = cfg.get("idle_text", "Acércate y salúdanos")
    greeting_lines = [line for line in cfg.get("greeting_texts", ["¡Hola! 👋", "Bienvenido/a a UNAB"])]
    prompt_wave_text = cfg.get("prompt_wave_text", "Levanta tu mano y saluda 👋")
    qr_url = (cfg.get("qr", {}) or {}).get("url", "https://www.unab.cl/carreras/mallas/ing_civil_informatica.pdf")
    wave_cd = cfg.get("cooldowns", {}).get("wave_seconds", 6)
    thumbs_cd = cfg.get("cooldowns", {}).get("thumbs_seconds", 6)

    # Íconos (opcional)
    icon_big_h, icon_small_h = int(height * 0.14), int(height * 0.08)
    wave_icon  = _resize(_read_rgba(PROJECT_ROOT / "src/assets/emoji/wave.png"),      height=icon_big_h)
    thumb_icon = _resize(_read_rgba(PROJECT_ROOT / "src/assets/emoji/thumbs_up.png"), height=icon_big_h)
    mute_icon  = _resize(_read_rgba(PROJECT_ROOT / "src/assets/emoji/mute.png"),      height=icon_small_h)

    # Ventana (grande, no fullscreen)
    win = "UNAB Greeter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    win_cfg = (cfg.get("window") or {})
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        w, h = win_cfg.get("size", [width, height])
        cv2.resizeWindow(win, int(w), int(h))
        x, y = win_cfg.get("position", [50, 50])
        try: cv2.moveWindow(win, int(x), int(y))
        except Exception: pass
        if win_cfg.get("always_on_top", False):
            try:
                if platform.system() == "Windows":
                    import win32gui, win32con
                    hwnd = win32gui.FindWindow(None, win)
                    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, int(x), int(y), int(w), int(h), 0)
            except Exception as e:
                print("No pude poner la ventana 'always on top':", e)

    # Estado
    state = STATE_IDLE
    last_trigger_t = 0
    muted = not tts_cfg.get("enabled", True)
    wave_count = 0
    thumbs_up_count = 0

    greet_tts   = "Hola! Bienvenido a Ingeniería Civil Informática de la UNAB. Si quieres saber más, muéstranos pulgar arriba."
    qr_tts      = "Perfecto. Escanea el código QR para conocer la carrera y sus proyectos."
    ask_game_tts= "¿Quieres ver información o jugar al laberinto? Pulgar arriba para el QR, mano abierta para jugar."
    maze_game = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Normaliza el frame al canvas fijo (width x height)
            fh, fw = frame.shape[:2]
            if (fw != width) or (fh != height):
                interp = cv2.INTER_CUBIC if (width*height) > (fw*fh) else cv2.INTER_AREA
                frame = cv2.resize(frame, (width, height), interpolation=interp)
            # ---- espejo para control natural frente a la TV ----
            if mirror_preview:
                frame = cv2.flip(frame, 1)

        out = det.process(frame)
        pose  = out.get("pose", {})
        hands_raw = out.get("hands", [])
        hands = normalize_hands(hands_raw)

        now = time.time()
        elapsed = now - last_trigger_t

        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord(cfg["keys"].get("quit", "q")): break
        if key == ord(cfg["keys"].get("mute", "m")):
            muted = not muted; tts.set_enabled(not muted); metrics.log("mute_toggled", f"{not muted}")
        if key == ord(cfg["keys"].get("fullscreen_toggle", "f")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
            if not fullscreen:
                w, h = win_cfg.get("size", [width, height])
                cv2.resizeWindow(win, int(w), int(h))
                x, y = win_cfg.get("position", [50, 50])
                try: cv2.moveWindow(win, int(x), int(y))
                except Exception: pass
                if win_cfg.get("always_on_top", False):
                    try:
                        if platform.system() == "Windows":
                            import win32gui, win32con
                            hwnd = win32gui.FindWindow(None, win)
                            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, int(x), int(y), int(w), int(h), 0)
                    except Exception as e:
                        print("No pude poner la ventana 'always on top':", e)
        if key == ord(cfg["keys"].get("reset", "r")):
            state = STATE_IDLE; metrics.log("state_reset")
        if key == ord('g'): # salir del juego manualmente
            maze_game = None
            state = STATE_COOLDOWN
            last_trigger_t = now

        # Render base (ya espejo si aplica)
        view = frame.copy()

        # Pose básica
        ls = pose.get("left_shoulder", (None, None))
        rs = pose.get("right_shoulder", (None, None))
        lw = pose.get("left_wrist", (None, None))
        rw = pose.get("right_wrist", (None, None))
        shoulders_y = [y for (_, y) in (ls, rs) if y is not None]
        person_present = bool(shoulders_y)

        # Actualizar detectores de hola
        wave_left_trigger = False
        wave_right_trigger = False
        if shoulders_y:
            min_sh_y = min(shoulders_y)
            if lw[0] is not None:
                wave_left_trigger = wave_left.update(lw[0], lw[1], min_sh_y, now)
            if rw[0] is not None:
                wave_right_trigger = wave_right.update(rw[0], rw[1], min_sh_y, now)

        # ------------------ Máquina de estados ------------------
        if state == STATE_IDLE:
            view = ui.render_idle(view, idle_text, prompt_wave_text)
            if wave_left_trigger or wave_right_trigger:
                state = STATE_GREETING
                last_trigger_t = now
                wave_count += 1
                metrics.log("wave_detected", "left" if wave_left_trigger else "right")
                if not muted: tts.speak_async(greet_tts)

        elif state == STATE_GREETING:
            view = ui.render_greeting(view, greeting_lines)
            if elapsed > 1.2:
                state = STATE_ASK_GAME
                last_trigger_t = now
                metrics.log("state", "ASK_GAME")
                if not muted: tts.speak_async(ask_game_tts)

        elif state == STATE_ASK_GAME:
            # Elección: 👍 QR  |  ✋ Jugar
            lines = ["¿Quieres saber más o jugar?", "👍 QR   |   ✋ Jugar laberinto"]
            view = ui.render_greeting(view, lines)

            thumbs_up = any(thumbs.is_thumbs_up(h) for h in hands)
            open_hand = any(is_hand_open(h) for h in hands)

            if thumbs_up:
                state = STATE_SHOW_QR
                last_trigger_t = now
                metrics.log("choice", "QR")
                if not muted: tts.speak_async(qr_tts)
            elif open_hand:
                state = STATE_GAME
                last_trigger_t = now
                metrics.log("choice", "GAME")
                if not muted: tts.speak_async("¡Vamos a jugar!")

            if elapsed > 12:
                state = STATE_IDLE; metrics.log("timeout_ask_game")

        elif state == STATE_WAIT_THUMBS:
            # Compatibilidad (no se usa en el nuevo flujo)
            thumbs_text = "Muéstranos Pulgar arriba 👍 para ver más"
            view = ui.render_greeting(view, [greeting_lines[0], thumbs_text])
            thumbs_up = any((lambda d=thumbs.debug(h): d["thumb_up"] and (d["folded_all"] or not thumbs.require_folded))(h) for h in hands)
            if thumbs_up:
                state = STATE_SHOW_QR; last_trigger_t = now
                thumbs_up_count += 1; metrics.log("thumbs_up", "detected")
                if not muted: tts.speak_async(qr_tts)
            if elapsed > 12:
                state = STATE_IDLE; metrics.log("timeout_wait_thumbs")

        elif state == STATE_SHOW_QR:
            view = ui.render_qr_panel(view, qr_url)
            if elapsed > 4.5:
                state = STATE_COOLDOWN; last_trigger_t = now; metrics.log("qr_shown")

        elif state == STATE_GAME:
            # Inicializar el laberinto al entrar al estado con el tamaño real del VIEW (ya espejo)
            if maze_game is None:
                fh, fw = view.shape[:2]  # alto, ancho del frame actual
                maze_cfg = (cfg.get("maze") or {})
                maze_game = MazeGame(
                    fw, fh,
                    grid_cols=maze_cfg.get("grid_cols", 11),  # menos celdas => pasillos anchos
                    grid_rows=maze_cfg.get("grid_rows", 7),
                    panel_ratio_w=maze_cfg.get("panel_ratio_w", 0.72),
                    panel_ratio_h=maze_cfg.get("panel_ratio_h", 0.58),
                    thin_ratio=maze_cfg.get("thin_ratio", 0.25),
                    collision_rel=maze_cfg.get("collision_rel", 0.006),  # más permisivo
                    start_goal_pad=maze_cfg.get("start_goal_pad", 0.30),  # inicio/meta más grandes
                    trail_thickness_rel=maze_cfg.get("trail_thickness_rel", 0.022),
                )

            # Obtener dedo índice en píxeles
            finger_px = get_index_tip_px(hands, width, height)

            # Actualizar juego
            status = maze_game.update(finger_px, now)

            # Renderizar en el frame
            view = maze_game.render(view, main_font_panel)

            # Reacciones
            if status.life_lost:
                metrics.log("maze_life_lost")
            if status.win:
                metrics.log("maze_win")
                if not muted: tts.speak_async("¡Ganaste el laberinto!")
                state = STATE_COOLDOWN; last_trigger_t = now; maze_game = None
            if status.game_over:
                metrics.log("maze_game_over")
                if not muted: tts.speak_async("¡Game Over! Gracias por jugar.")
                state = STATE_COOLDOWN; last_trigger_t = now; maze_game = None

        elif state == STATE_COOLDOWN:
            view = ui.render_idle(view, "¡Gracias! ✨", "Acércate y salúdanos")
            if elapsed > max(wave_cd, thumbs_cd):
                state = STATE_IDLE; metrics.log("cooldown_end")

        # --- Dibujo de LANDMARKS (pose + manos) ---
        if DRAW_LANDMARKS:
            view = draw_landmarks_basic(view, pose, width, height)
            view = draw_hands_landmarks(view, hands, width, height)

        # --- HUD ---
        if SHOW_PANEL:
            pconf = (cfg.get("debug", {}) or {}).get("panel", {}) or {}
            anchor = pconf.get("anchor", "bl")
            margin = pconf.get("margin", [20, 20])
            panel_w = int(width * float(pconf.get("width_ratio", 0.36)))
            panel_h = int(height * float(pconf.get("height_ratio", 0.34)))
            alpha = float(pconf.get("alpha", 0.65))

            left_dbg  = wave_left.get_debug()
            right_dbg = wave_right.get_debug()
            state_name = {
                0: "IDLE", 1: "GREETING", 2: "WAIT_THUMBS", 3: "SHOW_QR",
                4: "COOLDOWN", 5: "ASK_GAME", 6: "GAME"
            }[state]

            panel = np.full((panel_h, panel_w, 3), (25, 25, 25), dtype=np.uint8)
            x0, y0 = compute_panel_xy(anchor, panel_w, panel_h, width, height, margin)
            view = overlay_alpha(view, panel, x0, y0, alpha=alpha)
            tx, ty = x0 + 10, y0 + 10

            if main_font_panel:
                view = draw_text(view, "Reconocimiento (en vivo)", (tx, ty+2), main_font_panel, (255,255,255))
                view = draw_text(view, f"Manos detectadas: {len(hands)}", (tx, ty+26), main_font_panel, (220,220,220))
                draw_bool_dot(view, (tx + 10, ty + 54), person_present)
                view = draw_text(view, f"Persona detectada: {'Si' if person_present else 'No'}", (tx + 30, ty + 50), main_font_panel)
                lratio = left_dbg["peaks"] / max(1, wave_left.min_peaks)
                draw_bool_dot(view, (tx + 10, ty + 84), left_dbg["hand_raised"])
                view = draw_text(view, f"hola (izq): picos {left_dbg['peaks']}/{wave_left.min_peaks}  x.ptp={left_dbg['x_ptp']:.3f}", (tx + 30, ty + 80), main_font_panel)
                draw_progress_bar(view, tx + 30, ty + 104, int(panel_w * 0.28), 10, lratio)
                rratio = right_dbg["peaks"] / max(1, wave_right.min_peaks)
                draw_bool_dot(view, (tx + 10, ty + 134), right_dbg["hand_raised"])
                view = draw_text(view, f"hola (der): picos {right_dbg['peaks']}/{wave_right.min_peaks}  x.ptp={right_dbg['x_ptp']:.3f}", (tx + 30, ty + 130), main_font_panel)
                draw_progress_bar(view, tx + 30, ty + 154, int(panel_w * 0.28), 10, rratio)
                any_thumb = any(thumbs.debug(h)["thumb_up"] and (thumbs.debug(h)["folded_all"] or not thumbs.require_folded) for h in hands)
                any_open  = any(is_hand_open(h) for h in hands)
                draw_bool_dot(view, (tx + 10, ty + 184), any_thumb)
                view = draw_text(view, f"Pulgar arriba: {'Si' if any_thumb else 'No'}", (tx + 30, ty + 180), main_font_panel)
                draw_bool_dot(view, (tx + 10, ty + 214), any_open)
                view = draw_text(view, f"Mano abierta (jugar): {'Si' if any_open else 'No'}", (tx + 30, ty + 210), main_font_panel)
                view = draw_text(view, f"Estado: {state_name}    Mute: {'ON' if muted else 'OFF'}", (tx + 10, ty + 238), main_font_panel, (220,220,220))
            else:
                cv2.putText(view, "Panel no disponible: Falta fuente TTF", (tx, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Indicador mute
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
