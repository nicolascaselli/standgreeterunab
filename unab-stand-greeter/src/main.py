# -*- coding: utf-8 -*-
"""
UNAB Stand Greeter:
- Detección de saludo (ola 👋)
- Saludo en pantalla + TTS
- Espera gesto 👍 para mostrar QR
- Logo, brand y métricas
Controles:
  q: salir
  m: mute TTS
  f: toggle fullscreen
  r: reset estado
"""
import cv2
import time
import yaml
import numpy as np
from detector import Detector
from wave_gesture import WaveDetector
from thumbs_gesture import ThumbsUpDetector
from ui_renderer import UIRenderer
from tts import TTS
from metrics import Metrics

STATE_IDLE = 0
STATE_GREETING = 1
STATE_WAIT_THUMBS = 2
STATE_SHOW_QR = 3
STATE_COOLDOWN = 4

def get_pose_item(pose, key):
    return pose.get(key, (None, None))

def main():
    # --- Cargar config ---
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cam_index = cfg.get("camera_index", 0)
    width, height = cfg.get("resolution", [1280, 720])
    fullscreen = cfg.get("fullscreen", True)

    # --- Inicializar captura ---
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # --- Inicializar módulos ---
    det = Detector()
    wave_cfg = cfg.get("wave", {})
    wave = WaveDetector(
        window_seconds=wave_cfg.get("window_seconds", 1.5),
        amp_threshold=wave_cfg.get("amplitude_threshold", 0.04),
        min_peaks=wave_cfg.get("min_peaks", 3),
    )
    thumbs_cfg = cfg.get("thumbs", {})
    thumbs = ThumbsUpDetector(
        tip_above_wrist_margin=thumbs_cfg.get("tip_above_wrist_margin", 0.02),
        require_folded=thumbs_cfg.get("other_fingers_folded", True),
    )

    tts_cfg = cfg.get("tts", {})
    tts = TTS(enabled=tts_cfg.get("enabled", True),
              rate=tts_cfg.get("rate", 175),
              volume=float(tts_cfg.get("volume", 1.0)))
    metrics = Metrics(cfg.get("metrics", {}).get("csv_path", "metrics.csv"))

    brand = cfg.get("brand", {})
    assets = cfg.get("assets", {})
    ui = UIRenderer(width, height, brand_primary=brand.get("primary", "#A00321"),
                    text_color=brand.get("text", "#FFFFFF"),
                    logo_path=assets.get("logo_path", None))

    idle_text = cfg.get("idle_text", "👋 Acércate y salúdanos")
    prompt_wave_text = cfg.get("prompt_wave_text", "Levanta tu mano y saluda 👋")
    greeting_lines = cfg.get("greeting_texts", ["¡Hola!", "Bienvenido/a a UNAB"])
    qr_cfg = cfg.get("qr", {})
    qr_url = qr_cfg.get("url", "https://www.unab.cl")

    wave_cd = cfg.get("cooldowns", {}).get("wave_seconds", 6)
    thumbs_cd = cfg.get("cooldowns", {}).get("thumbs_seconds", 6)

    # --- Ventana ---
    win = "UNAB Greeter"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # --- Estado ---
    state = STATE_IDLE
    last_trigger_t = 0
    muted = not tts_cfg.get("enabled", True)

    # Mensajes TTS
    greet_tts = "¡Hola! Bienvenido a Ingeniería Civil Informática de la UNAB. Si quieres saber más, muéstranos pulgar arriba."
    qr_tts = "Perfecto. Escanea el código QR para conocer la carrera y sus proyectos."

    while True:
        ok, frame = cap.read()
        if not ok:
            # cámara caída
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Procesar detección
        out = det.process(frame)
        pose = out.get("pose", {})
        hands = out.get("hands", [])

        # Render base por estado
        now = time.time()
        elapsed_since_trigger = now - last_trigger_t

        # Teclas
        key = cv2.waitKey(1) & 0xFF
        if key == ord(cfg["keys"].get("quit", "q")):
            break
        if key == ord(cfg["keys"].get("mute", "m")):
            muted = not muted
            tts.set_enabled(not muted)
            metrics.log("mute_toggled", f"{not muted}")
        if key == ord(cfg["keys"].get("fullscreen_toggle", "f")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        if key == ord(cfg["keys"].get("reset", "r")):
            state = STATE_IDLE
            metrics.log("state_reset")

        # Lógica por estado
        if state == STATE_IDLE:
            view = ui.render_idle(frame, idle_text, prompt_wave_text)

            # Detección de wave
            ls = get_pose_item(pose, "left_shoulder")
            rs = get_pose_item(pose, "right_shoulder")
            lw = get_pose_item(pose, "left_wrist")
            rw = get_pose_item(pose, "right_wrist")

            shoulders_y = [y for (_, y) in [ls, rs] if y is not None]
            if shoulders_y:
                # evaluar mano izquierda
                if lw[0] is not None:
                    if wave.update(lw[0], lw[1], min(shoulders_y), now):
                        state = STATE_GREETING
                        last_trigger_t = now
                        metrics.log("wave_detected", "left")
                        if not muted:
                            tts.speak_async(greet_tts)
                # evaluar mano derecha
                if state == STATE_IDLE and rw[0] is not None:
                    if wave.update(rw[0], rw[1], min(shoulders_y), now):
                        state = STATE_GREETING
                        last_trigger_t = now
                        metrics.log("wave_detected", "right")
                        if not muted:
                            tts.speak_async(greet_tts)

        elif state == STATE_GREETING:
            view = ui.render_greeting(frame, greeting_lines)
            if elapsed_since_trigger > 1.2:
                # Pasamos a esperar 👍
                state = STATE_WAIT_THUMBS
                last_trigger_t = now
                metrics.log("state", "WAIT_THUMBS")

        elif state == STATE_WAIT_THUMBS:
            # Mostrar prompt de 👍
            lines = [greeting_lines[0],
                     "👆 Muéstranos 👍 para ver más"]
            view = ui.render_greeting(frame, lines)

            # Buscar thumbs-up en cualquiera de las manos detectadas
            thumbs_up = False
            for hand in hands:
                if thumbs.is_thumbs_up(hand):
                    thumbs_up = True
                    break

            if thumbs_up:
                state = STATE_SHOW_QR
                last_trigger_t = now
                metrics.log("thumbs_up", "detected")
                if not muted:
                    tts.speak_async(qr_tts)

            # Timeout suave a idle si pasa mucho tiempo sin gesto
            if elapsed_since_trigger > 12:
                state = STATE_IDLE
                metrics.log("timeout_wait_thumbs")

        elif state == STATE_SHOW_QR:
            view = ui.render_qr_panel(frame, qr_url)
            if elapsed_since_trigger > 4.5:
                state = STATE_COOLDOWN
                last_trigger_t = now
                metrics.log("qr_shown")

        elif state == STATE_COOLDOWN:
            view = ui.render_idle(frame, "¡Gracias! 👋", "Acércate y salúdanos")
            if elapsed_since_trigger > max(wave_cd, thumbs_cd):
                state = STATE_IDLE
                metrics.log("cooldown_end")

        else:
            view = frame

        # Indicador de mute
        if muted:
            cv2.putText(view, "🔇", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3, cv2.LINE_AA)

        cv2.imshow(win, view)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
