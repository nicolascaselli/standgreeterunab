# -*- coding: utf-8 -*-
"""
Wave gesture detector (saludo 👋) con métricas de depuración.
Detecta muñeca por encima del hombro y oscilación lateral.
"""
from collections import deque
import time
import numpy as np

class WaveDetector:
    def __init__(self, window_seconds=1.5, amp_threshold=0.04, min_peaks=3):
        self.window_seconds = window_seconds
        self.amp_threshold = amp_threshold
        self.min_peaks = min_peaks
        self.samples = deque()  # (t, x_norm, y_norm, shoulder_y)

        # últimos valores para HUD
        self.last_debug = {
            "hand_raised": False,
            "peaks": 0,
            "x_ptp": 0.0,       # peak-to-peak global de x en ventana
            "n_samples": 0
        }

    def update(self, wrist_x, wrist_y, shoulder_y, t=None):
        """Agrega muestra y evalúa si hay gesto 'wave'. Devuelve True si detecta saludo."""
        if t is None:
            t = time.time()

        self.samples.append((t, wrist_x, wrist_y, shoulder_y))

        # Limpiar ventana
        t0 = t - self.window_seconds
        while self.samples and self.samples[0][0] < t0:
            self.samples.popleft()

        self.last_debug["n_samples"] = len(self.samples)
        if len(self.samples) < 5:
            self.last_debug.update({"hand_raised": False, "peaks": 0, "x_ptp": 0.0})
            return False

        ys = np.array([s[2] for s in self.samples], dtype=float)
        shoulder_ys = np.array([s[3] for s in self.samples], dtype=float)
        hand_raised = np.any(ys < shoulder_ys - 1e-3)

        xs = np.array([s[1] for s in self.samples], dtype=float)
        xs_d = xs - xs.mean()
        x_ptp = float(xs_d.ptp())

        # contar picos (cruces por cero con amplitud local)
        peaks = 0
        for i in range(1, len(xs_d) - 1):
            if xs_d[i-1] < 0 <= xs_d[i] or xs_d[i-1] > 0 >= xs_d[i]:
                left = max(0, i - 3)
                right = min(len(xs_d)-1, i + 3)
                local_amp = xs_d[left:right+1].ptp()
                if local_amp >= self.amp_threshold:
                    peaks += 1

        self.last_debug.update({
            "hand_raised": bool(hand_raised),
            "peaks": int(peaks),
            "x_ptp": x_ptp
        })

        return hand_raised and (peaks >= self.min_peaks)

    def get_debug(self):
        return dict(self.last_debug)
