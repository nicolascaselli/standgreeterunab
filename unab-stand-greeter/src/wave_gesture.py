# -*- coding: utf-8 -*-
"""
Wave gesture detector (saludo 👋) basado en la muñeca sobre el hombro y
oscilación lateral en una ventana temporal.
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

    def update(self, wrist_x, wrist_y, shoulder_y, t=None):
        """Agrega muestra y evalúa si hay gesto 'wave'.
        Devuelve True si detecta saludo.
        """
        if t is None:
            t = time.time()

        # Guardar muestra
        self.samples.append((t, wrist_x, wrist_y, shoulder_y))

        # Limpiar ventana
        t0 = t - self.window_seconds
        while self.samples and self.samples[0][0] < t0:
            self.samples.popleft()

        if len(self.samples) < 5:
            return False

        # Condición mano levantada (muñeca más arriba que hombro)
        ys = np.array([s[2] for s in self.samples], dtype=float)
        shoulder_ys = np.array([s[3] for s in self.samples], dtype=float)
        if not np.any(ys < shoulder_ys - 1e-3):
            return False

        # Analizar oscilación lateral (x)
        xs = np.array([s[1] for s in self.samples], dtype=float)
        xs_d = xs - xs.mean()

        # Detectar picos por cambio de signo con amplitud mínima
        peaks = 0
        for i in range(1, len(xs_d) - 1):
            if xs_d[i-1] < 0 <= xs_d[i] or xs_d[i-1] > 0 >= xs_d[i]:
                # cruce por cero entre i-1 e i
                # verificar amplitud en ventana local
                left = max(0, i - 3)
                right = min(len(xs_d)-1, i + 3)
                local_amp = xs_d[left:right+1].ptp()  # peak-to-peak
                if local_amp >= self.amp_threshold:
                    peaks += 1

        return peaks >= self.min_peaks
