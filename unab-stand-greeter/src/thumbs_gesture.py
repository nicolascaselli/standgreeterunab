# -*- coding: utf-8 -*-
"""
Detector simple de pulgar arriba 👍 usando MediaPipe Hands (landmarks normalizados).
Reglas:
- Pulgar extendido: tip del pulgar por encima del wrist (y menor).
- Resto de dedos "plegados": tips por debajo de sus MCPs (y mayor).
Note: sistema de coordenadas MP: y crece hacia abajo; "arriba" = y más pequeño.
"""
from typing import Dict

# Índices de landmarks en MediaPipe Hands
THUMB_TIP = 4
WRIST = 0

INDEX_TIP = 8
INDEX_MCP = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9
RING_TIP = 16
RING_MCP = 13
PINKY_TIP = 20
PINKY_MCP = 17

class ThumbsUpDetector:
    def __init__(self, tip_above_wrist_margin=0.02, require_folded=True):
        self.tip_above_wrist_margin = tip_above_wrist_margin
        self.require_folded = require_folded

    def is_thumbs_up(self, hand_landmarks: Dict[int, tuple]) -> bool:
        """
        hand_landmarks: dict {index: (x_norm, y_norm)}
        """
        if WRIST not in hand_landmarks or THUMB_TIP not in hand_landmarks:
            return False

        wx, wy = hand_landmarks[WRIST]
        tx, ty = hand_landmarks[THUMB_TIP]

        # Pulgar por encima de la muñeca
        thumb_up = ty < (wy - self.tip_above_wrist_margin)

        if not thumb_up:
            return False

        if not self.require_folded:
            return True

        # Dedos índice/medio/anular/meñique "plegados": tip por debajo de MCP
        def folded(tip, mcp):
            if tip not in hand_landmarks or mcp not in hand_landmarks:
                return False
            _, tipy = hand_landmarks[tip]
            _, mcpy = hand_landmarks[mcp]
            return tipy > mcpy  # más abajo en imagen

        folded_all = all([
            folded(INDEX_TIP, INDEX_MCP),
            folded(MIDDLE_TIP, MIDDLE_MCP),
            folded(RING_TIP, RING_MCP),
            folded(PINKY_TIP, PINKY_MCP)
        ])
        return folded_all
