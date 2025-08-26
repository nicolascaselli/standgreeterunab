# -*- coding: utf-8 -*-
"""
Detector de pulgar arriba 👍 con información de depuración.
Reglas:
- Pulgar extendido: tip del pulgar por encima de la muñeca (y menor).
- Resto de dedos plegados: tip por debajo de su MCP (y mayor).
"""
from typing import Dict

THUMB_TIP = 4
WRIST = 0
INDEX_TIP, INDEX_MCP   = 8, 5
MIDDLE_TIP, MIDDLE_MCP = 12, 9
RING_TIP, RING_MCP     = 16, 13
PINKY_TIP, PINKY_MCP   = 20, 17

class ThumbsUpDetector:
    def __init__(self, tip_above_wrist_margin=0.02, require_folded=True):
        self.tip_above_wrist_margin = tip_above_wrist_margin
        self.require_folded = require_folded
        self._last_debug = {"thumb_up": False, "folded_all": False, "folded": [False]*4}

    def is_thumbs_up(self, hand_landmarks: Dict[int, tuple]) -> bool:
        dbg = self.debug(hand_landmarks)
        return dbg["thumb_up"] and (dbg["folded_all"] or not self.require_folded)

    def debug(self, hand_landmarks: Dict[int, tuple]):
        if WRIST not in hand_landmarks or THUMB_TIP not in hand_landmarks:
            self._last_debug = {"thumb_up": False, "folded_all": False, "folded": [False]*4}
            return self._last_debug

        _, wy = hand_landmarks[WRIST]
        _, ty = hand_landmarks[THUMB_TIP]
        thumb_up = ty < (wy - self.tip_above_wrist_margin)

        def folded(tip, mcp):
            if tip not in hand_landmarks or mcp not in hand_landmarks:
                return False
            _, tipy = hand_landmarks[tip]
            _, mcpy = hand_landmarks[mcp]
            return tipy > mcpy

        folded_list = [
            folded(INDEX_TIP, INDEX_MCP),
            folded(MIDDLE_TIP, MIDDLE_MCP),
            folded(RING_TIP, RING_MCP),
            folded(PINKY_TIP, PINKY_MCP)
        ]
        folded_all = all(folded_list)

        self._last_debug = {"thumb_up": bool(thumb_up), "folded_all": bool(folded_all), "folded": folded_list}
        return self._last_debug

    def last_debug(self):
        return dict(self._last_debug)
