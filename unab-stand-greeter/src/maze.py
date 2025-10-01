# src/maze.py
# -*- coding: utf-8 -*-
"""
Mini-juego de Laberinto para UNAB Greeter (nivel configurable).
- Beginner:
  * Menos celdas (laberinto más pequeño) => pasillos más anchos.
  * Colisión más permisiva (muros de colisión más delgados).
  * Inicio/Meta más grandes.
  * Trazo más visible.
- Render a prueba de desbordes y trazo pintado en tiempo real.
"""
from dataclasses import dataclass
import numpy as np
import cv2
import random
import time
from collections import deque
from typing import Optional, Tuple

@dataclass
class MazeStatus:
    playing: bool
    life_lost: bool
    win: bool
    game_over: bool

def _carve_maze(cols: int, rows: int, rng: random.Random):
    """Genera laberinto binario (1=pared, 0=camino) de tamaño (rows*2+1) x (cols*2+1)."""
    H = rows * 2 + 1
    W = cols * 2 + 1
    M = np.ones((H, W), dtype=np.uint8)      # 1=pared, 0=camino
    visited = np.zeros((rows, cols), dtype=np.uint8)

    def carve(r, c):
        visited[r, c] = 1
        M[r*2+1, c*2+1] = 0
        dirs = [(0,1),(1,0),(0,-1),(-1,0)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and visited[nr, nc] == 0:
                M[r*2+1+dr, c*2+1+dc] = 0
                carve(nr, nc)

    carve(0, 0)
    return M

class MazeGame:
    def __init__(
        self,
        screen_w: int,
        screen_h: int,
        # --- dificultad / tamaño ---
        grid_cols: Optional[int] = None,
        grid_rows: Optional[int] = None,
        panel_ratio_w: float = 0.70,
        panel_ratio_h: float = 0.60,
        seed: Optional[int] = None,
        # --- estilo / tolerancias ---
        thin_ratio: float = 0.30,         # adelgazado visual de paredes (0..0.6 recomendado)
        collision_rel: float = 0.006,     # radio relativo de colisión (menor = más permisivo)
        start_goal_pad: float = 0.25,     # padding relativo (en múltiplos de cell_px) para inicio/meta
        trail_thickness_rel: float = 0.020 # grosor del trazo relativo al min(screen_w, screen_h)
    ):
        """
        Si grid_cols/rows es None, se usan valores base fáciles.
        - collision_rel: p.ej. 0.006 ~ 6px por cada 1000px de lado menor.
        - start_goal_pad: 0.25 => expande inicio/meta aprox un cuarto de celda por lado.
        """
        self.sw, self.sh = int(screen_w), int(screen_h)
        self.panel_w = int(self.sw * panel_ratio_w)
        self.panel_h = int(self.sh * panel_ratio_h)

        # Tamaño de laberinto por defecto (BEGINNER)
        if grid_cols is None or grid_rows is None:
            base_cols = 11  # menos celdas => pasillos más anchos
            base_rows = 7
            grid_cols = base_cols
            grid_rows = base_rows
        self.cols = int(grid_cols)
        self.rows = int(grid_rows)

        self.rng = random.Random(seed if seed is not None else int(time.time()*1000) & 0xFFFFFFFF)

        # Malla binaria (1 pared, 0 camino)
        self.grid = _carve_maze(self.cols, self.rows, self.rng)   # shape: (rows*2+1, cols*2+1)
        gh, gw = self.grid.shape[:2]

        # Escala a píxeles del panel
        self.cell_px = min(self.panel_w // gw, self.panel_h // gh)
        self.cell_px = max(self.cell_px, 6)
        self.canvas_w = gw * self.cell_px
        self.canvas_h = gh * self.cell_px

        # Origen del panel centrado
        self.x0 = (self.sw - self.canvas_w) // 2
        self.y0 = (self.sh - self.canvas_h) // 2

        # Máscaras
        small = (self.grid * 255).astype(np.uint8)
        big = cv2.resize(small, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_NEAREST)
        self.wall_mask = big.copy()  # 255=pared, 0=camino

        # Colisión: dilatación suave (más pequeño => más permisivo)
        # radio en px relativo al tamaño de pantalla
        base = min(self.sw, self.sh)
        finger_r = max(3, int(collision_rel * base))
        if finger_r % 2 == 0:
            finger_r += 1  # preferir impar para kernel
        k_col = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, finger_r*2-1), max(1, finger_r*2-1)))
        self.wall_mask_dil = cv2.dilate(self.wall_mask, k_col, iterations=1)

        # Visual: paredes adelgazadas
        shrink_px = max(1, int(self.cell_px * float(thin_ratio)))
        shrink_px = min(shrink_px, max(1, self.cell_px - 2))
        k_vis = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shrink_px*2+1, shrink_px*2+1))
        self.wall_mask_vis = cv2.erode(self.wall_mask, k_vis, iterations=1)

        # Canvas visual
        self.canvas_bgr = np.full((self.canvas_h, self.canvas_w, 3), (245, 245, 245), dtype=np.uint8)
        self.canvas_bgr[self.wall_mask_vis == 255] = (25, 25, 25)

        # Inicio/Meta (en celdas pequeñas)
        self.start_cell = (1, 1)
        self.goal_cell  = (gw-2, gh-2)
        self.start_rect = self._cell_rect(*self.start_cell, self.cell_px)
        self.goal_rect  = self._cell_rect(*self.goal_cell, self.cell_px)

        # Expandir inicio/meta (BEGINNER)
        pad = int(self.cell_px * float(start_goal_pad))
        self.start_rect = self._pad_rect(self.start_rect, pad, self.canvas_w, self.canvas_h)
        self.goal_rect  = self._pad_rect(self.goal_rect,  pad, self.canvas_w, self.canvas_h)

        # Estado
        self.lives = 3
        self.started = False
        self.win_flag = False
        self.life_cooldown_until = 0.0

        # Trazo
        self.path_pts = deque(maxlen=4000)
        self.last_pt = None
        self.trail_thickness = max(6, int(trail_thickness_rel * base))
        self.trail_color = (50, 140, 255)

    @staticmethod
    def _pad_rect(rect, pad, max_w, max_h):
        x1, y1, x2, y2 = rect
        return (
            max(0, x1 - pad),
            max(0, y1 - pad),
            min(max_w - 1, x2 + pad),
            min(max_h - 1, y2 + pad),
        )

    def _cell_rect(self, cx, cy, cell_px):
        x1 = cx * cell_px
        y1 = cy * cell_px
        x2 = (cx+1) * cell_px - 1
        y2 = (cy+1) * cell_px - 1
        return (x1, y1, x2, y2)

    def _inside_rect(self, x, y, rect):
        x1, y1, x2, y2 = rect
        return (x1 <= x <= x2) and (y1 <= y <= y2)

    def reset(self):
        self.lives = 3
        self.started = False
        self.win_flag = False
        self.life_cooldown_until = 0.0
        self.path_pts.clear()
        self.last_pt = None

    def update(self, finger_px: Optional[Tuple[int, int]], now: float) -> MazeStatus:
        life_lost = False
        if self.win_flag:
            return MazeStatus(playing=False, life_lost=False, win=True, game_over=False)

        if finger_px is not None:
            fx, fy = finger_px
            lx, ly = fx - self.x0, fy - self.y0
            inside_panel = (0 <= lx < self.canvas_w and 0 <= ly < self.canvas_h)
        else:
            inside_panel = False

        # Espera tocar INICIO para comenzar
        if not self.started:
            if inside_panel and self._inside_rect(lx, ly, self.start_rect):
                self.started = True
                self.path_pts.clear()
                self.last_pt = (lx, ly)
            return MazeStatus(playing=True, life_lost=False, win=False, game_over=False)

        # Jugando
        if inside_panel:
            if self.last_pt is None or (abs(lx - self.last_pt[0]) + abs(ly - self.last_pt[1]) > 1):
                self.path_pts.append((lx, ly))
                self.last_pt = (lx, ly)

            # Colisión (con cooldown)
            if now >= self.life_cooldown_until:
                if self.wall_mask_dil[ly, lx] == 255:
                    self.lives -= 1
                    life_lost = True
                    self.started = False
                    self.life_cooldown_until = now + 0.7
                    self.path_pts.clear()
                    self.last_pt = None
                    if self.lives <= 0:
                        return MazeStatus(playing=False, life_lost=True, win=False, game_over=True)

            # Meta
            if self._inside_rect(lx, ly, self.goal_rect):
                self.win_flag = True
                return MazeStatus(playing=False, life_lost=False, win=True, game_over=False)

        return MazeStatus(playing=True, life_lost=life_lost, win=False, game_over=False)

    def render(self, frame_bgr, font=None):
        """
        Render del laberinto + HUD (vidas e instrucciones).
        """
        view = frame_bgr
        panel = self.canvas_bgr.copy()

        # Inicio/Meta visibles
        sx1, sy1, sx2, sy2 = self.start_rect
        gx1, gy1, gx2, gy2 = self.goal_rect
        cv2.rectangle(panel, (sx1, sy1), (sx2, sy2), (60, 180, 75), -1)
        cv2.rectangle(panel, (gx1, gy1), (gx2, gy2), (80, 180, 255), -1)

        # Trazo pintado
        if len(self.path_pts) >= 2:
            pts = np.array(self.path_pts, dtype=np.int32)
            cv2.polylines(panel, [pts], isClosed=False, color=self.trail_color,
                          thickness=self.trail_thickness, lineType=cv2.LINE_AA)
        if len(self.path_pts) >= 1:
            tail_n = max(1, self.trail_thickness * 3)
            for (px, py) in list(self.path_pts)[-tail_n:]:
                cv2.circle(panel, (px, py), self.trail_thickness//2, self.trail_color, -1, cv2.LINE_AA)

        # Pegado robusto
        H, W = view.shape[:2]
        x1 = max(0, self.x0); y1 = max(0, self.y0)
        x2 = min(self.x0 + self.canvas_w, W); y2 = min(self.y0 + self.canvas_h, H)
        if x1 < x2 and y1 < y2:
            px1 = max(0, -self.x0); py1 = max(0, -self.y0)
            px2 = px1 + (x2 - x1);  py2 = py1 + (y2 - y1)
            roi = view[y1:y2, x1:x2]
            sub = panel[py1:py2, px1:px2]
            cv2.addWeighted(sub, 0.95, roi, 0.05, 0, roi)
            view[y1:y2, x1:x2] = roi
            cv2.rectangle(view, (x1-2, y1-2), (x2+2, y2+2), (255,255,255), 2)

        # HUD
        lives_text = f"Vidas: {'❤'*self.lives}{' '*(3-self.lives)}"
        hud_y_above = (self.y0 - 10)
        hud_y_below = (self.y0 + self.canvas_h + 10)
        put_above = (hud_y_above > 30)

        if font is None:
            if put_above:
                cv2.putText(view, lives_text, (max(10, self.x0+10), hud_y_above),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(view, "Toca INICIO para comenzar. Recorre sin tocar paredes.",
                            (max(10, self.x0+10), hud_y_above - 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
            else:
                cv2.putText(view, lives_text, (max(10, self.x0+10), min(H-10, hud_y_below)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                cv2.putText(view, "Toca INICIO para comenzar. Recorre sin tocar paredes.",
                            (max(10, self.x0+10), min(H-30, hud_y_below + 24)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,220,220), 1)
        else:
            view = self._draw_text(view, lives_text,
                                   (max(10, self.x0+10), (hud_y_above if put_above else min(H-20, hud_y_below))),
                                   font, (0,0,255))
            view = self._draw_text(view, "Toca INICIO para comenzar. Recorre sin tocar paredes.",
                                   (max(10, self.x0+10), (hud_y_above-26 if put_above else min(H-40, hud_y_below + 24))),
                                   font, (230,230,230))
        return view

    @staticmethod
    def _draw_text(img, text, org, font, color=(255,255,255)):
        from PIL import ImageFont, Image, ImageDraw
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        draw.text((org[0] + 1, org[1] + 1), text, font=font, fill=(0, 0, 0))
        draw.text(org, text, font=font, fill=color)
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
