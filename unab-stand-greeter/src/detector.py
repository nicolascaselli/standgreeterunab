# -*- coding: utf-8 -*-
"""
Wrapper de MediaPipe para obtener pose (hombros) y manos (landmarks).
Devuelve un dict con:
- pose: {'left_shoulder': (x,y), 'right_shoulder': (x,y), 'left_wrist': (x,y), 'right_wrist': (x,y)}
- hands: lista de manos, cada una como {landmark_index: (x,y)}
Coordenadas normalizadas [0,1].
"""
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

class Detector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]

        pose_res = self.pose.process(frame_rgb)
        hands_res = self.hands.process(frame_rgb)

        out = {"pose": {}, "hands": []}

        # Pose landmarks
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            def p(idx):
                return (lm[idx].x, lm[idx].y)

            out["pose"] = {
                "left_shoulder": p(mp_pose.PoseLandmark.LEFT_SHOULDER),
                "right_shoulder": p(mp_pose.PoseLandmark.RIGHT_SHOULDER),
                "left_wrist": p(mp_pose.PoseLandmark.LEFT_WRIST),
                "right_wrist": p(mp_pose.PoseLandmark.RIGHT_WRIST),
            }

        # Hands
        if hands_res.multi_hand_landmarks:
            for hand in hands_res.multi_hand_landmarks:
                hand_dict = {}
                for i, l in enumerate(hand.landmark):
                    hand_dict[i] = (l.x, l.y)
                out["hands"].append(hand_dict)

        return out
