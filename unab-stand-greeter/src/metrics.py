# -*- coding: utf-8 -*-
"""
Métricas simples a CSV.
"""
import csv
import os
import time

class Metrics:
    def __init__(self, path="metrics.csv"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["timestamp", "event", "details"])

    def log(self, event, details=""):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([int(time.time()), event, details])
