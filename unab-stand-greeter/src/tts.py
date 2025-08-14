# -*- coding: utf-8 -*-
"""
Wrapper de TTS offline con pyttsx3.
"""
import pyttsx3
import threading

class TTS:
    def __init__(self, enabled=True, rate=175, volume=1.0):
        self.enabled = enabled
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        self._lock = threading.Lock()

    def set_enabled(self, on: bool):
        self.enabled = on

    def speak_async(self, text: str):
        if not self.enabled:
            return
        def run():
            with self._lock:
                self.engine.say(text)
                self.engine.runAndWait()
        threading.Thread(target=run, daemon=True).start()
