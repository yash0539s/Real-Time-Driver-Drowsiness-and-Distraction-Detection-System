import pygame
import time
import os

class AlertSystem:
    def __init__(self, sound_file="D:\\siren-alert-96052.mp3", enable=True, cooldown=3.0):
        self.enable = enable
        self.cooldown = cooldown  # seconds
        self.last_triggered = 0

        pygame.mixer.init()
        if not os.path.exists(sound_file):
            raise FileNotFoundError(f"❌ Sound file not found: {sound_file}")

        try:
            self.sound = pygame.mixer.Sound(sound_file)
        except Exception as e:
            raise RuntimeError(f"⚠️ Failed to load sound: {e}")

    def trigger(self):
        if not self.enable:
            return

        current_time = time.time()
        if current_time - self.last_triggered >= self.cooldown:
            self.sound.play()
            self.last_triggered = current_time
