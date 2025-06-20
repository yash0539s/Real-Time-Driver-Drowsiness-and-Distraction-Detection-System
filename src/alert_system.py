import pygame
pygame.mixer.init()

class AlertSystem:
    def __init__(self, sound_file="D:\\siren-alert-96052.mp3", enable=True):
        self.enable, self.sound = enable, pygame.mixer.Sound(sound_file)

    def trigger(self):
        if self.enable:
            self.sound.play()
