import pygame
from math import sin, cos, pi
from time import sleep
import numpy as np
from typing import Tuple
from enum import Enum
import random
import os

os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (1500, 450)
BILDSCHIRM_X = 320
BILDSCHIRM_Y = 200


SCHLAEGER_LAENGE = 25
SCHLAEGER_GESCHWINDIGKEIT = 1
BEWEGUNG_PUNKTE = -0.1
TREFFER_PUNKTE = 1000
VERLOREN_PUNKTE = -1000
GEWONNEN_PUNKTE = 100


class Game:
    def __init__(self, mit_grafik: bool):
        self.schlaeger_y = BILDSCHIRM_Y / 2
        self.ball_x = 1
        self.ball_y = 100
        self.ball_geschwindigkeit = 5
        self.ball_richtung = 45 / 180 * pi

        self.grafik = False
        self.surface = None
        self.quit = False

        self.treffer = 0
        self.spiellaenge = 20
        if mit_grafik:
            pygame.init()
            pygame.display.set_mode(size=(BILDSCHIRM_X + 1, BILDSCHIRM_Y))
            self.grafik = mit_grafik
            self.surface = pygame.display.get_surface()

    def reset(self, winkel: int = 0) -> np.ndarray:
        """Setzt die Spielumgebung zurück.


        Args:
            mit_grafik (bool): Flag zur initialisierung des pyGame environments

        Returns:
            np.ndarray: Vector des aktuellen Zustands
        """

        self.quit = False
        self.treffer = 0
        self.schlaeger_y = BILDSCHIRM_Y / 2 + SCHLAEGER_LAENGE
        self.ball_x = 1
        self.ball_y = 100
        self.ball_geschwindigkeit = 5
        self.ball_richtung = random.randint(-winkel, winkel) / 180 * pi

        return np.array(
            [
                self.schlaeger_y,
                self.ball_x,
                self.ball_y,
                self.ball_geschwindigkeit,
                self.ball_richtung,
            ]
        )

    def pong_step(
        self, action: int, learning_mode: bool = True
    ) -> Tuple[np.ndarray, float, bool]:
        """Führe einen Spielschritt aus

        Args:
            action (int): 0: tu nichts, 1 hoch, 2 runter

        Returns:
            Tuple[np.ndarray, float, bool]: Zustand, Reward, Ende des Spiels
        """
        belohnung = 0
        if action == 1:
            if self.schlaeger_y >= 1:
                self.schlaeger_y -= SCHLAEGER_GESCHWINDIGKEIT
                belohnung += BEWEGUNG_PUNKTE
        elif action == 2:
            if self.schlaeger_y < BILDSCHIRM_Y - SCHLAEGER_LAENGE:
                self.schlaeger_y += SCHLAEGER_GESCHWINDIGKEIT
                belohnung += BEWEGUNG_PUNKTE
        self.ball_x = self.ball_x + self.ball_geschwindigkeit * cos(self.ball_richtung)
        self.ball_y = self.ball_y + self.ball_geschwindigkeit * sin(self.ball_richtung)

        if self.ball_x <= 0:
            self.ball_richtung = ((2 * pi - self.ball_richtung) + pi) % (2 * pi)
            self.ball_x = -self.ball_x
        if self.ball_y > BILDSCHIRM_Y:
            self.ball_richtung = (2 * pi) - self.ball_richtung
            self.ball_y = BILDSCHIRM_Y - (self.ball_y - BILDSCHIRM_Y)
        if self.ball_y < 0:
            self.ball_richtung = (2 * pi) - self.ball_richtung
            self.ball_y = -self.ball_y
        if self.ball_x >= BILDSCHIRM_X:
            belohnung += (
                BILDSCHIRM_Y
                - abs(self.schlaeger_y + SCHLAEGER_LAENGE / 2 - self.ball_y)
            ) / 4
            if (
                self.ball_y >= self.schlaeger_y
                and self.ball_y <= self.schlaeger_y + SCHLAEGER_LAENGE
            ):
                self.ball_richtung = ((2 * pi - self.ball_richtung) + pi) % (2 * pi)
                self.ball_x = BILDSCHIRM_X - (self.ball_x % BILDSCHIRM_X)
                belohnung += TREFFER_PUNKTE
                self.treffer += 1
                if self.treffer >= self.spiellaenge:
                    # belohnung += GEWONNEN_PUNKTE
                    self.quit = True
                if learning_mode:
                    self.quit = True
            else:
                self.quit = True
                belohnung += VERLOREN_PUNKTE
        if self.grafik:

            self.surface.fill(color=(0, 0, 0))
            pygame.draw.circle(
                surface=self.surface,
                color=(255, 255, 255),
                center=(self.ball_x, self.ball_y),
                radius=2,
                width=0,
            )
            pygame.draw.line(
                surface=self.surface,
                color=(255, 0, 0),
                start_pos=(BILDSCHIRM_X - 1, self.schlaeger_y),
                end_pos=(BILDSCHIRM_X - 1, self.schlaeger_y + SCHLAEGER_LAENGE),
            )
            if evts := pygame.event.get():
                for evt in evts:
                    if evt.type == pygame.QUIT:
                        quit = True
            pygame.display.flip()
            # sleep(0.1)
            # if quit:
            #     pygame.quit()
        return (
            np.array(
                [
                    self.schlaeger_y,
                    self.ball_x,
                    self.ball_y,
                    self.ball_geschwindigkeit,
                    self.ball_richtung,
                ]
            ),
            belohnung,
            self.quit,
        )

    def quit_grafik():
        pygame.quit()
