import pygame
from pygame.locals import *
import time

def main() :
    pygame.init()

    while True:

        eventlist = pygame.event.get()

        eventlist = filter(lambda e : e.type == pygame.locals.JOYBUTTONDOWN , eventlist)

        print(map(lambda x : x.button,eventlist))

        time.sleep(0.1)

if __name__ == '__main__':
    pygame.joystick.init()
    try:
        joys = pygame.joystick.Joystick(0)
        joys.init()
        main()
    except pygame.error:
        print('ジョイスティックを差し込め')
