import pygame

import GridworldUtils


### PYGAME ###

def run():
    """ initiate Gridworld with reinforcement lerning """
    gridworld.reset_environment()
    initialize_pygame()
    done = False
    while not done:
        done = check_pygame_events()
        if not manual_mode:
            gridworld.run_online_step()
        draw_pygame_screen()
        end_pygame_loop()
    exit_pygame()

def initialize_pygame():
    """ run pygame.init(), build screen, start clock """
    global screen, clock
    pygame.init()
    pygame.display.set_caption(gridworld.name)
    h = HEIGHT * gridworld.height + MARGIN * (gridworld.height + 1)
    w = WIDTH * gridworld.width + MARGIN * (gridworld.width + 1)
    screen = pygame.display.set_mode((w + 128, h))
    clock = pygame.time.Clock()

def check_pygame_events():
    """ loop over pygame events, exiting or execute actions """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        check_logistic_events(event)
        check_gridworld_events(event)
    return False

def draw_pygame_screen():
    """ draw gridworld on screen """
    screen.fill((0,0,0))
    for i in range(gridworld.height):
        for j in range(gridworld.width):
            color = gridworld.color_grid[i * gridworld.width + j]
            x = (MARGIN + WIDTH) * j + MARGIN
            y = (MARGIN + HEIGHT) * i + MARGIN
            pygame.draw.rect(screen, color, [x, y, WIDTH, WIDTH])
    # draw side panel
    x = (MARGIN + WIDTH) * (j+1) + MARGIN
    y = (MARGIN + HEIGHT) * (i+1) - MARGIN
    pygame.draw.rect(screen, (255,255,255), [x, MARGIN, 128-MARGIN, y])    

def end_pygame_loop():
    """ tick counter and draw screen """
    clock.tick(GAME_SPEED)
    pygame.display.flip()

def exit_pygame():
    """ close display """
    pygame.display.quit()
    pygame.quit()


### EVENTS ###

def check_logistic_events(event):
    """ swicth between manual and automated, switch gridworld """
    global manual_mode
    # change between manual and automated
    if event.type == pygame.KEYDOWN:
        if event.unicode == 'a':
            manual_mode = not manual_mode
    # change gridworld
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_1:
            start_gridworld(1)
        if event.key == pygame.K_2:
            start_gridworld(2)
        if event.key == pygame.K_3:
            start_gridworld(3)
        if event.key == pygame.K_4:
            start_gridworld(4)
        if event.key == pygame.K_5:
            start_gridworld(5)
    
def check_gridworld_events(event):
    """ """
    # take actions in gridworld
    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_LEFT:
            gridworld.run_online_step('left')
        if event.key == pygame.K_RIGHT:
            gridworld.run_online_step('right')
        if event.key == pygame.K_UP:
            gridworld.run_online_step('up')
        if event.key == pygame.K_DOWN:
            gridworld.run_online_step('down')


### GRIDWORLD ###

def start_gridworld(version):
    """ define gridworld by version number """
    global gridworld
    exit_pygame()
    gridworld = GridworldUtils.get_gridworld(version)
    initialize_pygame()


### PARAMS ###

# pygame
gridworld = None
screen = None
clock = None

# gridworld
manual_mode = 0
INITIAL_GRIDWORLD = 1
N_PRE_EPISODES = 0
N_SKIP_EPISODES = 0

# screen
GAME_SPEED = 10
HEIGHT = 80
WIDTH = 80
MARGIN = 16


### PROGRAM ###

if __name__ == '__main__':
    
    # initialize gridworld
    gridworld = GridworldUtils.get_gridworld(INITIAL_GRIDWORLD)

    # run previous episodes
    if N_PRE_EPISODES > 0:
        print('Running {} episodes...'.format(N_PRE_EPISODES))
        gridworld.run_offline_episodes(n_episodes=N_PRE_EPISODES)

    # run gridworld
    if 1:
        print('Running program...')
        run()


