import os
import time
import random
import itertools
import numpy as np

import GridworldUtils as GU
from learning import GridworldAgent as GA

import sys
sys.path.append('C:\\Users\\Nick\\Desktop\\Ava\\Programs')
from Library.General import Colors
from Library.Learning import EnvironmentUtils


### GRIDWORLD ###

class Gridworld(EnvironmentUtils.Environment):
    """ Base Gridworld object """


    def __init__(self, paths, name, height, width):
        """ """
        EnvironmentUtils.Environment.__init__(self, paths, name)
        
        # state params
        self.height = height
        self.width = width
        self.state_size = height * width
        self.state = np.zeros(self.state_size, dtype=int)
        self.grid = np.reshape(self.state, (height, width))


    ### ENVIRONMENT ###

    def reset_environment(self):
        """ default reset gridworld environment """
        for i in range(self.state_size):
            self.state[i] = 0
        self.set_terminal_states() # GW
        self.agents.reset_all() # AG
        for agent in self.agents.as_list():
            self.state[agent.state_idx] = agent.KEY

    def iterate_environment(self): 
        """ default iteration of gridworld environment """
        pass

    def in_terminal_state(self):
        """ checks if agent is in terminal state """
        return self.main_agent.in_terminal_state()

    def environment_move_agent(self, key, idx_old, idx_new):
        """ move agent to location """
        self.state[idx_old] = 0
        self.state[idx_new] = key


    ### AGENT ###

    def main_agent_take_action(self, action):
        """ main agent take action """
        action = self.main_agent.interpret_action(action)
        self.agent_action_move(self.main_agent, action)
        self.iterate_environment()
        self.main_agent.learn(action)

    def agent_action_move(self, agent, action):
        """ moves piece desired direction """
        # continue if valid move action
        pt_old = GU.state_to_grid(self, agent.state_idx)
        pt_new = pt_old + np.array(action)
        if not GU.is_valid_grid(self, pt_new):
            return
        # execute move if empty state
        idx = GU.grid_to_state(self, pt_new)
        if self.state[idx] == 0:
            self.environment_move_agent(agent.KEY, agent.state_idx, idx)
            agent.state_idx = idx


    ### RUNTIME ###

    def run_offline_episodes(self, n_episodes):
        """ run episodes offline """
        for _ in range(n_episodes):
            self.run_offline_episode()

    def run_offline_episode(self):
        """ run episode until gridworld in terminal state """
        self.reset_environment()
        while not self.in_terminal_state():
            self.main_agent_take_action()

    def run_online_step(self, action=None):
        """ run step online, resetting gridworld if necessary """
        if self.in_terminal_state():
            self.reset_environment()
        else:
            self.main_agent_take_action(action)
        self.draw_color_grid()


    ### DRAW SCREEN ###

    def draw_color_grid(self):
        """ default states  """
        self.draw_blank_grid()
        self.draw_color_grid_extra()
        self.draw_terminal_states()
        self.draw_agents()

    def draw_color_grid_extra(self):
        """ default draw extra grids """
        pass

    def draw_blank_grid(self):
        """ reset color grid to all white """
        self.color_grid = [Colors.WHITE for _ in self.state]

    def draw_terminal_states(self, color=Colors.ORANGE):
        """ color terminal states """
        for idx in self.terminal_states:
            self.color_grid[idx] = color

    def draw_agents(self):
        """ color agent depending on current state """
        for agent in self.agents.all:
            self.color_grid[agent.state_idx] = agent.get_color() # AG


