from MCTS import MMCTS
from estimator import Estimator
from utils import compute_start_mask, state_from_obs

from kits.python.luxai_s2.env import LuxAI_S2

import numpy as np
from copy import deepcopy
import torch

class MMCTS_Agent():
    def __init__(self, player, env_cfg, agent_cfg):
        self.agent_cfg = agent_cfg
        self.first = True
        self.player = player
        self.env_cfg = env_cfg
        self.env = LuxAI_S2()
        self.env.reset()
        self.env.env_cfg.verbose, self.env_cfg.verbose = 0, 0
        self.start_mask = None

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        if self.first:
            estimator = Estimator(self.agent_cfg["estimator_n_iter"])
            path = f'solution/train/{self.agent_cfg["game_n"]}_{self.player}'
            n_iter = self.agent_cfg["MMCTS_n_iter"]
            player_number = 0 if self.player == "player_0" else 1
            T = self.agent_cfg["T"]
            need_save = self.agent_cfg["need_save"]
            weights_path = self.agent_cfg["weights_path"]
            try:
                estimator.load_state_dict(torch.load(weights_path))
            except:
                pass
            self.MCTS = MMCTS(deepcopy(self.env.get_state()), self.env, estimator, path, n_iter, player_number, self.env_cfg, T, need_save)
            self.first = False
        
        action = self.MCTS.step(step, obs)
        return action

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if step == 0:
            return dict(faction="AlphaStrike", bid=obs["board"]["factories_per_team"])
        
        my_turn_to_place = step % 2 == 1 if obs["teams"][self.player]["place_first"] == True else step % 2 == 0
        if my_turn_to_place:
            if self.start_mask is None:
                self.start_mask = compute_start_mask(obs)
            mask = self.start_mask * obs["board"]["valid_spawns_mask"]
            p = mask.reshape(-1) / np.sum(mask)
            spawn_loc = np.random.choice(48 * 48, p = p)
            spawn_loc = [spawn_loc // 48, spawn_loc % 48]

            state = state_from_obs(self.env.get_state(), obs, self.env_cfg, step)
            self.env.set_state(state)

            return dict(spawn=spawn_loc, metal=149, water=149)
        return {}
    
        