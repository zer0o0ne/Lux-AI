import sys
sys.path.append("kits/python/")

from MCTS import MMCTS
from estimator import Estimator
from utils import compute_start_mask, state_from_obs, to_json
from kits.python.lux.utils import direction_to
from kits.python.lux.kit import obs_to_game_state

from kits.python.luxai_s2.env import LuxAI_S2

import numpy as np
from copy import deepcopy
import torch
import json
from os import makedirs


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
        self.estimator = Estimator(self.agent_cfg["estimator_n_iter"])
        self.need_save = self.agent_cfg["need_save"]

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        if self.first:
            path = f'solution/train/{self.agent_cfg["game_n"]}_{self.player}'
            n_iter = self.agent_cfg["MMCTS_n_iter"]
            player_number = 0 if self.player == "player_0" else 1
            T = self.agent_cfg["T"]
            weights_path = self.agent_cfg["weights_path"]
            try:
                self.estimator.load_state_dict(torch.load(weights_path))
            except:
                pass
            self.MCTS = MMCTS(deepcopy(self.env.get_state()), self.env, self.estimator, path, n_iter, player_number, self.env_cfg, T, self.need_save)
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
            spawn_loc = np.argmax(p)
            spawn_loc = [spawn_loc // 48, spawn_loc % 48]

            state = state_from_obs(self.env.get_state(), obs, self.env_cfg, step)
            self.env.set_state(state)

            return dict(spawn=spawn_loc, metal=149, water=149)
        return {}
    

class AlgorithmicAgent(MMCTS_Agent):
    def __init__(self, player, env_cfg, agent_cfg):
        super().__init__(player, env_cfg, agent_cfg)

        class JustPath:
            def __init__(self, agent_cfg, player):
                self.path = f'solution/train/{agent_cfg["game_n"]}_{player}'

        self.MCTS = JustPath(agent_cfg, player)
        self.n = 0
        if self.need_save:
            makedirs(self.MCTS.path, exist_ok=True)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        actions = dict()
        self.n += 1
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        distribution = {}
        state = state_from_obs(self.env.get_state(), obs, self.env_cfg)
        for player in state.factories:
            fact_act, units_act = [], []
            factories = state.factories[player]
            state.teams[player].place_first
            factory_tiles, factory_units = [], []
            units = state.units[player]
            units_pos = [unit.pos for unit in units.values()]
            for unit_id, factory in factories.items():
                if 2 <= factory.cargo.water / 5 - 200:
                    fact_act.append(np.array([[0, 0, 1, 0]]))
                    if player == self.player: actions[unit_id] = 2
                elif factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST and \
                factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST and\
                not self._check(factory.pos, units_pos):
                    fact_act.append(np.array([[0, 1, 0, 0]]))
                    if player == self.player: actions[unit_id] = 1
                elif factory.power >= self.env_cfg.ROBOTS["LIGHT"].POWER_COST and \
                factory.cargo.metal >= self.env_cfg.ROBOTS["LIGHT"].METAL_COST and\
                not self._check(factory.pos, units_pos):
                    fact_act.append(np.array([[1, 0, 0, 0]]))
                    if player == self.player: actions[unit_id] = 0
                else:
                    fact_act.append(np.array([[0, 0, 0, 1]]))
                
                factory_tiles += [factory.pos.pos]
                factory_units += [factory]
            factory_tiles = np.array(factory_tiles)

            ice_map = state.board.ice
            ore_map = state.board.ore
            ice_tile_locations = np.argwhere(ice_map == 1)
            ore_tile_locations = np.argwhere(ore_map == 1)
            for unit_id, unit in units.items():
                
                if unit.unit_type.name == "HEAVY":
                    # track the closest factory
                    closest_factory = None
                    adjacent_to_factory = False
                    if len(factory_tiles) > 0:
                        factory_distances = np.mean((factory_tiles - unit.pos.pos) ** 2, 1)
                        closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                        closest_factory = factory_units[np.argmin(factory_distances)]
                        adjacent_to_factory = np.mean((closest_factory_tile - unit.pos.pos) ** 2) == 0

                        # previous ice mining code
                        if adjacent_to_factory and unit.power < 2000:
                            units_act.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]))
                            if player == self.player: actions[unit_id] = [np.array([2, 0, 4, 1000, 0, 1])]
                        elif unit.cargo.ice < 60:
                            ice_tile_distances = np.mean((ice_tile_locations - unit.pos.pos) ** 2, 1)
                            closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                            if np.all(closest_ice_tile == unit.pos.pos):
                                units_act.append(np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                if player == self.player: actions[unit_id] = [np.array([3, 0, 0, 0, 1, 1])]
                            else:
                                direction = direction_to(unit.pos.pos, closest_ice_tile)
                                if direction == 1: 
                                    units_act.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 1, 0, 0, 1, 1])]
                                if direction == 2: 
                                    units_act.append(np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 2, 0, 0, 1, 1])]
                                if direction == 3: 
                                    units_act.append(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 3, 0, 0, 1, 1])]
                                if direction == 4: 
                                    units_act.append(np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 4, 0, 0, 1, 1])]
                        # else if we have enough ice, we go back to the factory and dump it.
                        elif unit.cargo.ice >= 60:
                            direction = direction_to(unit.pos.pos, closest_factory_tile)
                            if adjacent_to_factory:
                                units_act.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]))
                                if player == self.player: 
                                    if direction == 0: actions[unit_id] = [np.array([1, 0, 0, unit.cargo.ice, 0, 1])]
                                    if direction == 1: actions[unit_id] = [np.array([1, 1, 0, unit.cargo.ice, 0, 1])]
                                    if direction == 2: actions[unit_id] = [np.array([1, 2, 0, unit.cargo.ice, 0, 1])]
                                    if direction == 3: actions[unit_id] = [np.array([1, 3, 0, unit.cargo.ice, 0, 1])]
                                    if direction == 4: actions[unit_id] = [np.array([1, 4, 0, unit.cargo.ice, 0, 1])]
                            else:
                                if direction == 1: 
                                    units_act.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 1, 0, 0, 1, 1])]
                                if direction == 2: 
                                    units_act.append(np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 2, 0, 0, 1, 1])]
                                if direction == 3: 
                                    units_act.append(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 3, 0, 0, 1, 1])]
                                if direction == 4: 
                                    units_act.append(np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 4, 0, 0, 1, 1])]

                        else:
                            units_act.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))

                elif unit.unit_type.name == "LIGHT":
                    # track the closest factory
                    closest_factory = None
                    adjacent_to_factory = False
                    if len(factory_tiles) > 0:
                        factory_distances = np.mean((factory_tiles - unit.pos.pos) ** 2, 1)
                        closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                        closest_factory = factory_units[np.argmin(factory_distances)]
                        adjacent_to_factory = np.mean((closest_factory_tile - unit.pos.pos) ** 2) == 0

                        # previous ice mining code
                        if adjacent_to_factory and unit.power < 100:
                            units_act.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]]))
                            if player == self.player: actions[unit_id] = [np.array([2, 0, 4, 50, 0, 1])]
                        elif unit.cargo.ore < 6:
                            ore_tile_distances = np.mean((ore_tile_locations - unit.pos.pos) ** 2, 1)
                            closest_ore_tile = ore_tile_locations[np.argmin(ore_tile_distances)]
                            if np.all(closest_ore_tile == unit.pos.pos):
                                units_act.append(np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                if player == self.player: actions[unit_id] = [np.array([3, 0, 0, 0, 1, 1])]
                            else:
                                direction = direction_to(unit.pos.pos, closest_ore_tile)
                                if direction == 1: 
                                    units_act.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 1, 0, 0, 1, 1])]
                                if direction == 2: 
                                    units_act.append(np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 2, 0, 0, 1, 1])]
                                if direction == 3: 
                                    units_act.append(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 3, 0, 0, 1, 1])]
                                if direction == 4: 
                                    units_act.append(np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 4, 0, 0, 1, 1])]
                        # else if we have enough ice, we go back to the factory and dump it.
                        elif unit.cargo.ore >= 6:
                            direction = direction_to(unit.pos.pos, closest_factory_tile)
                            if adjacent_to_factory:
                                units_act.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]))
                                if player == self.player: 
                                    if direction == 0: actions[unit_id] = [np.array([1, 0, 0, unit.cargo.ore, 0, 1])]
                                    if direction == 1: actions[unit_id] = [np.array([1, 1, 1, unit.cargo.ore, 0, 1])]
                                    if direction == 2: actions[unit_id] = [np.array([1, 2, 1, unit.cargo.ore, 0, 1])]
                                    if direction == 3: actions[unit_id] = [np.array([1, 3, 1, unit.cargo.ore, 0, 1])]
                                    if direction == 4: actions[unit_id] = [np.array([1, 4, 1, unit.cargo.ore, 0, 1])]
                            else:
                                if direction == 1: 
                                    units_act.append(np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 1, 0, 0, 1, 1])]
                                if direction == 2: 
                                    units_act.append(np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 2, 0, 0, 1, 1])]
                                if direction == 3: 
                                    units_act.append(np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 3, 0, 0, 1, 1])]
                                if direction == 4: 
                                    units_act.append(np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))
                                    if player == self.player: actions[unit_id] = [np.array([0, 4, 0, 0, 1, 1])]
                        else:
                            units_act.append(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]))

            factories = np.concatenate(fact_act) if len(fact_act) > 0 else 0
            units = np.concatenate(units_act) if len(units_act) > 0 else 0
            distribution[player] = {"factories": factories, "units": units}
            
        prepared_state = self.estimator._prepare_state(state)
        current_step = {"state": prepared_state, "distribution": distribution}
        path = self.MCTS.path + "/{}.json".format(self.n)
        if self.need_save:
            with open(path, 'w') as fp:
                json.dump(to_json(current_step), fp)
        return actions
    
    def _check(self, fact_pos, u_poss):
        for u_pos in  u_poss:
            if np.all(fact_pos.pos == u_pos.pos):
                return True
        return False
    