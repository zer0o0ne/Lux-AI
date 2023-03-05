import numpy as np
import json
from os import makedirs

from utils import *
from copy import deepcopy
from time import time

import torch.nn.functional as F

# TODO compare action queue with action for every robot
class MMCTS:
    def __init__(self, initial_state, simulator, estimator, path, n_iter, player_number, env_cfg, T = 1, need_save = False):

        #Support class which realizes graph logic
        class Node:
            def __init__(self, p, parent, action, raw_action = None, state = None, T = 1):
                self.p, self.u, self.T = p, T * p, T
                self.parent, self.state = parent, state
                self.action = action
                self.prepared_state, self.raw_action = None, raw_action
                self.Q, self.N, self.V = 0, 0, 0
                self.children = []

            def make_root(self):
                del self.parent
                self.parent = None

            def get_distribution(self):
                numbers = np.array([child.N for child in self.children]) 
                actions = [child.raw_action for child in self.children]

                sum = numbers.sum()
                distribution = numbers / (sum + 1e-9)
                probabilities_for_train = mean_across_dicts(list(zip(actions, numbers)), sum)
                return distribution, probabilities_for_train

            def get_leaf(self, n = 1):
                if len(self.children) == 0: return self, -1
                scores = [child.Q + child.u for child in self.children]
                leaf, score = self.children[np.argmax(scores)].get_leaf(n + 1)
                if score < 0:
                    score = n
                return leaf, score

            def inspect(self, raw_actions, env_actions, P, v, prepared_state, predictions, ids):
                assert len(P) == len(env_actions)

                # Need to future train 
                self.prepared_state = prepared_state
                self.predictions = predictions
                self.ids, self.v = ids, v

                for i in range(len(P)):
                    self.children.append(Node(P[i], self, env_actions[i], raw_actions[i], T = self.T))
                update(self, v)

            def _compute_action_queue(self, unit_id, player, n_in = 0):
                if len(self.children) == 0 or n_in == 20: return []

                numbers = np.array([child.N for child in self.children]) 
                best_child = np.argmax(numbers)
                if unit_id in self.children[best_child].action[player]:
                    return self.children[best_child].action[player][unit_id] + self.children[best_child]._compute_action_queue(unit_id, player, n_in + 1)
                else:
                    return [np.array([5, 0, 0, 0, 0, 1])] + self.children[best_child]._compute_action_queue(unit_id, player, n_in + 1)
                
            def update_states(self, simulator):
                for child in self.children:
                    for player in child.action:
                        for unit_id in child.action[player]:
                            if unit_id not in self.state.factories[player] and unit_id not in self.state.units[player] and unit_id in child.action:
                                child.action.pop(unit_id)

                    if child.state is not None:
                        simulator.set_state(deepcopy(self.state))
                        simulator.step(child.action)
                        child.state = deepcopy(simulator.get_state())
                        child.update_states(simulator)

        self.simulator = simulator
        self.estimator = estimator
        self.need_save = need_save
        self.path, self.n_iter, self.n = path, n_iter, 0
        self.root = Node(1, None, None, None, initial_state, T)
        self.env_cfg, self.player_number = env_cfg, player_number
        if need_save:
            makedirs(path, exist_ok=True)

    def _simulation_step(self): 
        current_node, depth = self.root.get_leaf()
        if depth > 25:
            return False
        if current_node.state is None:
            self.simulator.set_state(deepcopy(current_node.parent.state))
            _, _, dones, _ = self.simulator.step(current_node.action)
            if dones["player_0"] or dones["player_1"]:
                return True
            current_node.state = deepcopy(self.simulator.get_state())
        raw_actions, env_actions, P, v, prepared_state, predictions, ids = self.estimator.predict(deepcopy(current_node.state), mode = "full", player_number = self.player_number)
        current_node.inspect(raw_actions, env_actions, P, v, prepared_state, predictions, ids)
        return False

    def step(self, step, obs):
        time_ = time()
        state = state_from_obs(self.simulator.get_state(), obs, self.env_cfg, step)
        self._check_state(state)

        for i in range(self.n_iter):
            if time() - time_ < 2.85:
                if self._simulation_step():
                    self.simulator.reset()
                    self.simulator.env_cfg.verbose, self.env_cfg.verbose = 0, 0

        self.n += 1
        distribution, probabilities_for_train = self.root.get_distribution()
        current_step = {"state": self.root.prepared_state, "distribution": probabilities_for_train}

        path = self.path + "/{}.json".format(self.n)
        if self.need_save and np.sum(distribution) > 0:
            with open(path, 'w') as fp:
                json.dump(to_json(current_step), fp)
        
        predicted_action = self.root.children[np.argmax(distribution)].action["player_{}".format(self.player_number)]
        actual_queue_state = state.units["player_{}".format(self.player_number)]
        predicted_action = self._compare_action_and_queue(actual_queue_state, predicted_action, "player_{}".format(self.player_number))
        return predicted_action
    
    def _check_state(self, state):
        for current_node in self.root.children:
            if current_node.state is None:
                continue
            if state == current_node.state:
                self.root = current_node
                self.root.make_root()
                # self.root.state = deepcopy(state)
                return True
                
        # best_diff = 0.01
        # best_root = None
        # for current_node in self.root.children:
        #     if current_node.state is None:
        #         continue
        #     diff = self._difference(state, current_node.predictions, current_node.ids, current_node.v)
        #     if diff < best_diff:
        #         best_root = current_node
        #         best_diff = diff
        # if best_root is not None:
        #     self.root = best_root
        #     self.root.make_root()
        #     self.root.state = deepcopy(state)
        #     self.root.update_states(self.simulator)
        #     return True
        
        self.root.state = state
        self.root.children.clear()
        return False
    
    def _compare_action_and_queue(self, actual_queue_state, predicted_action, player):
        for unit_id in actual_queue_state:
            if len(actual_queue_state[unit_id].action_queue) == 0:
                predicted_action[unit_id] = self.root._compute_action_queue(unit_id, player)
                continue
            if unit_id not in predicted_action:
                if actual_queue_state[unit_id].action_queue[0].act_type == "recharge":
                    continue
                else:
                    predicted_action[unit_id] = self.root._compute_action_queue(unit_id, player)
                    continue
            if compare_action_and_array(actual_queue_state[unit_id].action_queue[0], predicted_action[unit_id][0]):
                predicted_action.pop(unit_id)
            else:
                predicted_action[unit_id] = self.root._compute_action_queue(unit_id, player)
                continue
        
        candidates = list(predicted_action.keys())
        for unit_id in candidates:
            if unit_id not in actual_queue_state and "unit" in unit_id:
                 predicted_action.pop(unit_id)
        return predicted_action

    def _difference(self, state, predictions, ids, v):
        actual_predictions, actual_ids, actual_v = self.estimator.predict(state, mode = "fast", player_number = self.player_number)
        diff = abs(v - actual_v)
        if not isequal(ids, actual_ids): 
            return 1
        for player, prediction in actual_predictions.items():
            for unit_type in ["factories", "units"]:
                if unit_type not in prediction or unit_type not in predictions[player]: return 1
                if not type(prediction[unit_type]) == type(predictions[player][unit_type]): return 1
                if len(prediction[unit_type]) == 0: continue
                if not prediction[unit_type].shape == predictions[player][unit_type].shape: return 1
                diff += torch.abs(F.softmax(prediction[unit_type], -1) - F.softmax(predictions[player][unit_type], -1)).mean(dim = 1).sum().item() / 4
        
        return diff
        

        