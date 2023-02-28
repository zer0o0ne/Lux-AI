import numpy as np
import json
from os import makedirs

from utils import *
from copy import deepcopy
from time import time

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

            def get_leaf(self):
                if len(self.children) == 0: return self
                scores = [child.Q + child.u for child in self.children]
                return self.children[np.argmax(scores)].get_leaf()

            def inspect(self, raw_actions, env_actions, P, v, prepared_state):
                assert len(P) == len(env_actions)

                # Need to feature train 
                self.prepared_state = prepared_state

                for i in range(len(P)):
                    self.children.append(Node(P[i], self, env_actions[i], raw_actions[i], T = self.T))
                update(self, v)

            def _compute_action_queue(self, unit_id, n_in = 0):
                if len(self.children) == 0 or n_in == 20: return []

                numbers = np.array([child.N for child in self.children]) 
                sum = numbers.sum()
                if sum == 0:
                    return []
                distribution = numbers / sum
                best_child = np.argmax(distribution)
                if unit_id in self.children[best_child].action:
                    return self.children[best_child].action[unit_id] + self.children[best_child]._compute_action_queue(unit_id, n_in + 1)
                else:
                    return [np.array([5, 0, 0, 0, 0, 1])] + self.children[best_child]._compute_action_queue(unit_id, n_in + 1)

        self.simulator = simulator
        self.estimator = estimator
        self.need_save = need_save
        self.path, self.n_iter, self.n = path, n_iter, 0
        self.root = Node(1, None, None, None, initial_state, T)
        self.env_cfg, self.player_number = env_cfg, player_number
        makedirs(path, exist_ok=True)

    def _simulation_step(self): 
        current_node = self.root.get_leaf()
        if current_node.state is None:
            self.simulator.set_state(deepcopy(current_node.parent.state))
            _, _, dones, _ = self.simulator.step(current_node.action)
            if dones["player_0"] or dones["player_1"]:
                return True
            current_node.state = deepcopy(self.simulator.get_state())
        raw_actions, env_actions, P, v, prepared_state = self.estimator.predict(current_node.state, mode = "fast", player_number = self.player_number)
        current_node.inspect(raw_actions, env_actions, P, v, prepared_state)
        return False

    def step(self, step, obs):
        state = state_from_obs(self.simulator.get_state(), obs, self.env_cfg, step)
        self._check_state(state)
        
        time_ = time()
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
        predicted_action = self._compare_action_and_queue(actual_queue_state, predicted_action)
        return predicted_action
    
    def _check_state(self, state):
        for current_node in self.root.children:
            if current_node.state is None:
                continue
                # Not necessary
                # self.simulator.set_state(current_node.parent.state)
                # self.simulator.step(current_node.action)
                # current_node.state = self.simulator.get_state()
            if state == current_node.state:
                self.root = current_node
                self.root.make_root()
                return True
        self.root.state = state
        self.root.children.clear()
        return False
    
    def _compare_action_and_queue(self, actual_queue_state, predicted_action):
        for unit_id in actual_queue_state:
            if unit_id not in predicted_action:
                predicted_action[unit_id] = self.root._compute_action_queue(unit_id)
                continue
            if len(actual_queue_state[unit_id].action_queue) == 0:
                predicted_action[unit_id] = self.root._compute_action_queue(unit_id)
                continue
            if len(actual_queue_state[unit_id].action_queue) == 0:
                predicted_action[unit_id] = self.root._compute_action_queue(unit_id)
                continue
            if np.all(actual_queue_state[unit_id].action_queue[0] == predicted_action[unit_id]):
                predicted_action.pop(unit_id)
            else:
                predicted_action[unit_id] = self.root._compute_action_queue(unit_id)
                continue

        return predicted_action




        

        