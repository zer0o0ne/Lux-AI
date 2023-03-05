from copy import deepcopy
import torch

def identity(state, distribution):
    return deepcopy(state), deepcopy(distribution)

def ninety_degrees(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.rot90(state["board"], k = 1, dims = (2, 3))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [15 - units_state[1][i][1], units_state[1][i][0]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = 2 - units_state[0][0, i, -1], units_state[0][0, i, -2]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 3] = distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 3], distribution[player][unit_type][i, 0]
    return state, distribution


def hundred_eighteen_degrees(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.rot90(state["board"], k = 2, dims = (2, 3))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [15 -units_state[1][i][1], 15 - units_state[1][i][0]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = 2 - units_state[0][0, i, -1], 2 - units_state[0][0, i, -2]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 3] = distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1]
    return state, distribution

def two_hundred_seventeen_degrees(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.rot90(state["board"], k = 3, dims = (2, 3))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [units_state[1][i][1], 15 - units_state[1][i][0]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = units_state[0][0, i, -1], 2 - units_state[0][0, i, -2]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 3] = distribution[player][unit_type][i, 3], distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2]
    return state, distribution

def side_diagonal_flip(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.rot90(state["board"], k = 1, dims = (2, 3))
    state["board"] = torch.flip(state["board"], dims = (3,))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [15 - units_state[1][i][1], 15 - units_state[1][i][0]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = 2 - units_state[0][0, i, -1], 2 - units_state[0][0, i, -2]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 3] = distribution[player][unit_type][i, 3], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 0]
    return state, distribution

def main_diagonal_flip(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.rot90(state["board"], k = 1, dims = (2, 3))
    state["board"] = torch.flip(state["board"], dims = (2,))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [units_state[1][i][1], units_state[1][i][0]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = units_state[0][0, i, -1], units_state[0][0, i, -2]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 3] = distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 3], distribution[player][unit_type][i, 2]
    return state, distribution

def vertical_flip(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.flip(state["board"], dims = (3,))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [units_state[1][i][0], 15 - units_state[1][i][1]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = units_state[0][0, i, -2], 2 - units_state[0][0, i, -1]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 1], distribution[player][unit_type][i, 3] = distribution[player][unit_type][i, 3], distribution[player][unit_type][i, 1]
    return state, distribution

def horizontal_flip(state, distribution):
    state, distribution = deepcopy(state), deepcopy(distribution)
    state["board"] = torch.flip(state["board"], dims = (2,))
    for player in ["player_0", "player_1"]:
        for unit_type, units_state in state[player].items():
            if len(units_state) > 0:
                for i in range(units_state[0].shape[1]):
                    units_state[1][i] = [15 - units_state[1][i][0], units_state[1][i][1]]
                    units_state[0][0, i, -2], units_state[0][0, i, -1] = 2 - units_state[0][0, i, -2], units_state[0][0, i, -1]
                    if unit_type == "units":
                        distribution[player][unit_type][i, 0], distribution[player][unit_type][i, 2] = distribution[player][unit_type][i, 2], distribution[player][unit_type][i, 0]
    return state, distribution
