import numpy as np
import torch
from math import floor, sin
from copy import deepcopy

from scipy.ndimage import distance_transform_cdt
from scipy.spatial import KDTree

from kits.python.lux.kit import obs_to_game_state

def update(node, v):
    node.N += 1
    node.V += v
    node.Q, node.u = node.V / node.N, node.T * node.p / (1 + node.N)
    if node.parent is not None:
        update(node.parent, v)

def isequal(dict_1, dict_2):
    if not list(dict_1.keys()) == list(dict_2.keys()): return False
    for key in dict_1.keys():
        if not type(dict_1[key]) == type(dict_2[key]): return False
        
        if isinstance(dict_1[key], dict): return isequal(dict_1[key], dict_2[key])
        elif isinstance(dict_1[key], np.ndarray): 
            if not dict_1[key].shape == dict_2[key].shape: return False
            return np.all(dict_1[key] == dict_2[key])

        else: return dict_1[key] == dict_2[key]

def mean_across_dicts(dicts, N):
    N += 1e-9
    factories_0, units_0 = 0, 0
    factories_1, units_1 = 0, 0
    for dict, n in dicts:
        factories_0 += dict["player_0"]["factories"] * n
        units_0 += dict["player_0"]["units"] * n
        factories_1 += dict["player_1"]["factories"] * n
        units_1 += dict["player_1"]["units"] * n
    return {"player_0": {"factories": factories_0 / N, "units": units_0 / N}, "player_1": {"factories": factories_1 / N, "units": units_1 / N}}

def state_from_obs(state_, obs, env_cfg, step):
    state = deepcopy(state_)
    return state.from_obs(obs, env_cfg)

def create_mask(state, type, subtype = None):
    mask = {"player_0": torch.zeros(1, 48, 48), "player_1": torch.zeros(1, 48, 48)}
    if type == "factories":
        for player in ["player_0", "player_1"]:
            for unit_id, unit in state.factories[player].items():
                if subtype == "place":
                    mask[player][0, unit.pos.x - 1 : unit.pos.x + 2, unit.pos.y - 1 : unit.pos.y + 2] = torch.ones(3, 3)
                if subtype == "energy":
                    mask[player][0, unit.pos.x - 1 : unit.pos.x + 2, unit.pos.y - 1 : unit.pos.y + 2] = torch.ones(3, 3) * unit.power
                if subtype == "ore":
                    mask[player][0, unit.pos.x - 1 : unit.pos.x + 2, unit.pos.y - 1 : unit.pos.y + 2] = torch.ones(3, 3) * unit.cargo.ore
                if subtype == "ice":
                    mask[player][0, unit.pos.x - 1 : unit.pos.x + 2, unit.pos.y - 1 : unit.pos.y + 2] = torch.ones(3, 3) * unit.cargo.ice
                if subtype == "metal":
                    mask[player][0, unit.pos.x - 1 : unit.pos.x + 2, unit.pos.y - 1 : unit.pos.y + 2] = torch.ones(3, 3) * unit.cargo.metal
                if subtype == "water":
                    mask[player][0, unit.pos.x - 1 : unit.pos.x + 2, unit.pos.y - 1 : unit.pos.y + 2] = torch.ones(3, 3) * unit.cargo.water

    if type == "units":
        for player in ["player_0", "player_1"]:
            for unit_id, unit in state.units[player].items():
                if subtype == "HEAVY" and unit.unit_type.name == "HEAVY":
                    mask[player][0, unit.pos.x, unit.pos.y] = 1
                    continue
                if subtype == "LIGHT" and unit.unit_type.name == "LIGHT":
                    mask[player][0, unit.pos.x, unit.pos.y] = 1
                    continue

    if type == "time":
        if state.real_env_steps % 50 > 29:
            mask["player_0"] += 1
        mask["player_1"] += sin((state.real_env_steps % 50) * 6.28 / 50.)


    return mask["player_0"], mask["player_1"]

def invalid_actions_factories(factory, prepared_state, player_number):
    inv_act = []
    if factory.cargo.water < 5:
        inv_act.append(2)

    collision = (prepared_state["board"][0, player_number + 2, factory.pos.x, factory.pos.y] == 1 or prepared_state["board"][0, player_number + 4, factory.pos.x, factory.pos.y] == 1).item()
    # if len(prepared_state["player_0"]["units"]) > 0 or len(prepared_state["player_1"]["units"]) > 0:
    #     breakpoint()
    if factory.cargo.metal < 100 or factory.power < 500 or collision:
        inv_act.append(1)
    if factory.cargo.metal < 10 or factory.power < 50 or collision:
        inv_act.append(0)
    return inv_act

def invalid_actions_unit(player, unit_id, prepared_state, state):
    inv_act = []

    pos_x = state.units[player][unit_id].pos.x
    pos_y = state.units[player][unit_id].pos.y
    map_n = 0 if player == "player_0" else 1
    if pos_y == 47 or prepared_state["board"][0, 1 - map_n, pos_x, min(pos_y + 1, 47)] == 1 : inv_act.append(2)
    if pos_x == 47 or prepared_state["board"][0, 1 - map_n, min(pos_x + 1, 47), pos_y] == 1 : inv_act.append(1)
    if pos_y == 0 or prepared_state["board"][0, 1 - map_n, pos_x, max(pos_y - 1, 0)] == 1 : inv_act.append(0)
    if pos_x == 0 or prepared_state["board"][0, 1 - map_n, max(pos_x - 1, 0), pos_y] == 1 : inv_act.append(3)

    if prepared_state["board"][0, map_n, pos_x, pos_y] == 0: inv_act += [6, 7, 8, 9, 10]
    else:
        if prepared_state["useful"][6 + map_n, pos_x, pos_y] == 0: inv_act.append(6)
        if prepared_state["useful"][4 + map_n, pos_x, pos_y] == 0: inv_act.append(7)
        if prepared_state["useful"][2 + map_n, pos_x, pos_y] == 0: inv_act.append(8)
        if prepared_state["useful"][8 + map_n, pos_x, pos_y] == 0: inv_act.append(9)
        if prepared_state["useful"][0 + map_n, pos_x, pos_y] == 0: inv_act.append(10)

    if np.all([prepared_state["board"][0, map_n, pos_x, pos_y] == 0,
               prepared_state["board"][0, map_n, min(pos_x + 1, 47), pos_y] == 0, 
               prepared_state["board"][0, map_n, max(pos_x - 1, 0), pos_y] == 0, 
               prepared_state["board"][0, map_n, pos_x, max(pos_y - 1, 0)] == 0, 
               prepared_state["board"][0, map_n, pos_x, min(pos_y + 1, 47)] == 0]):
        
        count = np.sum([prepared_state["board"][0, map_n + 2, pos_x, pos_y] == 0,
                        prepared_state["board"][0, map_n + 2, min(pos_x + 1, 47), pos_y] == 0, 
                        prepared_state["board"][0, map_n + 2, max(pos_x - 1, 0), pos_y] == 0, 
                        prepared_state["board"][0, map_n + 2, pos_x, max(pos_y - 1, 0)] == 0, 
                        prepared_state["board"][0, map_n + 2, pos_x, min(pos_y + 1, 47)] == 0, 
                        prepared_state["board"][0, map_n + 4, pos_x, pos_y] == 0,
                        prepared_state["board"][0, map_n + 4, min(pos_x + 1, 47), pos_y] == 0, 
                        prepared_state["board"][0, map_n + 4, max(pos_x - 1, 0), pos_y] == 0, 
                        prepared_state["board"][0, map_n + 4, pos_x, max(pos_y - 1, 0)] == 0, 
                        prepared_state["board"][0, map_n + 4, pos_x, min(pos_y + 1, 47)] == 0,])
        if not count == 1:
            inv_act += [11, 12, 13, 14, 15]

    if state.units[player][unit_id].cargo.ice == 0: inv_act.append(11)
    if state.units[player][unit_id].cargo.ore == 0: inv_act.append(12)
    if state.units[player][unit_id].cargo.water == 0: inv_act.append(13)
    if state.units[player][unit_id].cargo.metal == 0: inv_act.append(14)
    if state.units[player][unit_id].power == 0: inv_act.append(15)

    destruct_price = 10 if state.units[player][unit_id].unit_type.name == "LIGHT" else 100
    if state.units[player][unit_id].power < destruct_price: inv_act.append(5)

    dig_price = 5 if state.units[player][unit_id].unit_type.name == "LIGHT" else 60
    if state.units[player][unit_id].power < dig_price or prepared_state["board"][0, map_n, pos_x, pos_y ] == 1: inv_act.append(4)

    down_price = floor(1 + 0.05 * state.board.rubble[pos_x][min(pos_y + 1, 47)]) if state.units[player][unit_id].unit_type.name == "LIGHT" else floor(5 + state.board.rubble[pos_x][min(pos_y + 1, 47)])
    if state.units[player][unit_id].power < down_price: inv_act.append(2)

    up_price = floor(1 + 0.05 * state.board.rubble[pos_x][max(pos_y - 1, 0)]) if state.units[player][unit_id].unit_type.name == "LIGHT" else floor(5 + state.board.rubble[pos_x][max(pos_y - 1, 0)])
    if state.units[player][unit_id].power < up_price: inv_act.append(0)

    right_price = floor(1 + 0.05 * state.board.rubble[min(pos_x + 1, 47)][pos_y]) if state.units[player][unit_id].unit_type.name == "LIGHT" else floor(5 + state.board.rubble[min(pos_x + 1, 47)][pos_y])
    if state.units[player][unit_id].power < right_price: inv_act.append(1)

    left_price = floor(1 + 0.05 * state.board.rubble[max(pos_x - 1, 0)][pos_y]) if state.units[player][unit_id].unit_type.name == "LIGHT" else floor(5 + state.board.rubble[max(pos_x - 1, 0)][pos_y])
    if state.units[player][unit_id].power < left_price: inv_act.append(3)

    return inv_act

def compute_direction(prepared_state_board, map_n, pos_x, pos_y):
    if prepared_state_board[0, map_n, pos_x, pos_y] == 1: return 0
    if prepared_state_board[0, map_n, min(pos_x + 1, 47), pos_y] == 1: return 2 
    if prepared_state_board[0, map_n, max(pos_x - 1, 0), pos_y] == 1: return 4 
    if prepared_state_board[0, map_n, pos_x, max(pos_y - 1, 0)] == 1: return 1 
    if prepared_state_board[0, map_n, pos_x, min(pos_y + 1, 47)] == 1: return 3

def compute_raw_action(best_ids, dim_action_space):
    raw_action = []
    for i in best_ids:
        action = torch.zeros(1, dim_action_space)
        action[0, i] = 1
        raw_action.append(action)

    return torch.cat(raw_action)

def compute_best(best, predictions):
    min_dif = {"player_0": {"units": 0, "factories": 0}, "player_1": {"units": 0, "factories": 0}}
    for player, prediction in predictions.items():
        if best[player]["units"] is not None:
            best_value = [prediction["units"].squeeze(0)[i, best[player]["units"][0, i]] for i in range(best[player]["units"].shape[1])]
            differences = (prediction["units"].squeeze(0).transpose(0, 1) - torch.tensor(best_value, dtype = torch.float)).transpose(0, 1)
            differences[differences >= 0] = -torch.inf
            min_dif[player]["units"] = differences
            # new_best_idx = torch.argmax(differences).item()
            # best[player]["units"][0, new_best_idx // prediction["units"].shape[-1]] = new_best_idx % prediction["units"].shape[-1]

        best_value = [prediction["factories"].squeeze(0)[i, best[player]["factories"][0, i]] for i in range(best[player]["factories"].shape[1])]
        differences = (prediction["factories"].squeeze(0).transpose(0, 1) - torch.tensor(best_value, dtype = torch.float)).transpose(0, 1)
        differences[differences >= 0] = -torch.inf
        min_dif[player]["factories"] = differences
        # new_best_idx = torch.argmax(differences).item()
        # best[player]["factories"][0, new_best_idx // prediction["factories"].shape[-1]] = new_best_idx % prediction["factories"].shape[-1]
    best_diff = -torch.inf
    best_idx = None
    for player, prediction in predictions.items():
        for unit_type in ["units", "factories"]:
            if isinstance(min_dif[player][unit_type], int):
                continue
            if min_dif[player][unit_type].max() > best_diff:
                best_diff = min_dif[player][unit_type].max()
                best_player = player
                best_type = unit_type
                best_idx = torch.argmax(min_dif[player][unit_type]).item()
    if best_idx is None:
        return None 
    best[best_player][best_type][0, best_idx // predictions[best_player][best_type].shape[-1]] = best_idx % predictions[best_player][best_type].shape[-1]
    return best

def compute_start_mask(obs):
    ice = obs["board"]["ice"]
    ore = obs["board"]["ore"]
    rubble = obs["board"]["rubble"]

    def manhattan_dist_to_nth_closest(arr, n):
        if n == 1:
            distance_map = distance_transform_cdt(1-arr, metric='taxicab')
            return distance_map
        else:
            true_coords = np.transpose(np.nonzero(arr)) # get the coordinates of true values
            tree = KDTree(true_coords) # build a KDTree
            dist, _ = tree.query(np.transpose(np.nonzero(~arr)), k=n, p=1) # query the nearest to nth closest distances using p=1 for Manhattan distance
            return np.reshape(dist[:, n-1], arr.shape) # reshape the result to match the input shape and add an extra dimension for the different closest distances

    # this is the distance to the n-th closest ice, for each coordinate
    ice_distances = [manhattan_dist_to_nth_closest(ice, i) for i in range(1,5)]

    # this is the distance to the n-th closest ore, for each coordinate
    ore_distances = [manhattan_dist_to_nth_closest(ore, i) for i in range(1,5)]

    ICE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25]) 
    weigthed_ice_dist = np.sum(np.array(ice_distances) * ICE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

    ORE_WEIGHTS = np.array([1, 0.5, 0.33, 0.25])
    weigthed_ore_dist = np.sum(np.array(ore_distances) * ORE_WEIGHTS[:, np.newaxis, np.newaxis], axis=0)

    ICE_PREFERENCE = 1

    combined_resource_score = (weigthed_ice_dist * ICE_PREFERENCE + weigthed_ore_dist)
    combined_resource_score = (np.max(combined_resource_score) - combined_resource_score) * obs["board"]["valid_spawns_mask"]

    low_rubble = (rubble<25)

    def count_region_cells(array, start, min_dist=2, max_dist=np.inf, exponent=1):
        
        def dfs(array, loc):
            distance_from_start = abs(loc[0]-start[0]) + abs(loc[1]-start[1])
            if not (0<=loc[0]<array.shape[0] and 0<=loc[1]<array.shape[1]):   # check to see if we're still inside the map
                return 0
            if (not array[loc]) or visited[loc]:     # we're only interested in low rubble, not visited yet cells
                return 0
            if not (min_dist <= distance_from_start <= max_dist):      
                return 0
            
            visited[loc] = True

            count = 1.0 * exponent**distance_from_start
            count += dfs(array, (loc[0]-1, loc[1]))
            count += dfs(array, (loc[0]+1, loc[1]))
            count += dfs(array, (loc[0], loc[1]-1))
            count += dfs(array, (loc[0], loc[1]+1))

            return count

        visited = np.zeros_like(array, dtype=bool)
        return dfs(array, start)

    low_rubble_scores = np.zeros_like(low_rubble, dtype=float)

    for i in range(low_rubble.shape[0]):
        for j in range(low_rubble.shape[1]):
            low_rubble_scores[i,j] = count_region_cells(low_rubble, (i,j), min_dist=0, max_dist=8, exponent=0.9)

    overall_score = low_rubble_scores*2 + combined_resource_score
    return overall_score

def to_json(obj):
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [to_json(s) for s in obj]
    elif isinstance(obj, dict):
        out = {}
        for k in obj:
            out[k] = to_json(obj[k])
        return out
    else:
        return obj