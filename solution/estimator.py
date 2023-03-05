import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from time import time
from copy import deepcopy

from utils import *

FACTORY_ACTIONS = [
    0, # Build light
    1, # Build heavy
    2, # Water
]

ROBOT_ACTIONS = [
    [np.array([0, 1, 0, 0, 0, 1])], # Move up
    [np.array([0, 2, 0, 0, 0, 1])], # Move right
    [np.array([0, 3, 0, 0, 0, 1])], # Move down
    [np.array([0, 4, 0, 0, 0, 1])], # Move left

    [np.array([3, 0, 0, 0, 0, 1])], # Dig
    [np.array([4, 0, 0, 0, 0, 1])], # Self destruct

    [np.array([2, 0, 0, -100, 0, 1])], # Pick up ice
    [np.array([2, 0, 1, -100, 0, 1])], # Pick up ore
    [np.array([2, 0, 2, -100, 0, 1])], # Pick up water
    [np.array([2, 0, 3, -100, 0, 1])], # Pick up metal
    [np.array([2, 0, 4, -100, 0, 1])], # Pick up power

    # Direction is not necessary
    [np.array([1, -100, 0, -100, 0, 1])], # Transfer
    [np.array([1, -100, 1, -100, 0, 1])], # Transfer
    [np.array([1, -100, 2, -100, 0, 1])], # Transfer
    [np.array([1, -100, 3, -100, 0, 1])], # Transfer
    [np.array([1, -100, 4, -100, 0, 1])], # Transfer
]

class Estimator(nn.Module):
    def __init__(self, n_iter):
        super().__init__()
        self.net = UltimateNet()
        self.n_iter = n_iter

    def predict(self, state, mode, player_number):
        if mode == "fast" or mode == "full":
            prepared_state = self._prepare_state(state)
            self.net.eval()
            with torch.no_grad():
                predictions, v = self.net(prepared_state)
            if mode == "full":
                raw_actions, env_actions, P, v = self._prepare_prediction(deepcopy(predictions), v, prepared_state, state, player_number)
                return raw_actions, env_actions, P, v, prepared_state, predictions, deepcopy(self.ids)
            return predictions, deepcopy(self.ids), v * (1 - 2 * player_number)
        elif mode == "train":
            self.net.train()
            predictions, v = self.net(state) 
            return predictions, v        

    def _prepare_state(self, state):
        self.ids = {"player_0": {"factories": [], "units": []}, "player_1": {"factories": [], "units": []}}

        factories_0, factories_1 = create_mask(state, type = "factories", subtype = "place")
        energy_0, energy_1 = create_mask(state, type = "factories", subtype = "energy")
        water_0, water_1 = create_mask(state, type = "factories", subtype = "water")
        metal_0, metal_1 = create_mask(state, type = "factories", subtype = "metal")
        ore_0, ore_1 = create_mask(state, type = "factories", subtype = "ore")
        ice_0, ice_1 = create_mask(state, type = "factories", subtype = "ice")
        lichen_0, lichen_1 = create_mask(state, type = "factories", subtype = "lichen")
        light_0, light_1 = create_mask(state, type = "units", subtype = "LIGHT")
        heavy_0, heavy_1 = create_mask(state, type = "units", subtype = "HEAVY")
        is_night, time = create_mask(state, type = "time")
        rubble = torch.tensor(state.board.rubble, dtype = torch.float).unsqueeze(0) / 100
        ore = torch.tensor(state.board.ore, dtype = torch.float).unsqueeze(0)
        ice = torch.tensor(state.board.ice, dtype = torch.float).unsqueeze(0)
        lichen = torch.tensor(state.board.lichen, dtype = torch.float).unsqueeze(0) / 100
        board = torch.cat([factories_0, factories_1, light_0, light_1, heavy_0, heavy_1, 
                           rubble, ore, ice, lichen, lichen_0, lichen_1, is_night, time]).unsqueeze(0) # 14 feature cards 
        
        useful = torch.cat([energy_0, energy_1, water_0, water_1, ore_0, ore_1, ice_0, ice_1, metal_0, metal_1])

        prepared_state = {"board": board, "useful": useful, "player_0": {}, "player_1": {}}
        for team in ["player_0", "player_1"]:
            units_per_team, units_pos = [], []
            for unit_id, unit in state.units[team].items():
                self.ids[team]["units"].append(unit_id)

                power_abs = unit.power / 1000
                ice_abs = unit.cargo.ice / 300
                ore_abs = unit.cargo.ore / 300
                water_abs = unit.cargo.water / 300
                metal_abs = unit.cargo.metal / 300

                is_heavy = 1 if unit.unit_type.name == "HEAVY" else 0
                is_light = 0 if unit.unit_type.name == "HEAVY" else 1

                power_relative = unit.power / 150 if is_light == 1 else unit.power / 3000
                ice_relative = unit.cargo.ice / 100 if is_light == 1 else unit.cargo.ice / 1000
                ore_relative = unit.cargo.ore / 100 if is_light == 1 else unit.cargo.ore / 1000
                water_relative = unit.cargo.water / 100 if is_light == 1 else unit.cargo.water / 1000
                metal_relative = unit.cargo.metal / 100 if is_light == 1 else unit.cargo.metal / 1000

                pos_x = unit.pos.x % 3
                pos_y = unit.pos.y % 3

                units_pos.append([unit.pos.x // 3, unit.pos.y // 3])
                units_per_team.append(torch.tensor([[power_abs, ice_abs, ore_abs, water_abs, metal_abs,
                                                     is_heavy, is_light, power_relative, ice_relative, 
                                                     ore_relative, water_relative, metal_relative, pos_x, pos_y]], dtype = torch.float)) # 14 features
            prepared_state[team]["units"] = [torch.cat(units_per_team).unsqueeze(0), units_pos] if len(units_per_team) > 0 else []

            factories_per_team, factories_pos = [], []
            for unit_id, unit in state.factories[team].items():
                self.ids[team]["factories"].append(unit_id)    

                power_abs = unit.power / 1000
                ice_abs = unit.cargo.ice / 300
                ore_abs = unit.cargo.ore / 300
                water_abs = unit.cargo.water / 300
                metal_abs = unit.cargo.metal / 300

                pos_x = unit.pos.x % 3
                pos_y = unit.pos.y % 3

                factories_pos.append([unit.pos.x // 3, unit.pos.y // 3])
                factories_per_team.append(torch.tensor([[power_abs, ice_abs, ore_abs, water_abs, metal_abs,
                                                     pos_x, pos_y]], dtype = torch.float)) # 7 features
            prepared_state[team]["factories"] = [torch.cat(factories_per_team).unsqueeze(0), factories_pos]

        return prepared_state

    def _prepare_prediction(self, predictions, v, prepared_state, state, player_number):
        v *= 1 - 2 * player_number # Compute v for actual player
        for player, prediction in predictions.items():
            for i, unit_id in enumerate(self.ids[player]["units"]):
                prediction["units"][0, i][invalid_actions_unit(player, unit_id, prepared_state, state)] = -torch.inf

            for i, unit_id in enumerate(self.ids[player]["factories"]):
                prediction["factories"][0, i][invalid_actions_factories(state.factories[player][unit_id], prepared_state, player_number)] = -torch.inf

        best = {"player_0": {}, "player_1": {}}
        for player, prediction in predictions.items():
            best[player]["factories"] = torch.argmax(prediction["factories"], dim = 2)
            if len(prediction["units"]) == 0:
                best[player]["units"] = None
                continue
            best[player]["units"] = torch.argmax(prediction["units"], dim = 2)

        P, env_actions, raw_actions = [], [], []
        for i in range(self.n_iter):
            prob, action, raw_action = 0, {"player_0": {}, "player_1": {}}, {"player_0": {}, "player_1": {}}
            for player, prediction in predictions.items():
                prob += np.sum([prediction["factories"][0, i, best[player]["factories"][0, i]].item() for i in range(prediction["factories"].shape[1])])
                raw_action[player]["factories"] = compute_raw_action(best[player]["factories"], 4)
                raw_action[player]["units"] = 0

                if len(prediction["units"]) > 0:
                    prob += np.sum([prediction["units"][0, i, best[player]["units"][0, i]].item() for i in range(prediction["units"].shape[1])])     
                    raw_action[player]["units"] = compute_raw_action(best[player]["units"], 17)

                for i, factory_id in enumerate(self.ids[player]["factories"]):   
                    if best[player]["factories"][0, i] == 3:
                        continue
                    action[player][factory_id] = best[player]["factories"][0, i].item()

                for i, unit_id in enumerate(self.ids[player]["units"]): 
                    pos_x = state.units[player][unit_id].pos.x
                    pos_y = state.units[player][unit_id].pos.y
                    map_n = 0 if player == "player_0" else 1

                    if best[player]["units"][0, i] == 16:
                        continue
                    elif best[player]["units"][0, i] < 6:
                        action[player][unit_id] = ROBOT_ACTIONS[best[player]["units"][0, i]]
                    elif best[player]["units"][0, i] > 5 and best[player]["units"][0, i] < 11:
                        if best[player]["units"][0, i] < 10:
                            cargo_space = 100 if state.units[player][unit_id].unit_type.name == "LIGHT" else 1000
                            if best[player]["units"][0, i] == 6:
                                amount = min(cargo_space - state.units[player][unit_id].cargo.ice, prepared_state["useful"][6 + map_n, pos_x, pos_y].item())
                            if best[player]["units"][0, i] == 7:
                                amount = min(cargo_space - state.units[player][unit_id].cargo.ore, prepared_state["useful"][4 + map_n, pos_x, pos_y].item())
                            if best[player]["units"][0, i] == 8:
                                amount = min(cargo_space - state.units[player][unit_id].cargo.water, prepared_state["useful"][2 + map_n, pos_x, pos_y].item())
                            if best[player]["units"][0, i] == 9:
                                amount = min(cargo_space - state.units[player][unit_id].cargo.metal, prepared_state["useful"][8 + map_n, pos_x, pos_y].item())
                        else:
                            cargo_space = 150 if state.units[player][unit_id].unit_type.name == "LIGHT" else 3000
                            amount = min(cargo_space - state.units[player][unit_id].power, prepared_state["useful"][map_n, pos_x, pos_y].item())
                        action[player][unit_id] = ROBOT_ACTIONS[best[player]["units"][0, i]]
                        action[player][unit_id][0][3] = int(amount)
                    elif best[player]["units"][0, i] > 10:
                        if best[player]["units"][0, i] == 15:
                            amount = state.units[player][unit_id].power
                        if best[player]["units"][0, i] == 14:
                            amount = state.units[player][unit_id].cargo.metal
                        if best[player]["units"][0, i] == 13:
                            amount = state.units[player][unit_id].cargo.water
                        if best[player]["units"][0, i] == 12:
                            amount = state.units[player][unit_id].cargo.ore
                        if best[player]["units"][0, i] == 11:
                            amount = state.units[player][unit_id].cargo.ice
                        direction = compute_direction(prepared_state["board"], map_n, pos_x, pos_y)
                        action[player][unit_id] = ROBOT_ACTIONS[best[player]["units"][0, i]]
                        action[player][unit_id][0][3] = int(amount)
                        action[player][unit_id][0][1] = direction
            P.append(prob.item())
            env_actions.append(action)
            raw_actions.append(raw_action)
            best = compute_best(best, predictions)
            if best is None: 
                break

        return raw_actions, env_actions, list(np.exp(P) / sum(np.exp(P))), v.item()



# Compute v for player_0 and probabilities of every unit action for each player
class UltimateNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = FourSequenceMHAttention(31, 38, 32, 2)
        self.extractor = ResMHAttention(32, 32, 32, 2, 2)
        self.v_estimator = nn.Sequential(
            nn.Conv2d(66, 96, 3, padding=1), nn.PReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(96, 128, 3, padding=1), nn.PReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(), nn.Linear(128 * 16, 1), nn.Tanh()
        )

        self.factory_output = nn.Sequential(nn.Linear(32, 32), nn.PReLU(), nn.Linear(32, 4))
        self.robot_output = nn.Sequential(nn.Linear(32, 32), nn.PReLU(), nn.Linear(32, 17))
        

        self.spatial_extractor = nn.Sequential(
            nn.Conv2d(14, 48, 3, padding=1), nn.PReLU(),
            nn.Conv2d(48, 96, 3, padding=1), nn.PReLU(), 
            nn.Conv2d(96, 24, 3, stride=3)
        )

    def forward(self, state):
        spatial_map = self.spatial_extractor(state["board"])
        v_map = spatial_map.clone()
        modified_state = {"player_0": {"factories": [], "units": []}, "player_1": {"factories": [], "units": []}}
        for team in ["player_0", "player_1"]:
            global_mask_units, global_mask_factories = torch.zeros(v_map.shape[0], 16, 16, 14), torch.zeros(v_map.shape[0], 16, 16, 7)
            for unit_type in state[team]:
                if len(state[team][unit_type]) == 0:
                    modified_state[team][unit_type] = {}
                    continue
                for i in range(state[team][unit_type][0].shape[1]):
                    pos_x, pos_y = state[team][unit_type][1][i][0], state[team][unit_type][1][i][1]
                    modified_state[team][unit_type].append(torch.cat([state[team][unit_type][0][:, i].clone(), spatial_map[:, :, pos_x, pos_y].clone()], 1).unsqueeze(1))
                    if unit_type == "factories":
                        mask = torch.zeros(v_map.shape[0], 16, 16, 7)
                        mask[0, pos_x, pos_y] = 1
                        mask *= state[team][unit_type][0][:, i].clone()
                        global_mask_factories += mask
                    if unit_type == "units":
                        mask = torch.zeros(v_map.shape[0], 16, 16, 14)
                        mask[0, pos_x, pos_y] = 1
                        mask *= state[team][unit_type][0][:, i].clone()
                        global_mask_units += mask
                modified_state[team][unit_type] = torch.cat(modified_state[team][unit_type], 1)
            v_map = torch.cat([global_mask_units.transpose(2, 3).transpose(1, 2), global_mask_factories.transpose(2, 3).transpose(1, 2), v_map], dim = 1)

        encoded_sequences, lengths = self.enc(modified_state)
        extracted_sequences = self.extractor(encoded_sequences, lengths)
        for team in extracted_sequences:
            extracted_sequences[team]["factories"] = self.factory_output(extracted_sequences[team]["factories"])
            if lengths[team]["units"] == 0:
                continue
            extracted_sequences[team]["units"] = self.robot_output(extracted_sequences[team]["units"])

        v = self.v_estimator(v_map)
        return extracted_sequences, v

class FourSequenceMHAttention(nn.Module):
    def __init__(self, factories_input_dim, units_input_dim, hidden_dim, num_heads):
        super().__init__()
        self.factories = nn.ModuleList([
            nn.Linear(factories_input_dim, hidden_dim) for _ in range(6 * num_heads)
        ])    

        self.units = nn.ModuleList([
            nn.Linear(units_input_dim, hidden_dim) for _ in range(6 * num_heads)
        ])    

        self.linear = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.num_heads = num_heads
        self.scale = hidden_dim ** 0.5

    def forward(self, state):
        result = []
        for head in range(self.num_heads):
            Q, K, V = [], [], []
            lengths = {"player_0": {}, "player_1": {}}
            for i, player in enumerate(["player_0", "player_1"]):
                Q.append(self.factories[head * 6 + i * 3](state[player]["factories"]))
                K.append(self.factories[head * 6 + i * 3 + 1](state[player]["factories"]))
                V.append(self.factories[head * 6 + i * 3 + 2](state[player]["factories"]))
                lengths[player]["factories"] = state[player]["factories"].shape[1]

                if len(state[player]["units"]) > 0:
                    Q.append(self.units[head * 6 + i * 3](state[player]["units"]))
                    K.append(self.units[head * 6 + i * 3 + 1](state[player]["units"]))
                    V.append(self.units[head * 6 + i * 3 + 2](state[player]["units"]))
                    lengths[player]["units"] = state[player]["units"].shape[1]

                else:
                    lengths[player]["units"] = 0
            
            Q = torch.cat(Q, 1)
            K = torch.cat(K, 1)
            V = torch.cat(V, 1)
            result.append(F.softmax(Q @ K.transpose(1, 2) / self.scale, dim = 2) @ V)
        result = torch.cat(result, 2)
        return self.linear(result), lengths
       
class ResMHAttention(nn.Module):
    def __init__(self, factories_input_dim, units_input_dim, hidden_dim, num_heads, depth):
        super().__init__()
        self.attns = nn.ModuleList([FourSequenceMHAttention(factories_input_dim, units_input_dim, hidden_dim, num_heads) for _ in range(depth)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(depth)])
        
    def forward(self, seq, lengths):
        for step in range(len(self.attns)):
            i = 0
            state = {"player_0": {}, "player_1": {}}
            for player in ["player_0", "player_1"]:
                state[player]["factories"] = seq[:, i : i + lengths[player]["factories"]].clone()
                i += lengths[player]["factories"]
                if lengths[player]["units"] == 0:
                    state[player]["units"] = {}
                    continue
                state[player]["units"] = seq[:, i : i + lengths[player]["units"]].clone()
                i += lengths[player]["units"]

            seq_new, lengths = self.attns[step](state)
            seq += seq_new
            seq = self.norm[step](seq)

        i = 0
        state = {"player_0": {}, "player_1": {}}
        for player in ["player_0", "player_1"]:
            state[player]["factories"] = seq[:, i : i + lengths[player]["factories"]].clone()
            i += lengths[player]["factories"]
            if lengths[player]["units"] == 0:
                    state[player]["units"] = {}
                    continue
            state[player]["units"] = seq[:, i : i + lengths[player]["units"]].clone()
            i += lengths[player]["units"]

        return state
