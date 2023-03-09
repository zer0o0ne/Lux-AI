import torch
from dataset import DatasetBuilder
from tqdm.auto import tqdm

def value_weight_policy_loss(predictions, v, vs, policies):
    v_loss, policy_loss = 0, 0
    v_loss += (v - vs["player_0"]) ** 2 + (v + vs["player_1"]) ** 2
    for player, prediction in predictions.items():
        for unit_type, policy in prediction.items():
            real_policy = policies[player][unit_type]
            if not isinstance(policy, dict):
                policy_loss += vs[player] * (real_policy * torch.log(policy + 1e-9)).sum()
    return v_loss - policy_loss

def value_policy_loss(predictions, v, vs, policies):
    v_loss, policy_loss = 0, 0
    v_loss += (v - vs["player_0"]) ** 2 + (v + vs["player_1"]) ** 2
    for player, prediction in predictions.items():
        for unit_type, policy in prediction.items():
            real_policy = policies[player][unit_type]
            if not isinstance(policy, dict):
                policy_loss += (real_policy * policy).sum()
    return v_loss - policy_loss


class Trainer:
    def __init__(self, weights_path, model, player, dataset = None, weight_decay = 1e-5, n_games = 200):
        if dataset is None:
            dataset = DatasetBuilder(player, n_games=n_games)
        self.dataset = dataset
        self.weights_path = weights_path
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(), weight_decay=weight_decay)

    def train(self, batch_size, cycle = 1, epochs = 1):
        try:
            self.model.load_state_dict(torch.load(self.weights_path))
        except:
            pass
        
        for epoch in range(epochs):
            batch, loss = 0, 0
            for state, vs, policies in tqdm(self.dataset, desc = "objects in dataset"):
                if batch == batch_size:
                    batch = 0
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    loss = 0

                predictions, v = self.model.predict(state, mode = "train", player_number = None)
                loss += value_weight_policy_loss(predictions, v, vs, policies)
                batch += 1

        number = self.weights_path.split("_")[-1]
        path = f'solution/weights/{cycle}_weights_' + number
        torch.save(self.model.state_dict(), path)
