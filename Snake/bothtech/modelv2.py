import os
from typing import Union, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Directory and filename for saving models\n
MODEL_DIR = "./model"
DEFAULT_FILENAME = "model.pth"


class Linear_QNet(nn.Module):
    """
    Simple feed-forward neural network for Deep Q-Learning.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with ReLU activation on the hidden layer.
        """
        x = F.relu(self.linear1(x))
        return self.linear2(x)

    def save(self, filename: str = DEFAULT_FILENAME) -> None:
        """
        Save the model's state dictionary to disk.
        """
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, filename)
        torch.save(self.state_dict(), path)

    def load(
        self, filepath: str, map_location: Union[str, torch.device] = None
    ) -> None:
        """
        Load a state dictionary from disk into the model.
        """
        state = torch.load(filepath, map_location=map_location)
        self.load_state_dict(state)


class QTrainer:
    """
    Trainer for the Q-network, performing single-step Q-learning updates.
    """

    def __init__(self, model: Linear_QNet, lr: float, gamma: float) -> None:
        self.model = model
        self.gamma = gamma
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(
        self,
        states: Union[Sequence, torch.Tensor],
        actions: Union[Sequence, torch.Tensor],
        rewards: Union[Sequence, torch.Tensor],
        next_states: Union[Sequence, torch.Tensor],
        dones: Union[Sequence, bool],
    ) -> None:
        """
        Execute a Q-learning update on a batch of transitions.
        """
        # Convert inputs to tensors
        state_batch = torch.tensor(states, dtype=torch.float32)
        next_batch = torch.tensor(next_states, dtype=torch.float32)
        action_batch = torch.tensor(actions, dtype=torch.long)
        reward_batch = torch.tensor(rewards, dtype=torch.float32)

        # Ensure batch dimension exists
        if state_batch.dim() == 1:
            state_batch = state_batch.unsqueeze(0)
            next_batch = next_batch.unsqueeze(0)
            action_batch = action_batch.unsqueeze(0)
            reward_batch = reward_batch.unsqueeze(0)
            dones = (dones,)

        # Predicted Q-values for current states
        pred_q = self.model(state_batch)
        target_q = pred_q.clone().detach()

        # Compute target Q-values
        for idx, done in enumerate(dones):
            q_new = reward_batch[idx]
            if not done:
                q_new = reward_batch[idx] + self.gamma * torch.max(
                    self.model(next_batch[idx])
                )
            target_q[idx, action_batch[idx].argmax().item()] = q_new

        # Backpropagate loss
        self.optimizer.zero_grad()
        loss = self.criterion(target_q, pred_q)
        loss.backward()
        self.optimizer.step()
