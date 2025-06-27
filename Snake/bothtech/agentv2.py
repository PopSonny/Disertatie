import random
from collections import deque

import numpy as np
import torch

from gamev2 import SnakeGameAI, Direction, Point, BLOCK_SIZE
from helper import plot
from modelv2 import Linear_QNet, QTrainer

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1_000
LEARNING_RATE = 0.001
GAMMA = 0.9

class Agent:
    """
    Reinforcement learning agent using Deep Q-Learning for Snake.
    """
    def __init__(self):
        self.game_count = 0               # number of completed games
        self.exploration_rate = 0.0       # epsilon for exploration
        self.replay_buffer = deque(maxlen=MAX_MEMORY)

        # Initialize model and trainer
        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=GAMMA)

    def add_experience(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def replay_train(self):
        """Train on a batch of past experiences."""
        if len(self.replay_buffer) >= BATCH_SIZE:
            batch = random.sample(self.replay_buffer, BATCH_SIZE)
        else:
            batch = list(self.replay_buffer)

        states, actions, rewards, next_states, dones = zip(*batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def online_train(self, state, action, reward, next_state, done):
        """Immediate training on the latest step."""
        self.trainer.train_step(state, action, reward, next_state, done)

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        # Decaying exploration
        self.exploration_rate = max(0.01, 80 - self.game_count)
        if random.random() < self.exploration_rate / 200:
            action = [0, 0, 0]
            action[random.randint(0, 2)] = 1
            return action

        state_tensor = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state_tensor)
        move_index = torch.argmax(prediction).item()
        action = [0, 0, 0]
        action[move_index] = 1
        return action

    def _simulate_move(self, head, direction, move):
        """Compute the next head position without altering game state."""
        directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = directions.index(direction)

        if     np.array_equal(move, [0, 1, 0]): new_dir = directions[(idx + 1) % 4]
        elif   np.array_equal(move, [0, 0, 1]): new_dir = directions[(idx - 1) % 4]
        else: new_dir = direction

        x, y = head.x, head.y
        if new_dir == Direction.RIGHT: x += BLOCK_SIZE
        if new_dir == Direction.LEFT:  x -= BLOCK_SIZE
        if new_dir == Direction.DOWN:  y += BLOCK_SIZE
        if new_dir == Direction.UP:    y -= BLOCK_SIZE

        return Point(x, y)

    def get_state(self, game: SnakeGameAI):
        """Extract a 14-dimensional state representation from the game."""
        head = game.snake[0]
        neighbors = {
            'left':  Point(head.x - BLOCK_SIZE, head.y),
            'right': Point(head.x + BLOCK_SIZE, head.y),
            'up':    Point(head.x, head.y - BLOCK_SIZE),
            'down':  Point(head.x, head.y + BLOCK_SIZE)
        }
        dirs = {
            'left':  game.direction == Direction.LEFT,
            'right': game.direction == Direction.RIGHT,
            'up':    game.direction == Direction.UP,
            'down':  game.direction == Direction.DOWN
        }

        # Danger in current and lookahead positions
        danger = {
            'straight': dirs['right'] and game.is_collision(neighbors['right'])
                    or dirs['left']  and game.is_collision(neighbors['left'])
                    or dirs['up']    and game.is_collision(neighbors['up'])
                    or dirs['down']  and game.is_collision(neighbors['down']),
            'right':    dirs['up']   and game.is_collision(neighbors['right'])
                    or dirs['down'] and game.is_collision(neighbors['left'])
                    or dirs['left'] and game.is_collision(neighbors['up'])
                    or dirs['right']and game.is_collision(neighbors['down']),
            'left':     dirs['down'] and game.is_collision(neighbors['right'])
                    or dirs['up']   and game.is_collision(neighbors['left'])
                    or dirs['right']and game.is_collision(neighbors['up'])
                    or dirs['left'] and game.is_collision(neighbors['down'])
        }

        # Imminent death signals
        imminent = {
            key: game.is_collision(self._simulate_move(head, game.direction, move))
            for key, move in zip(['straight','right','left'],
                                 [[1,0,0],[0,1,0],[0,0,1]])
        }

        # Food relative position
        food = game.food
        food_dir = {
            'left':  food.x < head.x,
            'right': food.x > head.x,
            'up':    food.y < head.y,
            'down':  food.y > head.y
        }

        # Build state vector
        state = [
            danger['straight'], danger['right'], danger['left'],
            imminent['straight'], imminent['right'], imminent['left'],
            dirs['left'], dirs['right'], dirs['up'], dirs['down'],
            food_dir['left'], food_dir['right'], food_dir['up'], food_dir['down']
        ]
        return np.array(state, dtype=int)


def train():
    """Main training loop."""
    scores, mean_scores, total_score, best_score = [], [], 0, 0
    agent = Agent()
    game  = SnakeGameAI()

    while True:
        s_old = agent.get_state(game)
        action = agent.select_action(s_old)
        reward, done, score = game.play_step(action)
        s_new = agent.get_state(game)

        agent.online_train(s_old, action, reward, s_new, done)
        agent.add_experience(s_old, action, reward, s_new, done)

        if done:
            game.reset()
            agent.game_count += 1
            agent.replay_train()

            if score > best_score:
                best_score = score
                agent.model.save()

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / agent.game_count)
            print(f"Game {agent.game_count} Score {score} Best {best_score}")
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()
