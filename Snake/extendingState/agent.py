import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from collections import deque
from model import QTrainer, Linear_QNet
from helper import plot


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(
            maxlen=MAX_MEMORY
        )  # the deque will do popleft() when it is full

        self.model = Linear_QNet(14, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
    def simulate_move(self, head, direction, action):
        # Define the clockwise order of directions
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(direction)
        
        if np.array_equal(action, [1, 0, 0]):  # straight
            new_direction = direction
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_direction = clock_wise[(idx + 1) % 4]
        elif np.array_equal(action, [0, 0, 1]):  # left turn
            new_direction = clock_wise[(idx - 1) % 4]
        
        # Compute new head position based on new_direction (assuming BLOCK_SIZE step)
        x, y = head.x, head.y
        if new_direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_direction == Direction.UP:
            y -= BLOCK_SIZE
        
        return Point(x, y)

    # Then, update your get_state function to include additional "imminent death" features.
    def get_state(self, game):
        head = game.snake[0]
        # Existing points for immediate neighbors (used for general danger)
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # Existing danger features based on current direction and one block away
        danger_straight = (
            (dir_r and game.is_collision(point_r))
            or (dir_l and game.is_collision(point_l))
            or (dir_u and game.is_collision(point_u))
            or (dir_d and game.is_collision(point_d))
        )
        danger_right = (
            (dir_u and game.is_collision(point_r))
            or (dir_d and game.is_collision(point_l))
            or (dir_l and game.is_collision(point_u))
            or (dir_r and game.is_collision(point_d))
        )
        danger_left = (
            (dir_d and game.is_collision(point_r))
            or (dir_u and game.is_collision(point_l))
            or (dir_r and game.is_collision(point_u))
            or (dir_l and game.is_collision(point_d))
        )

        # Now, add lookahead features:
        # Simulate what would happen if the snake goes straight, right, or left
        future_straight = self.simulate_move(head, game.direction, [1, 0, 0])
        future_right = self.simulate_move(head, game.direction, [0, 1, 0])
        future_left = self.simulate_move(head, game.direction, [0, 0, 1])
        
        imminent_death_straight = game.is_collision(future_straight)
        imminent_death_right = game.is_collision(future_right)
        imminent_death_left = game.is_collision(future_left)

        # Optionally include the food distance information (as before, or as distances rather than booleans)
        food_left = game.food.x < head.x
        food_right = game.food.x > head.x
        food_up = game.food.y < head.y
        food_down = game.food.y > head.y

        # Build the state vector with the extra lookahead features (you now have 11+3=14 dimensions)
        state = np.array([
            danger_straight,
            danger_right,
            danger_left,
            imminent_death_straight,
            imminent_death_right,
            imminent_death_left,
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            food_left,
            food_right,
            food_up,
            food_down,
        ], dtype=int)

        return state

    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done)
        )  # popleft if Max_memory is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
    # Adjust exploration-exploitation using a minimum epsilon
        self.epsilon = max(0.01, 80 - self.n_games)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move



def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get the old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print("Game", agent.n_games, "Score", score, "Record:", record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)


if __name__ == "__main__":
    train()
