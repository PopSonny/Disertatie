# set up the imports
import vizdoom as vzd
import random
import time
import numpy as np
from gymnasium import Env  # Change from gym to gymnasium
from gymnasium.spaces import Box, Discrete  # Import spaces from gymnasium
import cv2
from matplotlib import pyplot as plt

# import callback
import os
from stable_baselines3.common.callbacks import BaseCallback

# import ppo for training
from stable_baselines3 import PPO

# import evel policy
from stable_baselines3.common.evaluation import evaluate_policy

# import environment checker
from stable_baselines3.common import env_checker

CHECKPOING_DIR = "./train/train_corridor"
LOG_DIR = "./logs/log_corridor"


class VizDoomGym(Env):

    def __init__(
        self,
        render=False,
        config="github/ViZDoom-master/scenarios/deadly_corridor_s1.cfg",
    ):

        super().__init__()

        self.game = vzd.DoomGame()

        self.game.load_config(config)

        if render == False:

            self.game.set_window_visible(False)

        else:

            self.game.set_window_visible(True)

        self.game.init()

        self.observation_space = Box(
            low=0, high=255, shape=(100, 160, 1), dtype=np.uint8
        )

        self.action_space = Discrete(7)

        # HEALTH DAMAGE_TAKEN HITCOUNT SELECTED_WEAPON_AMMO

        self.damage_taken = 0
        self.hitcount = 0
        self.ammo = 52

    def step(self, action):

        actions = np.identity(7, dtype=np.uint8)

        movement_reward = self.game.make_action(actions[action], 4)

        reward = 0
        if self.game.get_state():

            state = self.game.get_state().screen_buffer

            state = self.grayscale(state)

            # reward shaping
            game_variables = self.game.get_state().game_variables
            health, damage_taken, hitcount, ammo = game_variables

            # calculate reward deltas
            damage_taken_delta = -damage_taken + self.damage_taken
            self.damage_taken = damage_taken
            hitcount_delta = hitcount - self.hitcount
            self.hitcount = hitcount
            ammo_delta = ammo - self.ammo
            self.ammo = ammo

            # the numbers can be tweaked for different results/different scenarios
            reward = (
                movement_reward
                + damage_taken_delta * 10
                + hitcount_delta * 600
                + ammo_delta * 5
            )

            info = ammo

        else:

            state = np.zeros(self.observation_space.shape)

            info = 0

        info = {"info": info}
        terminated = self.game.is_episode_finished()
        truncated = False  # Set this to True if using a max step limit

        return state, reward, terminated, truncated, info

    def render(self):
        pass

    def reset(self, *, seed=None, options=None):
        self.game.new_episode()

        state = self.game.get_state().screen_buffer

        return self.grayscale(state), {}

    def grayscale(self, observation):

        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)

        resize = cv2.resize(gray, (160, 100), interpolation=cv2.INTER_CUBIC)

        state = np.reshape(resize, (100, 160, 1))

        return state

    def close(self):
        self.game.close()


class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(
                self.save_path, "best_model_{}".format(self.n_calls)
            )
            self.model.save(model_path)

        return True


# train model using curriculum
def trainModel():
    env = VizDoomGym(config="github/ViZDoom-master/scenarios/deadly_corridor_s3.cfg")

    CHECKPOING_DIR = "./train/train_corridors3"
    LOG_DIR = "./logs/log_corridors3"

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOING_DIR)
    # n steps mai mare ptr ce ii mai greu
    # model = PPO(
    #     "CnnPolicy",
    #     env,
    #     tensorboard_log=LOG_DIR,
    #     verbose=1,
    #     learning_rate=0.0001,
    #     n_steps=4096,
    # )

    model = PPO(
        "CnnPolicy",
        env,
        tensorboard_log=LOG_DIR,
        verbose=1,
        learning_rate=0.00001,
        n_steps=8192,
        clip_range=0.1,
        gamma=0.95,
        gae_lambda=0.9,
    )

    model.learn(total_timesteps=100000, callback=callback)


def testModel():
    # reload model from disk
    #model = PPO.load("./train/train_corridors3/best_model_100000")
    model = PPO.load("./train//DeadlyCorridor560k")
    env = VizDoomGym(render=True)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    print(mean_reward)
    env.close()


def curriculumLearningv2(model):
    CHECKPOING_DIR = "./train_s2/train_corridor"
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOING_DIR)
    model.load("./train/train_corridor/best_model_400000.zip")
    env = VizDoomGym(config="github/ViZDoom-master/scenarios/deadly_corridor_s2.cfg")
    model.set_env(env)
    model.learn(total_timesteps=40000, callback=callback)

    # to continue we go again but with s3....


def curriculumLearningv3(model):
    CHECKPOING_DIR = "./train_s3/train_corridor"
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOING_DIR)
    model.load("./train/train_corridor/best_model_400000.zip")
    env = VizDoomGym(config="github/ViZDoom-master/scenarios/deadly_corridor_s3.cfg")
    model.set_env(env)
    model.learn(total_timesteps=40000, callback=callback)

    # to continue we go again but with s4....


def curriculumLearningv4(model):
    CHECKPOING_DIR = "./train_s4/train_corridor"
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOING_DIR)
    model.load("./train/train_corridor/best_model_400000.zip")
    env = VizDoomGym(config="github/ViZDoom-master/scenarios/deadly_corridor_s4.cfg")
    model.set_env(env)
    model.learn(total_timesteps=40000, callback=callback)

    # to continue we go again but with s5....


def curriculumLearningv5(model):
    CHECKPOING_DIR = "./train_s5/train_corridor"
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOING_DIR)
    model.load("./train/train_corridor/best_model_400000.zip")
    env = VizDoomGym(config="github/ViZDoom-master/scenarios/deadly_corridor_s5.cfg")
    model.set_env(env)
    model.learn(total_timesteps=40000, callback=callback)

    # to continue we go again but with s3....


def testModelCustomLoop(env, model):
    for episode in range(5):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(0.1)
            total_reward += reward
        print("Total reward for episode {} is {}".format(total_reward, episode))
        time.sleep(2)


def testFunction():
    # setup game
    game = vzd.DoomGame()
    game.load_config("github/ViZDoom-master/scenarios/deadly_corridor_s1.cfg")
    game.init()
    actions = np.identity(3, dtype=np.uint8)
    state = game.get_state()
    print(state.game_variables)


def initializeEnv():
    env = VizDoomGym(render=True)
    env_checker.check_env(env)


if __name__ == "__main__":
    # testFunction()
    # initializeEnv()
    testModel()
    # trainModel()
