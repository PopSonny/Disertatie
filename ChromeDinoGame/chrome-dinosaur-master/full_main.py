import os
import time
import cv2
import numpy as np
import pytesseract
from mss import mss
from matplotlib import pyplot as plt
import pydirectinput
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class WebGame(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(
            low=0, high=255, shape=(1, 83, 100), dtype=np.uint8
        )
        self.action_space = Discrete(3)
        self.cap = mss()
        self.game_location = {"top": 200, "left": 75, "width": 200, "height": 200}
        self.done_location = {"top": 270, "left": 450, "width": 200, "height": 40}
        self.up_location = {"top": 200, "left": 225, "width": 100, "height": 100}
        self.down_location = {"top": 310, "left": 275, "width": 100, "height": 100}

    def step(self, action, variabila):
        action_map = {0: "space", 1: "down", 2: "no_op"}
        self.down_location["width"] = self.up_location["width"] + variabila
        self.up_location["width"] += variabila
        imgDown, imgUp, pressSpace, pressDown = self.get_enemies()

        if pressSpace:
            action = 0
        elif pressDown:
            action = 1

        if action == 0:
            pydirectinput.press(action_map[action])
        elif action == 1:
            pydirectinput.keyDown(action_map[action])
            time.sleep(0.5)
            pydirectinput.keyUp(action_map[action])

        done, _ = self.get_done()
        new_observation = self.get_observation()
        reward = 1
        return new_observation, reward, done, done, {}

    def render(self):
        cv2.imshow("Game", self.current_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.close()

    def reset(self, *, seed=None, options=None):
        time.sleep(1)
        pydirectinput.click(x=350, y=350)
        pydirectinput.press("w")
        self.up_location = {"top": 200, "left": 225, "width": 100, "height": 100}
        self.down_location = {"top": 310, "left": 275, "width": 100, "height": 100}
        return self.get_observation(), {}

    def close(self):
        cv2.destroyAllWindows()

    def get_observation(self):
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3]
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (100, 83))
        return np.reshape(resized, (1, 83, 100))

    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3]
        gray = cv2.cvtColor(done_cap, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, np.ones((1, 1), np.uint8), iterations=1)
        res = pytesseract.image_to_string(thresh, lang="eng", config="--psm 7")[:4]
        return res in ["Game", "GAHE"], done_cap

    def get_enemies(self):
        rawUp = np.array(self.cap.grab(self.up_location))[:, :, :3]
        grayUp = cv2.cvtColor(rawUp, cv2.COLOR_BGR2GRAY)
        rawDown = np.array(self.cap.grab(self.down_location))[:, :, :3]
        grayDown = cv2.cvtColor(rawDown, cv2.COLOR_BGR2GRAY)
        resizedDown = cv2.resize(grayDown, (300, 300))
        resizedUp = cv2.resize(grayUp, (300, 300))
        pressSpace = np.sum(resizedDown < 100) > 3000
        pressDown = np.sum(resizedUp < 100) > 2000
        return resizedUp, resizedDown, pressSpace, pressDown


class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, f"best_model_{self.n_calls}"))
        return True


def test_environment():
    env = WebGame()
    obs = env.get_observation()
    plt.imshow(cv2.cvtColor(obs[0], cv2.COLOR_BGR2RGB))
    plt.show()


def run_episodes():
    env = WebGame()
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample(), 0
            )
            done = terminated
            total_reward += reward
        print(f"Total reward for episode {episode} is {total_reward}")


def train_model():
    env = WebGame()
    env_checker.check_env(env)
    callback = TrainAndLoggingCallback(check_freq=1000, save_path="./train/")
    model = DQN(
        "CnnPolicy",
        env,
        tensorboard_log="./logs/",
        verbose=1,
        buffer_size=1000000,
        learning_starts=1000,
    )
    model.learn(total_timesteps=5000, callback=callback)


def test_trained_model():
    # model = DQN.load(os.path.join("train", "best_model_88000"))
    env = WebGame()
    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            _, _, pressSpace, pressDown = env.get_enemies()
            #action, _ = model.predict(obs)
            action = 0 if pressSpace else 1 if pressDown else 2
            obs, reward, terminated, truncated, _ = env.step(
                int(action), int(total_reward / 60)
            )
            done = terminated
            total_reward += reward
        print(f"Total Reward for episode {episode} is {total_reward}")


if __name__ == "__main__":
    # You can call any of the following:
    #test_environment()
    # run_episodes()
    # train_model()
    test_trained_model()
    pass
