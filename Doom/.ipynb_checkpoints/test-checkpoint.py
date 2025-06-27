import vizdoom as vzd
import random
import time
import numpy as np

game = vzd.DoomGame()
game.load_config("github/ViZDoom-master/scenarios/basic.cfg")


if __name__ == "__main__":
    game.init()
    actions = np.identity(3, dtype=np.uint8)
    
    episodes = 10
    for episode in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            img = state.screen_buffer
            info = state.game_variables
            reward = game.make_action(random.choice(actions))
            print('reward:', reward)
            time.sleep(0.02)
        print('Result:', game.get_total_reward())
        time.sleep(2)
