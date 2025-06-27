# main.py

import torch
import numpy as np
from gamev2 import SnakeGameAI
from agentv2 import Agent
from modelv2 import Linear_QNet

def main():
    # 1. Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 2. Create an Agent (for its get_state method) and replace its model
    agent = Agent()
    agent.epsilon = 0              # turn off exploration
    agent.model = Linear_QNet(14, 256, 3).to(device)  # match your final input_size
    agent.model.load_state_dict(
        torch.load('model/bestModel.pth', map_location=device)
    )
    agent.model.eval()

    # 3. Launch the game
    game = SnakeGameAI()

    while True:
        # 4. Get current state
        state = agent.get_state(game)

        # 5. Model inference
        state_v = torch.tensor(state, dtype=torch.float).to(device)
        with torch.no_grad():
            preds = agent.model(state_v)
        move = torch.argmax(preds).item()
        final_move = [0, 0, 0]
        final_move[move] = 1

        # 6. Play one step
        reward, done, score = game.play_step(final_move)

        if done:
            print(f"Game over! Final Score: {score}")
            break

if __name__ == '__main__':
    main()
