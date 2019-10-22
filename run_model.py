from argparse import ArgumentParser
from unityagents import UnityEnvironment
import torch
from utils.utils import unwrap_env_info
from src.agent import Agent

def main(args):
    env = UnityEnvironment(args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=False)[brain_name]

    state = env_info.vector_observations[0]
    state_size = len(state)
    action_size = brain.vector_action_space_size

    agent = Agent(state_size=state_size, action_size=action_size, seed=25)
    agent.qnetwork_local.load_state_dict(torch.load(args.model))

    score = 0

    while True:
        action = agent.act(state)
        env_info = env.step(action)[brain_name]
        next_state, reward, done = unwrap_env_info(env_info)
        score += reward
        state = next_state

        if done:
            break

    env.close()
    print(f'Score: {score}')

def argument_parser():
    parser = ArgumentParser(description='Running the Deep Q-Learning model on an Unity Environment')

    parser.add_argument('--env',
                        type=str,
                        default='Banana.app',
                        help='The path of the Unity Environment')

    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='Where to load the model')
    
    return parser.parse_args()

if __name__ == '__main__':
    ARGS = argument_parser()
    main(ARGS)
