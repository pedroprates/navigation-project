from argparse import ArgumentParser
import time
from collections import deque
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment
from src.agent import Agent

def main(args):
    """ Training a Deep Q-Learning algorithm on the Unity Environment """
    env = UnityEnvironment(args.env)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state_size = len(env_info.vector_observations[0])

    agent = Agent(state_size=state_size, action_size=action_size, seed=args.seed)
    scores = dqn(agent=agent, env=env, brain_name=brain_name, model_path=args.model,
                n_episodes=args.episodes, max_t=args.steps, eps_start=args.eps_start,
                eps_end=args.eps_min, eps_decay=args.eps_decay) 

    fig = plt.figure()
    _ = fig.add_suplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.xlabel('Score')
    plt.ylabel('Episode #')
    plt.show()

    env.close()

def dqn(agent, env, brain_name, model_path, n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """ Deep Q-Learning
    Params
    =====
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): first value of epsilon (for epsilon-greedy selection)
        eps_end (float): minimum value of epsilon
        eps_decay (float): the decay rate of epsilon per episode
    """
    
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        for _ in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            next_state, reward, done = unwrap_env_info(env_info)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_decay*eps, eps_end)
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}', end='')
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

    save_model(agent, model_path)
    return scores

def save_model(agent: Agent, model_path: str):
    timestamp = time.strftime('%b-%d-%Y_%H%M', time.localtime())
    model_pth = os.path.join(model_path, f'checkpoint_{timestamp}.pth')
    torch.save(agent.qnetwork_local.state_dict(), model_pth)

def parse_arguments():
    parser = ArgumentParser(description='Parsing the parameters to run the Deep Q-Learning algorithm')
    parser.add_argument('--env',
                        type=str,
                        default='Banana.app',
                        help='The path of the Unity Environment')

    parser.add_argument('--model',
                        type=str,
                        default='models/',
                        help='Where to save the model')

    parser.add_argument('--episodes',
                        type=int,
                        default=2000,
                        help='The number of episodes the model will be trained on.')

    parser.add_argument('--steps',
                        type=int,
                        default=1000,
                        help='The number of moves in one episode')

    parser.add_argument('--eps_start',
                        type=float,
                        default=1.0,
                        help='The max epsilon to be used on the epsilon-greedy selection')

    parser.add_argument('--eps_min',
                        type=float,
                        default=.01,
                        help='The minimum epsilon to be used on the epsilon-greedy selection')

    parser.add_argument('--eps_decay',
                        type=float,
                        default=0.995,
                        help='The decay rate of the epsilon per episode')

    parser.add_argument('--seed',
                        type=int,
                        default=25,
                        help='The seed to input to random functions')

    return parser.parse_args()


def unwrap_env_info(obj):
    return obj.vector_observations[0], obj.rewards[0], obj.local_done[0]

if __name__ == '__main__':
    print('Starting code...')
    args = parse_arguments()
    main(args)
