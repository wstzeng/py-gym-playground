import gymnasium as gym
from torch import nn
import torch.optim as optim
from agent.encoder import VectorEncoder, CompositeEncoder
from agent.policy import ActorCriticPolicy
from agent.buffer import ActorCriticBuffer
from agent import ActorCriticAgent
from utils.train import train_loop
from utils.test import test_loop

def main(env_name='LunarLander-v3', T=200, N=10):
    # Initialize environment to extract dimensions
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    # Construct agent and monitor
    v_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU()
    )
    encoder = VectorEncoder(network=v_net, input_dim=state_dim)

    actor_head = nn.Linear(encoder.feature_dim, n_actions)
    critic_head = nn.Linear(encoder.feature_dim, 1)
    
    policy = ActorCriticPolicy(actor_net=actor_head, critic_net=critic_head)

    optimizer = optim.Adam([
        {'params': encoder.parameters()},
        {'params': policy.parameters()}
    ], lr=1e-3)

    buffer = ActorCriticBuffer()

    agent = ActorCriticAgent(
        encoder=encoder,
        policy=policy,
        buffer=buffer,
        optimizer=optimizer, 
        device="cpu"
    )

    # Run training
    train_loop(env_name, agent, T, N, monitor_mode=['cli', 'live', 'file'])

    # Render a test episode
    test_loop(env_name, agent)

    agent.save_policy(f'checkpoints/{env_name}.ckpt')

if __name__ == '__main__':
    main(T=500)
