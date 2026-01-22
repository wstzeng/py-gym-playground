import gymnasium as gym
from torch import nn
import torch.optim as optim
from agent.encoder import VectorEncoder, CompositeEncoder
from agent.policy import DiscretePolicy
from agent.buffer import ReinforceBuffer
from agent import ReinforceAgent
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

    p_net = nn.Linear(encoder.feature_dim, n_actions)
    policy = DiscretePolicy(network=p_net)

    optimizer = optim.Adam(
        list(v_net.parameters()) + list(p_net.parameters()), 
        lr=1e-3
    )

    buffer = ReinforceBuffer()

    agent = ReinforceAgent(
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
