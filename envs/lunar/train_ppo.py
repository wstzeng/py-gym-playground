import gymnasium as gym
from torch import nn
import torch.optim as optim
from agent.encoder import VectorEncoder
from agent.policy import ActorCriticPolicy
from agent import PPOAgent
from agent.buffer import PPOBuffer
from utils.train import train_loop
from utils.test import test_loop

def main(env_name='LunarLander-v3', T=200, N=10):
    # Initialize environment to extract dimensions
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    # Shared feature extractor (Encoder)
    v_net = nn.Sequential(
        nn.Linear(state_dim, 128),
        nn.LeakyReLU(),
        nn.Linear(128, 128),
        nn.LeakyReLU()
    )
    encoder = VectorEncoder(network=v_net, input_dim=state_dim)

    # Actor and Critic heads
    actor_head = nn.Linear(encoder.feature_dim, n_actions)
    critic_head = nn.Sequential(
        nn.Linear(encoder.feature_dim, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 32),
        nn.LeakyReLU(),
        nn.Linear(32, 1)
    )
    
    # PPO uses the same ActorCriticPolicy structure
    policy = ActorCriticPolicy(actor_net=actor_head, critic_net=critic_head)

    # Differential learning rates with AdamW
    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': 1e-3},
        {'params': policy.actor.parameters(), 'lr': 1e-3},
        {'params': policy.critic.parameters(), 'lr': 1e-4},
    ], weight_decay=1e-2)

    # Use PPO-specific buffer
    buffer = PPOBuffer()

    # Initialize PPOAgent with clipped objective settings
    agent = PPOAgent(
        encoder=encoder,
        policy=policy,
        buffer=buffer,
        optimizer=optimizer,
        device="cpu",
        eps_clip=0.2,
        k_epochs=10,
        entropy_weight=0.01
    )

    # Run training
    train_loop(env_name, agent, T, N, monitor_mode=['cli', 'live', 'file'])

    # Render a test episode
    test_loop(env_name, agent)

    agent.save_checkpoints(f'checkpoints/{env_name}_ppo.ckpt')

if __name__ == '__main__':
    # Starting with T=500 to compare with previous AC results
    main(T=500)
