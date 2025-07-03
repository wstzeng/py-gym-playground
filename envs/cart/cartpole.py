import gymnasium as gym

from policy.discrete_policy import DiscretePolicyNetwork as Policy
from agents import ReinforceAgent as Agent
from utils.train import train_loop
from utils.test import test_loop

def main(env_name='CartPole-v1', T=200, N=10):
    # Initialize environment to extract dimensions
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    # Construct agent and monitor
    actor = Policy(state_dim, n_actions)
    agent = Agent(actor, lr=1e-3, gamma=0.99, device='cpu')

    # Run training
    train_loop(env_name, agent, T, N, monitor_mode=['cli', 'live', 'file'])

    # Render a test episode
    test_loop(env_name, agent)
    
    agent.save_policy(f'checkpoints/{env_name}.ckpt')

if __name__ == '__main__':
    main()
