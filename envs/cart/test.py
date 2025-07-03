import gymnasium as gym
from policy.discrete_policy import DiscretePolicyNetwork as Policy
from agents import ReinforceAgent as Agent
from utils.test import test_loop

def main(env_name='CartPole-v1'):
    # Initialize environment to extract dimensions
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    env.close()

    # Construct agent and monitor
    actor = Policy(state_dim, n_actions)
    agent = Agent(actor, lr=1e-3, gamma=0.99, device='cpu')

    agent.load_policy(f'checkpoints/{env_name}.ckpt')

    # Render a test episode
    test_loop(env_name, agent)    

if __name__ == '__main__':
    main()
