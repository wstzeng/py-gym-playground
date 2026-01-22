import gymnasium as gym
from agent import BaseAgent

def test_loop(
    env_name: str,
    agent: BaseAgent,
    T: int = 1000,
    S: int = 5,
):
    env = gym.make(env_name, render_mode='human')
    
    for _ in range(S):
        state, _ = env.reset()
        for _ in range(T):
            env.render()
            
            # Unpack (action, info) based on the new standardized agent interface
            # We only need 'action' for the environment step
            action, info = agent.select_action(state)

            state, _, terminated, truncated, _ = env.step(action)
            
            if terminated or truncated:
                break
    
    env.close()
