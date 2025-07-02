import gymnasium as gym
from agent import BaseAgent

def test_loop(
    env_name: str,
    agent: BaseAgent,
    T: int = 1000,
):
    render_env = gym.make(env_name, render_mode='human')
    state, _ = render_env.reset(seed=42)
    for _ in range(T):
        render_env.render()
        action = agent.policy.select_action(state)[0]
        state, _, terminated, truncated, _ = render_env.step(action)
        if terminated or truncated:
            state, _ = render_env.reset()
    render_env.close()

