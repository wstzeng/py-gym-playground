import gymnasium as gym
from agent import BaseAgent

def test_loop(
    env_name: str,
    agent: BaseAgent,
    T: int = 1000,
    S: int = 5,
):
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset()
    for _ in range(S):
        for _ in range(T):
            env.render()
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state)
            elif hasattr(agent, 'policy') and hasattr(agent.policy, 'select_action'):
                action = agent.policy.select_action(state)[0]
            else:
                raise AttributeError('Agent does not have a valid `select_action` method.')

            state, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                state, _ = env.reset()
                break
    env.close()
