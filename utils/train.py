import gymnasium as gym
from agent import BaseAgent
from .monitor.training_monitor import TrainingMonitor as Monitor

def train_loop(
    env_name: str,
    agent: BaseAgent,
    T: int,
    N: int,
    monitor_mode = 'cli',
):
    env = gym.make(env_name)
    monitor = Monitor(env_name, T, mode=monitor_mode, window_size=50)

    for t in range(1, T + 1):
        total_rewards = []

        for _ in range(N):
            state, _ = env.reset()
            agent.start_episode()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                agent.record_reward(reward)
                state = next_state
                done = terminated or truncated
                total_reward += reward

            agent.end_episode()
            total_rewards.append(total_reward)

        loss = agent.update_policy()
        avg_reward = sum(total_rewards) / N
        monitor.update(t, avg_reward, loss)

    monitor.close()
    env.close()
