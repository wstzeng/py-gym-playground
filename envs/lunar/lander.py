import gymnasium as gym
from policy.discrete_policy import DiscretePolicyNetwork as Policy
from agent import ReinforceAgent as Agent

def main(env_name='LunarLander-v3', T=200, N=10):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    hidden_sizes = [128, 64, 32]
    policy = Policy(state_dim, n_actions, hidden_sizes=hidden_sizes)
    agent = Agent(policy, lr=1e-4, gamma=0.99, device='cuda')

    for t in range(1, T + 1):
        total_rewards = []

        for _ in range(N):
            state, _ = env.reset()
            done = False
            total_reward = 0
            agent.start_episode()

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
        print(f"[{t:3d}/{T}] Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")

    env.close()

    # Render a test episode
    env = gym.make(env_name, render_mode='human')
    state, _ = env.reset(seed=42)
    for _ in range(1000):
        env.render()
        action = agent.policy.select_action(state)[0]
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()
    env.close()

if __name__ == '__main__':
    main()
