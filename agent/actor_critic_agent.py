# agents/actor_critic_agent.py
import torch
from .base_agent import BaseAgent

class ActorCriticAgent(BaseAgent):
    def __init__(
            self, encoder, policy, buffer, optimizer, 
            gamma=0.99, device="auto"):
        super().__init__(encoder, device=device)
        self.policy = policy.to(self.device)
        self.buffer = buffer
        self.gamma = gamma
        self.optimizer = optimizer

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        features = self.encoder(state_tensor)
        
        logits, value = self.policy(features)
        
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        
        info = {
            "log_prob": dist.log_prob(action),
            "value": value
        }
        
        return action.item(), info

    def record(self, info, reward, done):
        self.buffer.store(
            log_prob=info["log_prob"],
            value=info["value"],
            reward=reward,
            done=done
        )

    def update(self):
        log_probs, values, rewards, dones = self.buffer.get_data()
        if not log_probs:
            return 0.0

        returns = []
        g = 0
        for r, d in zip(reversed(rewards), reversed(dones)):
            g = r + self.gamma * g * (1 - d)
            returns.insert(0, g)

        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()  # From [T, 1] to [T]

        advantages = returns - values.detach()

        actor_loss = -(log_probs * advantages).mean()
        
        critic_loss = torch.nn.functional.mse_loss(values, returns)
        
        loss = actor_loss + 0.5 * critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.buffer.clear()
        return loss.item()
