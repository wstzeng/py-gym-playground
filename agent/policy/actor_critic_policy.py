from .base_policy import BasePolicy

class ActorCriticPolicy(BasePolicy):
    """
    Implementation of Actor-Critic architecture inheriting from BasePolicy.
    """
    def __init__(self, encoder, actor_net, critic_net):
        super().__init__()
        # Use composition to allow different encoders or heads
        self.encoder = encoder
        self.actor = actor_net
        self.critic = critic_net

    def forward(self, x):
        """
        Forward pass returning both action distribution and state value.
        """
        features = self.encoder(x)
        action_logits = self.actor(features)
        state_value = self.critic(features)
        return action_logits, state_value
    
    def get_distribution(self, x):
        """
        Convenience method if only action is needed.
        """
        features = self.encoder(x)
        return self.actor_head(features)

    def get_value(self, x):
        """
        Convenience method if only state value is needed.
        """
        features = self.encoder(x)
        return self.critic_head(features)
