"""
Neural network architecture for the sword fight bot
Lightweight design optimized for Intel Iris Xe integrated graphics
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron for feature-based input
    Optimized for CPU inference with minimal parameters
    """
    
    def __init__(self, input_dim: int, 
                 hidden_layers: List[int] = [256, 128, 64],
                 output_dim: int = 8,
                 dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize MLP network
        
        Args:
            input_dim: Number of input features
            hidden_layers: List of hidden layer sizes
            output_dim: Number of output actions/values
            dropout: Dropout probability
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(MLPNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Select activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.ReLU()
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        # Calculate total parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"MLP Network initialized with {self.num_params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> torch.Tensor:
        """
        Get action from state
        
        Args:
            state: Current state
            deterministic: If True, use argmax; otherwise sample
            
        Returns:
            Action tensor
        """
        with torch.no_grad():
            output = self.forward(state)
            
            if deterministic:
                return torch.argmax(output, dim=-1)
            else:
                # Add small noise for exploration
                probs = F.softmax(output, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic network for PPO reinforcement learning
    Shared backbone with separate heads for policy and value
    """
    
    def __init__(self, input_dim: int, 
                 action_dim: int,
                 hidden_layers: List[int] = [256, 128, 64],
                 dropout: float = 0.1):
        """
        Initialize Actor-Critic network
        
        Args:
            input_dim: Number of input features
            action_dim: Number of possible actions
            hidden_layers: List of hidden layer sizes
            dropout: Dropout probability
        """
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extractor
        self.shared = MLPNetwork(
            input_dim=input_dim,
            hidden_layers=hidden_layers[:-1],  # One less layer for shared
            output_dim=hidden_layers[-1],
            dropout=dropout
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1] // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_layers[-1], hidden_layers[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_layers[-1] // 2, 1)
        )
        
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Actor-Critic Network initialized with {self.num_params:,} parameters")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input state tensor
            
        Returns:
            Tuple of (action_logits, state_value)
        """
        features = self.shared.network[:-1](x)  # Get features before final layer
        
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, state_value
    
    def get_action(self, state: torch.Tensor, 
                   deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action with log probability and value
        
        Args:
            state: Current state
            deterministic: If True, use greedy action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            logits, value = self.forward(state)
            probs = F.softmax(logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            
            log_prob = F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(-1)).squeeze(-1)
            
            return action, log_prob, value


def create_model(model_type: str, input_dim: int, output_dim: int,
                config: dict) -> nn.Module:
    """
    Factory function to create models based on configuration
    
    Args:
        model_type: Type of model ('mlp', 'actor_critic', 'cnn')
        input_dim: Input dimension
        output_dim: Output dimension
        config: Configuration dictionary
        
    Returns:
        Initialized model
    """
    if model_type == 'mlp':
        return MLPNetwork(
            input_dim=input_dim,
            hidden_layers=config.get('hidden_layers', [256, 128, 64]),
            output_dim=output_dim,
            dropout=config.get('dropout', 0.1),
            activation=config.get('activation', 'relu')
        )
    
    elif model_type == 'actor_critic':
        return ActorCriticNetwork(
            input_dim=input_dim,
            action_dim=output_dim,
            hidden_layers=config.get('hidden_layers', [256, 128, 64]),
            dropout=config.get('dropout', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test network creation
    print("Testing MLP Network...")
    mlp = MLPNetwork(input_dim=30, hidden_layers=[256, 128, 64], output_dim=8)
    
    # Test forward pass
    dummy_input = torch.randn(4, 30)  # Batch of 4
    output = mlp(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\nTesting Actor-Critic Network...")
    ac = ActorCriticNetwork(input_dim=30, action_dim=8)
    
    # Test forward pass
    logits, value = ac(dummy_input)
    print(f"Logits shape: {logits.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test action sampling
    action, log_prob, val = ac.get_action(dummy_input)
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
