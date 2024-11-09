from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import logging
from collections import deque
from langchain_core.globals import set_verbose, set_debug
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from sentence_transformers import SentenceTransformer
import gc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Experience:
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool
    log_prob: float
    value: float

class MemoryBuffer:
    def __init__(self, max_size: int = 10000):
        self.buffer = deque(maxlen=max_size)
        
    def push(self, experience: Experience):
        self.buffer.append(experience)
        
    def clear(self):
        self.buffer.clear()
        
    def __len__(self):
        return len(self.buffer)
        
    def get_batch(self, batch_size: int) -> List[Experience]:
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        return [self.buffer[i] for i in indices]

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        action_probs = F.softmax(self.policy_net(state), dim=-1)
        value = self.value_net(state)
        return action_probs, value
    
    @torch.no_grad()
    def act(self, state: torch.Tensor) -> Tuple[int, float, float]:
        action_probs, value = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = MemoryBuffer()
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device
        
    def update(self):
        if len(self.memory) < self.batch_size:
            return
            
        for _ in range(self.n_epochs):
            experiences = self.memory.get_batch(self.batch_size)
            
            states = torch.stack([e.state for e in experiences]).to(self.device)
            actions = torch.tensor([e.action for e in experiences]).to(self.device)
            rewards = torch.tensor([e.reward for e in experiences]).to(self.device)
            next_states = torch.stack([e.next_state for e in experiences]).to(self.device)
            dones = torch.tensor([e.done for e in experiences]).to(self.device)
            old_log_probs = torch.tensor([e.log_prob for e in experiences]).to(self.device)
            old_values = torch.tensor([e.value for e in experiences]).to(self.device)
            
            # Compute advantages
            with torch.no_grad():
                _, next_values = self.policy(next_states)
                next_values = next_values.squeeze()
                td_target = rewards + self.gamma * next_values * (1 - dones)
                advantages = td_target - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            action_probs, values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            values = values.squeeze()
            
            # Policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, td_target)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + self.c1 * value_loss - self.c2 * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            # Free memory
            del states, actions, rewards, next_states, dones, advantages
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
        
        self.memory.clear()

class ChatAgentWithRL:
    def __init__(
        self,
        llm_model: str = "phi",
        state_dim: int = 384,  # BERT embedding dimension
        action_dim: int = 2,  # Binary action space for demonstration
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = ChatOllama(
            model=llm_model,
            num_gpu=1 if torch.cuda.is_available() else 0,
            num_thread=4,
            temperature=0.7,
            num_predict=2048,
            top_k=30,
            repeat_penalty=1.1
        )
        
        self.prompt = ChatPromptTemplate.from_template("""
            You are a helpful assistant that provides concise answers.
            Question: {question}
            Please provide a clear and concise answer.
        """)
        
        # Initialize sentence encoder for state representation
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=device)
        
        # Initialize PPO agent
        self.agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
            batch_size=16,  # Smaller batch size for memory efficiency
            hidden_dim=64   # Smaller hidden dimension
        )
        
    def encode_text(self, text: str) -> torch.Tensor:
        with torch.no_grad():
            embedding = self.encoder.encode(text, convert_to_tensor=True)
            return embedding.to(self.device)
    
    def generate_response(self, query: str, state: torch.Tensor) -> Tuple[str, int, float, float]:
        action, log_prob, value = self.agent.policy.act(state)
        
        # Use action to modify response generation (e.g., temperature, response length)
        temp_mapping = {0: 0.5, 1: 0.8}  # Map actions to different temperatures
        self.model.temperature = temp_mapping[action]
        
        chain = self.prompt | self.model | StrOutputParser()
        response = chain.invoke({"question": query})
        
        return response, action, log_prob, value
    
    @torch.no_grad()
    def calculate_reward(self, query: str, response: str) -> float:
        # Simplified reward calculation based on response length and query relevance
        query_embedding = self.encode_text(query)
        response_embedding = self.encode_text(response)
        
        # Compute cosine similarity between query and response
        similarity = F.cosine_similarity(query_embedding.unsqueeze(0), 
                                      response_embedding.unsqueeze(0)).item()
        
        # Penalize extremely short or long responses
        length_penalty = -abs(len(response.split()) - 50) / 100
        
        reward = similarity + length_penalty
        return max(min(reward, 1.0), -1.0)  # Clip reward to [-1, 1]
    
    def chat(self, query: str) -> str:
        try:
            # Encode current state
            current_state = self.encode_text(query)
            
            # Generate response using current policy
            response, action, log_prob, value = self.generate_response(query, current_state)
            
            # Calculate reward
            reward = self.calculate_reward(query, response)
            
            # Store experience
            next_state = self.encode_text(response)
            experience = Experience(
                state=current_state,
                action=torch.tensor(action),
                reward=reward,
                next_state=next_state,
                done=True,  # Single-step episode
                log_prob=log_prob,
                value=value
            )
            self.agent.memory.push(experience)
            
            # Update policy if enough experiences are collected
            self.agent.update()
            
            # Free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
            
            return response
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            return f"An error occurred while processing your question: {str(e)}"