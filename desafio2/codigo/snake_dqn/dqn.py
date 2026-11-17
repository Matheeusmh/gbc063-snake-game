"""
dqn.py
======
Implementação completa de Deep Q-Network (DQN) e Double DQN em PyTorch do zero.
Sem uso de Gym/Gymnasium nem stable-baselines.
Disciplina GBC063 - Inteligência Artificial (UFU).

Componentes:
- ReplayBuffer: buffer circular para Experience Replay com tuplas (s, a, r, s', done_terminal)
- DQNNetwork: rede neural MLP (11 -> hidden_size -> 3)
- DQNAgent: agente DQN com política ε-greedy, rede-alvo (hard update) e suporte a Double DQN
"""

import os
import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Experience Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Buffer circular de repetição de experiências (Experience Replay).
    Quebra a correlação temporal entre amostras consecutivas.
    """

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done_terminal: bool):
        """
        Armazena uma transição no buffer.

        ATENÇÃO: 'done_terminal' deve ser True SOMENTE em estados terminais reais
        (colisão ou vitória). Em caso de truncamento por timeout, done_terminal DEVE
        ser False para não corromper o cálculo da função valor de Bellman com bootstrap.
        """
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done_terminal),
        ))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Amostra uniformemente um lote de transições e converte para tensores PyTorch.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t      = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t     = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t     = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t       = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        return states_t, actions_t, rewards_t, next_states_t, dones_t

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# 2. Rede Neural de Aproximação de Q (DQNNetwork)
# =============================================================================

class DQNNetwork(nn.Module):
    """
    Rede neural MLP com arquitetura:
        Linear(input_size, hidden_size) -> ReLU -> Linear(hidden_size, output_size)
    """

    def __init__(self, input_size: int = 11, hidden_size: int = 256, output_size: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagação direta: retorna os Q-valores para cada uma das 3 ações."""
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =============================================================================
# 3. Agente DQN (DQNAgent)
# =============================================================================

class DQNAgent:
    """
    Agente DQN com suporte opcional a Double DQN.

    Hiperparâmetros padrão:
    - Otimizador: Adam com lr = 1e-3
    - Perda: Huber Loss (Smooth L1)
    - Gradient Clipping: norma máxima 10.0
    - Warmup: 1000 transições no buffer antes do treino
    - Decaimento de epsilon: multiplicativo por episódio
    """

    def __init__(
        self,
        state_size: int = 11,
        action_size: int = 3,
        hidden_size: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.95,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update: int = 500,
        eps_start: float = 1.0,
        eps_min: float = 0.01,
        eps_decay: float = 0.995,
        double_dqn: bool = False,
        warmup: int = 1000,
        device: Optional[torch.device] = None,
        # Aliases para compatibilidade flexível
        **kwargs,
    ):
        # Trata aliases caso venham de outros scripts
        state_size = kwargs.get("state_dim", state_size)
        action_size = kwargs.get("n_actions", action_size)
        target_update = kwargs.get("target_update_freq", target_update)
        eps_start = kwargs.get("epsilon", eps_start)
        eps_min = kwargs.get("epsilon_min", eps_min)
        eps_decay = kwargs.get("epsilon_decay", eps_decay)
        double_dqn = kwargs.get("double", double_dqn)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.double_dqn = double_dqn
        self.warmup = warmup

        # Dispositivo de processamento (CPU ou GPU)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Redes Principal (Policy) e Alvo (Target)
        self.policy_net = DQNNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, hidden_size, action_size).to(self.device)

        # Cópia hard inicial dos pesos para a rede alvo
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Rede alvo sempre em modo de avaliação

        # Otimizador Adam
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Replay Buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

        # Estado do agente
        self.epsilon = eps_start
        self.steps_done = 0
        self.losses = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Seleciona uma ação para o estado dado.
        Se training=True: política ε-greedy.
        Se training=False: política puramente gulosa (argmax Q).
        """
        if training and (random.random() < self.epsilon):
            return random.randint(0, self.action_size - 1)

        # Inferência gulosa pela policy_net
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.policy_net.eval()
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        self.policy_net.train()

        return int(torch.argmax(q_values, dim=1).item())

    def push_transition(self, state, action: int, reward: float, next_state, done_terminal: bool):
        """Armazena transição no buffer de repetição."""
        self.memory.push(state, action, reward, next_state, done_terminal)

    def update(self) -> Optional[float]:
        """
        Executa um passo de otimização dos pesos da policy_net.
        Retorna o valor da perda (loss) ou None se estiver no período de warmup.
        """
        if len(self.memory) < self.warmup:
            return None

        # 1. Amostra lote do buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # 2. Q-valores previstos pela policy_net para as ações tomadas
        q_predicted = self.policy_net(states).gather(1, actions)

        # 3. Cálculo do valor alvo de Bellman:
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN:
                # Ação ótima a* escolhida pela policy_net:
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                # Avaliação de a* feita pela target_net:
                q_next = self.target_net(next_states).gather(1, best_actions)
            else:
                # DQN Padrão: max_a' Q_target(s', a')
                q_next = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            # Se done_terminal for True (colisão ou vitória), não há recompensa futura
            q_target = rewards + self.gamma * (1.0 - dones) * q_next

        # 4. Cálculo da perda Huber (Smooth L1 Loss)
        loss = F.smooth_l1_loss(q_predicted, q_target)

        # 5. Otimização e clipping de gradiente
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 6. Atualização periódica da rede alvo (hard update)
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        """Aplica o decaimento multiplicativo de ε ao final de cada episódio."""
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

    def save(self, path: str):
        """Salva os pesos da policy_net e o estado do agente em arquivo .pt."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "epsilon": self.epsilon,
            "steps_done": self.steps_done,
        }, path)

    def load(self, path: str):
        """Carrega os pesos salvos para a policy_net e sincroniza a target_net."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "policy_net" in checkpoint:
            self.policy_net.load_state_dict(checkpoint["policy_net"])
            self.epsilon = checkpoint.get("epsilon", self.epsilon)
            self.steps_done = checkpoint.get("steps_done", self.steps_done)
        else:
            # Caso tenha sido salvo apenas o state_dict puro
            self.policy_net.load_state_dict(checkpoint)

        # Sincroniza a target_net com os pesos carregados
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
