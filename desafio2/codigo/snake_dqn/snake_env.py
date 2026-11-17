"""
snake_env.py
============
Ambiente do jogo Snake implementado do zero para Aprendizado por Reforço (DQN).
Disciplina GBC063 - Inteligência Artificial (UFU).

Dinâmica:
- Grade grid_size × grid_size com paredes nas bordas (sair da grade colide com a parede).
- Cobra inicia no centro com tamanho 3, orientada para a direita.
- Comida sorteada aleatoriamente de forma uniforme em células livres.
- Ações relativas (3): 0 = reto, 1 = virar à direita, 2 = virar à esquerda.
- Recompensas: comer = +10.0, colisão = -100.0, passo normal = -0.1.
- Timeout / truncamento: se passos_sem_comer > 100 * tamanho_da_cobra.
- Observação parcialmente observável: vetor de 11 floats binários (0.0 ou 1.0).
"""

import argparse
import numpy as np


class SnakeEnv:
    """
    Ambiente Snake com interface no padrão Gym:
        reset() -> obs
        step(action) -> (obs, reward, done, info)

    Parâmetros
    ----------
    grid_size : int, padrão 10
        Dimensão da grade quadrada (jogável de 0 a grid_size - 1).
    seed : int ou None, padrão None
        Semente para o gerador de números aleatórios (np.random.RandomState).
    """

    # Símbolos direcionais para renderização do corpo
    _DIR_SYMBOLS = {
        (-1,  0): "^",   # Cima
        ( 1,  0): "v",   # Baixo
        ( 0, -1): "<",   # Esquerda
        ( 0,  1): ">",   # Direita
    }

    def __init__(self, grid_size: int = 10, seed: int = None):
        self.grid_size = grid_size
        self.rng = np.random.RandomState(seed)

        # Estado real interno S (não exposto integralmente ao agente)
        self.snake = []               # Lista de tuplas (row, col); cabeça é snake[0]
        self.direction = (0, 1)       # Tupla (dr, dc): direção atual do movimento
        self.food = (0, 0)            # Posição (row, col) da comida
        self.steps_without_food = 0   # Contador de passos sem comer (usado para timeout)

        # Métricas internas do episódio
        self.score = 0                # Total de maçãs comidas
        self.steps = 0                # Total de passos no episódio

        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reinicia o ambiente para o estado inicial e retorna a observação inicial.

        Retorna
        -------
        np.ndarray
            Vetor de 11 floats float32 com a observação O.
        """
        # Cobra inicia no centro da grade com tamanho 3, voltada para a direita
        cy = self.grid_size // 2
        cx = self.grid_size // 2

        self.snake = [
            (cy, cx),       # Cabeça
            (cy, cx - 1),   # Corpo
            (cy, cx - 2),   # Cauda
        ]
        self.direction = (0, 1)       # Direita: (dr=0, dc=+1)
        self.steps_without_food = 0
        self.score = 0
        self.steps = 0

        # Sorteia a comida em uma célula livre
        self.food = self._place_food()

        return self._get_obs()

    def step(self, action: int):
        """
        Executa uma ação no ambiente e avança a simulação em um passo.

        Parâmetros
        ----------
        action : int
            0 = seguir reto
            1 = virar 90° à direita (horário)
            2 = virar 90° à esquerda (anti-horário)

        Retorna
        -------
        obs : np.ndarray
            Vetor de observação com shape (11,) dtype float32.
        reward : float
            Recompensa imediata (+10.0 comer, -100.0 colisão, -0.1 passo normal).
        done : bool
            True se o episódio terminou (colisão, vitória ou timeout).
        info : dict
            Dicionário com 'score', 'steps' e 'truncated'.
        """
        # 1. Atualiza direção conforme ação relativa
        self.direction = self._turn(self.direction, action)

        # 2. Calcula a nova posição da cabeça
        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

        self.steps += 1
        self.steps_without_food += 1

        # 3. Verifica colisão com a parede (fora dos limites da grade 0 .. grid_size - 1)
        if self._out_of_bounds(new_head):
            reward = -100.0
            done = True
            info = {"score": self.score, "steps": self.steps, "truncated": False}
            return self._get_obs(), reward, done, info

        # 4. Verifica se a nova cabeça come a comida
        ate = (new_head == self.food)

        # 5. Tratamento de colisão com o próprio corpo:
        # Se NÃO comeu: a cauda é removida no mesmo passo, logo a cabeça pode
        # ocupar a célula onde a cauda estava sem colidir.
        # Se COMEU: a cauda NÃO é removida (cobra cresce), portanto a cauda é obstáculo.
        body_to_check = self.snake[:-1] if not ate else self.snake[:]
        if new_head in body_to_check:
            reward = -100.0
            done = True
            info = {"score": self.score, "steps": self.steps, "truncated": False}
            return self._get_obs(), reward, done, info

        # 6. Movimento bem-sucedido: adiciona nova cabeça
        self.snake.insert(0, new_head)

        if ate:
            self.score += 1
            self.steps_without_food = 0
            reward = 10.0

            # Verifica vitória: cobra preencheu toda a grade
            if len(self.snake) == self.grid_size * self.grid_size:
                done = True
                info = {"score": self.score, "steps": self.steps, "truncated": False}
                return self._get_obs(), reward, done, info

            # Sorteia nova comida
            self.food = self._place_food()
        else:
            # Não comeu: remove o último segmento da cauda
            self.snake.pop()
            reward = -0.1

        # 7. Verifica timeout (truncamento)
        # Timeout encerra o episódio se passos_sem_comer > 100 * tamanho_da_cobra
        truncated = self.steps_without_food > 100 * len(self.snake)
        done = truncated

        info = {
            "score": self.score,
            "steps": self.steps,
            "truncated": truncated,
        }

        return self._get_obs(), reward, done, info

    def render_ascii(self):
        """
        Exibe a grade no terminal em formato ASCII para depuração visual.
        Moldura com '#', cabeça com 'H', corpo com direcional, comida com 'F'.
        """
        head = self.snake[0]
        body_set = set(self.snake[1:])
        body_symbol = self._DIR_SYMBOLS.get(self.direction, "*")

        print("+" + "---" * self.grid_size + "+")
        for r in range(self.grid_size):
            linha = ["|"]
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == head:
                    linha.append(" H ")
                elif pos in body_set:
                    linha.append(f" {body_symbol} ")
                elif pos == self.food:
                    linha.append(" F ")
                else:
                    linha.append(" . ")
            linha.append("|")
            print("".join(linha))
        print("+" + "---" * self.grid_size + "+")
        print(f"Passos: {self.steps} | Pontuação: {self.score} | Sem comer: {self.steps_without_food}\n")

    # =========================================================================
    # Métodos auxiliares
    # =========================================================================

    def _turn(self, direction: tuple, action: int) -> tuple:
        """
        Calcula nova direção absoluta a partir da direção atual e ação relativa:
            0 = reto: (dr, dc)
            1 = virar à direita (horário): (dc, -dr)
            2 = virar à esquerda (anti-horário): (-dc, dr)
        """
        dr, dc = direction
        if action == 0:
            return (dr, dc)
        elif action == 1:
            return (dc, -dr)
        elif action == 2:
            return (-dc, dr)
        else:
            raise ValueError(f"Ação inválida: {action}. Use 0, 1 ou 2.")

    def _out_of_bounds(self, pos: tuple) -> bool:
        """Retorna True se a coordenada estiver fora da grade [0, grid_size - 1]."""
        r, c = pos
        return r < 0 or r >= self.grid_size or c < 0 or c >= self.grid_size

    def _is_dangerous(self, pos: tuple) -> bool:
        """
        Retorna True se pos é parede ou corpo da cobra.
        Para perigo imediato, a cauda é considerada livre (pois se moverá se a cobra não comer).
        """
        if self._out_of_bounds(pos):
            return True
        # Exclui a cauda da verificação de perigo imediato (ela se moverá no próximo passo se não comer)
        body = self.snake[:-1] if len(self.snake) > 1 else self.snake
        return pos in body

    def _place_food(self) -> tuple:
        """Sorteia uma posição livre na grade de forma uniforme."""
        snake_set = set(self.snake)
        free_cells = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
            if (r, c) not in snake_set
        ]
        if not free_cells:
            return (-1, -1)  # Grade totalmente preenchida (vitória)
        idx = self.rng.randint(0, len(free_cells))
        return free_cells[idx]

    def _get_obs(self) -> np.ndarray:
        """
        Constrói o vetor de observação O de 11 floats binários (0.0 ou 1.0):
            0: perigo à frente
            1: perigo à direita
            2: perigo à esquerda
            3: direção atual cima
            4: direção atual baixo
            5: direção atual esquerda
            6: direção atual direita
            7: comida à esquerda da cabeça (food.col < head.col)
            8: comida à direita da cabeça  (food.col > head.col)
            9: comida acima da cabeça     (food.row < head.row)
           10: comida abaixo da cabeça    (food.row > head.row)
        """
        head = self.snake[0]
        dr, dc = self.direction

        # Próximas posições relativas
        pos_frente  = (head[0] + dr,    head[1] + dc)
        pos_direita = (head[0] + dc,    head[1] - dr)
        pos_esq     = (head[0] - dc,    head[1] + dr)

        perigo_frente  = 1.0 if self._is_dangerous(pos_frente) else 0.0
        perigo_direita = 1.0 if self._is_dangerous(pos_direita) else 0.0
        perigo_esq     = 1.0 if self._is_dangerous(pos_esq) else 0.0

        # Direção atual (one-hot)
        dir_cima     = 1.0 if self.direction == (-1,  0) else 0.0
        dir_baixo    = 1.0 if self.direction == ( 1,  0) else 0.0
        dir_esquerda = 1.0 if self.direction == ( 0, -1) else 0.0
        dir_direita  = 1.0 if self.direction == ( 0,  1) else 0.0

        # Posição da comida em relação à cabeça (coordenadas absolutas da grade)
        food_esq    = 1.0 if self.food[1] < head[1] else 0.0
        food_dir    = 1.0 if self.food[1] > head[1] else 0.0
        food_acima  = 1.0 if self.food[0] < head[0] else 0.0
        food_abaixo = 1.0 if self.food[0] > head[0] else 0.0

        obs = np.array([
            perigo_frente,   # 0
            perigo_direita,  # 1
            perigo_esq,      # 2
            dir_cima,        # 3
            dir_baixo,       # 4
            dir_esquerda,    # 5
            dir_direita,     # 6
            food_esq,        # 7
            food_dir,        # 8
            food_acima,      # 9
            food_abaixo,     # 10
        ], dtype=np.float32)

        return obs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Teste rápido do ambiente SnakeEnv")
    parser.add_argument("--episodes", type=int, default=10, help="Quantidade de episódios de teste")
    parser.add_argument("--grid-size", type=int, default=10, help="Tamanho da grade")
    parser.add_argument("--seed", type=int, default=42, help="Semente do RNG")
    args = parser.parse_args()

    env = SnakeEnv(grid_size=args.grid_size, seed=args.seed)
    rng = np.random.RandomState(args.seed)

    pontuacoes = []
    passos_total = []

    print(f"Executando {args.episodes} episódios com política aleatória na grade {args.grid_size}×{args.grid_size}...")
    for ep in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        recompensa_acumulada = 0.0

        while not done:
            acao = rng.randint(0, 3)
            obs, reward, done, info = env.step(acao)
            recompensa_acumulada += reward

        pontuacoes.append(info["score"])
        passos_total.append(info["steps"])
        print(f"Episódio {ep:2d} | Pontuação: {info['score']} | Passos: {info['steps']} | "
              f"Recompensa: {recompensa_acumulada:.1f} | Truncado: {info['truncated']}")

    print("\n--- Estatísticas do Agente Aleatório ---")
    print(f"Pontuação média : {np.mean(pontuacoes):.2f} (máx: {np.max(pontuacoes)}, mín: {np.min(pontuacoes)})")
    print(f"Passos médios   : {np.mean(passos_total):.2f}")
    print("Teste concluído com sucesso sem erros!")
