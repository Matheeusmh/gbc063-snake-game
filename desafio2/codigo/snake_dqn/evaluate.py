"""
evaluate.py
===========
Avaliação comparativa entre 3 agentes no Snake:
1. DQN Treinado (carregado de model.pt com ε = 0)
2. Agente Aleatório (escolha uniforme entre as 3 ações)
3. Agente Guloso Simples (heurística com base nos 11 floats da observação)

Gera:
- Tabela de estatísticas no terminal (média, desvio, mín, máx)
- CSV results/<run_name>/eval.csv
- Gráficos para apresentação:
    * Slide 4: results/<run_name>/training_curves.png (Recompensa e Pontuação com médias móveis)
    * Slide 4 (extra): results/<run_name>/training_extra.png (Epsilon e Loss)
    * Slide 5: results/<run_name>/eval_comparison.png (Boxplot comparativo dos 3 agentes)

Disciplina GBC063 - Inteligência Artificial (UFU).
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # Renderização headless sem interface gráfica
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from snake_env import SnakeEnv
from dqn import DQNAgent


# =============================================================================
# 1. Agente Guloso Simples (Heurística baseada nos 11 floats de observação)
# =============================================================================

def greedy_action(obs: np.ndarray, rng: np.random.RandomState) -> int:
    """
    Decisão heurística gulosa a partir do vetor de 11 floats:
      obs[0]: perigo reto
      obs[1]: perigo direita
      obs[2]: perigo esquerda
      obs[3]: dir_cima
      obs[4]: dir_baixo
      obs[5]: dir_esquerda
      obs[6]: dir_direita
      obs[7]: comida_esquerda
      obs[8]: comida_direita
      obs[9]: comida_acima
      obs[10]: comida_abaixo

    Regras:
      1. Descarta ações que levam a perigo imediato (obs[a] == 1.0).
      2. Se nenhuma for segura, escolhe aleatoriamente entre as 3.
      3. Entre as seguras, prefere a que move em direção à comida.
    """
    perigos = [obs[0], obs[1], obs[2]]
    acoes_seguras = [a for a in [0, 1, 2] if perigos[a] == 0.0]

    # Se nenhuma ação for segura, sorteia qualquer uma
    if not acoes_seguras:
        return int(rng.randint(0, 3))

    # Se apenas uma ação for segura, executa-a diretamente
    if len(acoes_seguras) == 1:
        return acoes_seguras[0]

    # Determina a direção resultante para cada ação:
    # Ação 0 (reto): mantém direção atual
    # Ação 1 (direita): cima->dir, baixo->esq, esq->cima, dir->baixo
    # Ação 2 (esquerda): cima->esq, baixo->dir, esq->baixo, dir->cima
    dir_cima     = bool(obs[3])
    dir_baixo    = bool(obs[4])
    dir_esquerda = bool(obs[5])
    dir_direita  = bool(obs[6])

    food_esq    = bool(obs[7])
    food_dir    = bool(obs[8])
    food_acima  = bool(obs[9])
    food_abaixo = bool(obs[10])

    def score_acao(acao: int) -> int:
        pontos = 0
        if acao == 0:  # Reto
            if dir_cima and food_acima: pontos += 1
            if dir_baixo and food_abaixo: pontos += 1
            if dir_esquerda and food_esq: pontos += 1
            if dir_direita and food_dir: pontos += 1
        elif acao == 1:  # Virar à direita (horário)
            if dir_cima and food_dir: pontos += 1
            if dir_baixo and food_esq: pontos += 1
            if dir_esquerda and food_acima: pontos += 1
            if dir_direita and food_abaixo: pontos += 1
        elif acao == 2:  # Virar à esquerda (anti-horário)
            if dir_cima and food_esq: pontos += 1
            if dir_baixo and food_dir: pontos += 1
            if dir_esquerda and food_abaixo: pontos += 1
            if dir_direita and food_acima: pontos += 1
        return pontos

    # Encontra a melhor ação segura
    scores = [(score_acao(a), a) for a in acoes_seguras]
    max_score = max(s[0] for s in scores)
    melhores = [a for s, a in scores if s == max_score]

    return int(melhores[rng.randint(0, len(melhores))])


# =============================================================================
# 2. Geração dos Gráficos de Treino (Slide 4)
# =============================================================================

def gerar_graficos_treino(results_dir: str):
    """
    Lê train_log.csv e gera:
    1. training_curves.png (Recompensa e Pontuação com média móvel 100)
    2. training_extra.png (Decaimento do Epsilon e Evolução do Loss)
    """
    log_file = os.path.join(results_dir, "train_log.csv")
    if not os.path.exists(log_file):
        print(f"[AVISO] {log_file} não encontrado. Gráficos de treino ignorados.")
        return

    df = pd.read_csv(log_file)
    if len(df) == 0:
        return

    # Janela de média móvel (100 ou o total de linhas se menor)
    janela = min(100, len(df))
    df["reward_ma"] = df["total_reward"].rolling(window=janela, min_periods=1).mean()
    df["score_ma"] = df["score"].rolling(window=janela, min_periods=1).mean()

    # --- Gráfico Principal: training_curves.png (Slide 4) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=200, facecolor="white")

    # Subgráfico 1: Recompensa Total
    ax1.scatter(df["episode"], df["total_reward"], color="#4A90E2", alpha=0.20, s=8, label="Recompensa por ep.")
    ax1.plot(df["episode"], df["reward_ma"], color="#0A3D91", linewidth=2.2, label=f"Média móvel ({janela} ep.)")
    ax1.set_title("Recompensa Total por Episódio", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Episódio", fontsize=11)
    ax1.set_ylabel("Recompensa Total", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="lower right", frameon=True, fontsize=10)

    # Subgráfico 2: Pontuação (Maçãs)
    ax2.scatter(df["episode"], df["score"], color="#FF8C00", alpha=0.25, s=8, label="Pontuação por ep.")
    ax2.plot(df["episode"], df["score_ma"], color="#B84500", linewidth=2.2, label=f"Média móvel ({janela} ep.)")
    ax2.set_title("Pontuação (Maçãs Comidas) por Episódio", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Episódio", fontsize=11)
    ax2.set_ylabel("Maçãs Comidas", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", frameon=True, fontsize=10)

    plt.tight_layout()
    curves_path = os.path.join(results_dir, "training_curves.png")
    fig.savefig(curves_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico salvo: {curves_path}")

    # --- Gráfico Extra: training_extra.png (Slide 4 extra) ---
    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5), dpi=200, facecolor="white")

    # Epsilon
    ax3.plot(df["episode"], df["epsilon"], color="#2E7D32", linewidth=2.0)
    ax3.set_title("Decaimento da Taxa de Exploração (ε)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Episódio", fontsize=11)
    ax3.set_ylabel("Valor de ε", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.35)

    # Loss Médio
    loss_valid = df[df["mean_loss"] > 0]
    if len(loss_valid) > 0:
        ax4.plot(loss_valid["episode"], loss_valid["mean_loss"], color="#D32F2F", alpha=0.5, label="Loss médio por ep.")
        loss_ma = loss_valid["mean_loss"].rolling(window=janela, min_periods=1).mean()
        ax4.plot(loss_valid["episode"], loss_ma, color="#8B0000", linewidth=2.0, label=f"Média móvel ({janela} ep.)")
        ax4.legend(loc="upper right", frameon=True, fontsize=10)
    ax4.set_title("Função de Perda Huber (Loss)", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Episódio", fontsize=11)
    ax4.set_ylabel("Perda Média", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    extra_path = os.path.join(results_dir, "training_extra.png")
    fig2.savefig(extra_path, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f"[OK] Gráfico salvo: {extra_path}")


# =============================================================================
# 3. Gráfico de Comparação de Avaliação (Slide 5)
# =============================================================================

def gerar_grafico_avaliacao(eval_df: pd.DataFrame, grid_size: int, episodes: int, output_path: str):
    """
    Gera o boxplot comparativo dos 3 agentes (DQN, Guloso, Aleatório)
    com a média destacada por marcadores em forma de diamante ('D').
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200, facecolor="white")

    agentes = ["DQN", "Guloso", "Aleatório"]
    dados = [eval_df[eval_df["agente"] == ag]["score"].values for ag in agentes]

    # Criação do Boxplot estilizado
    bp = ax.boxplot(
        dados,
        patch_artist=True,
        labels=agentes,
        showmeans=True,
        meanline=False,
        meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="#FFD700", markersize=8, label="Média"),
        medianprops=dict(color="#B71C1C", linewidth=2),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
    )

    # Cores personalizadas para as caixas
    cores = ["#4A90E2", "#50E3C2", "#B8E986"]
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)

    ax.set_title(f"Avaliação de Desempenho — Grade {grid_size}×{grid_size} ({episodes} Episódios)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Agente", fontsize=12, labelpad=8)
    ax.set_ylabel("Pontuação (Maçãs Comidas)", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax.legend(handles=[bp["means"][0]], loc="upper right", frameon=True, fontsize=11)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Gráfico salvo: {output_path}")


# =============================================================================
# 4. Loop Principal de Avaliação
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Avaliação comparativa dos agentes no Snake")
    parser.add_argument("--run-name", type=str, required=True, help="Nome do experimento (subpasta em results/)")
    parser.add_argument("--episodes", type=int, default=100, help="Quantidade de episódios de teste (padrão: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Semente base para avaliação (padrão: 42)")
    args = parser.parse_args()

    results_dir = os.path.join("results", args.run_name)
    if not os.path.exists(results_dir):
        print(f"[ERRO] Diretório de resultados não encontrado: {results_dir}")
        sys.exit(1)

    # 1. Gera os gráficos de treino a partir do log
    gerar_graficos_treino(results_dir)

    # 2. Carrega hiperparâmetros salvos
    config_file = os.path.join(results_dir, "config.json")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {"grid_size": 10}

    grid_size = config.get("grid_size", 10)
    hidden_size = config.get("hidden_size", 256)
    double_dqn = config.get("double", False)

    # 3. Carrega o modelo DQN treinado
    model_path = os.path.join(results_dir, "model.pt")
    if not os.path.exists(model_path):
        print(f"[ERRO] Arquivo de pesos {model_path} não encontrado!")
        sys.exit(1)

    dqn_agent = DQNAgent(
        state_size=11,
        action_size=3,
        hidden_size=hidden_size,
        double_dqn=double_dqn,
        eps_start=0.0,
        eps_min=0.0,
    )
    dqn_agent.load(model_path)
    dqn_agent.epsilon = 0.0  # Garante política 100% gulosa na avaliação

    print(f"\n=======================================================")
    print(f" Avaliação — {args.episodes} episódios (Grade {grid_size}×{grid_size})")
    print(f" Run: {args.run_name} | Semente base: {args.seed}")
    print(f"=======================================================")

    eval_rows = []

    # Executa os 100 episódios para cada um dos 3 agentes com as MESMAS sementes
    for ep in range(args.episodes):
        ep_seed = args.seed + ep

        # --- A) DQN Treinado ---
        env_dqn = SnakeEnv(grid_size=grid_size, seed=ep_seed)
        obs = env_dqn.reset()
        done = False
        while not done:
            action = dqn_agent.select_action(obs, training=False)
            obs, _, done, info = env_dqn.step(action)
        eval_rows.append({"agente": "DQN", "episodio": ep + 1, "score": info["score"], "steps": info["steps"]})

        # --- B) Agente Aleatório ---
        env_rnd = SnakeEnv(grid_size=grid_size, seed=ep_seed)
        rng_rnd = np.random.RandomState(ep_seed)
        obs = env_rnd.reset()
        done = False
        while not done:
            action = int(rng_rnd.randint(0, 3))
            obs, _, done, info = env_rnd.step(action)
        eval_rows.append({"agente": "Aleatório", "episodio": ep + 1, "score": info["score"], "steps": info["steps"]})

        # --- C) Agente Guloso Simples ---
        env_grd = SnakeEnv(grid_size=grid_size, seed=ep_seed)
        rng_grd = np.random.RandomState(ep_seed)
        obs = env_grd.reset()
        done = False
        while not done:
            action = greedy_action(obs, rng_grd)
            obs, _, done, info = env_grd.step(action)
        eval_rows.append({"agente": "Guloso", "episodio": ep + 1, "score": info["score"], "steps": info["steps"]})

    # Salva eval.csv
    eval_df = pd.DataFrame(eval_rows)
    eval_csv_path = os.path.join(results_dir, "eval.csv")
    eval_df.to_csv(eval_csv_path, index=False, encoding="utf-8")
    print(f"[OK] Resultados salvos em: {eval_csv_path}")

    # Imprime tabela estatística formatada
    print("\n" + "=" * 54)
    print(f" Avaliação — {args.episodes} episódios (grade {grid_size}×{grid_size})")
    print("=" * 54)
    print(f"{'Agente':<13} | {'Média':>7} | {'Desvio':>7} | {'Mín':>5} | {'Máx':>5}")
    print("-" * 13 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 5 + "-+-" + "-" * 5)

    for ag in ["DQN", "Guloso", "Aleatório"]:
        sub = eval_df[eval_df["agente"] == ag]["score"]
        media = sub.mean()
        desvio = sub.std()
        minimo = int(sub.min())
        maximo = int(sub.max())
        print(f"{ag:<13} | {media:7.2f} | {desvio:7.2f} | {minimo:5d} | {maximo:5d}")
    print("=" * 54 + "\n")

    # Gera gráfico do Slide 5
    eval_plot_path = os.path.join(results_dir, "eval_comparison.png")
    gerar_grafico_avaliacao(eval_df, grid_size, args.episodes, eval_plot_path)


if __name__ == "__main__":
    main()
