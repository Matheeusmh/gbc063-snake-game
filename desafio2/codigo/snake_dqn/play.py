"""
play.py
=======
Gera um GIF animado de uma partida jogada pelo agente DQN treinado (ε = 0).
Disciplina GBC063 - Inteligência Artificial (UFU).

Uso:
    python play.py --run-name g10
    python play.py --run-name g10 --seed 42 --fps 6
"""

import argparse
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

try:
    from PIL import Image
    TEM_PIL = True
except ImportError:
    TEM_PIL = False

from snake_env import SnakeEnv
from dqn import DQNAgent


def renderizar_frame(snake: list, food: tuple, grid_size: int, step: int, score: int):
    """Gera um frame visual da partida usando Matplotlib e retorna como imagem PIL."""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # Linha 0 no topo (coordenadas de matriz)

    # Fundo do tabuleiro
    ax.set_facecolor("#F8F9FA")

    # Desenha as linhas da grade
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color="#E0E0E0", linewidth=0.8)
        ax.axvline(i - 0.5, color="#E0E0E0", linewidth=0.8)

    # Desenha a comida (círculo vermelho)
    fr, fc = food
    comida_circ = patches.Circle((fc, fr), 0.40, color="#E53935", ec="#B71C1C", lw=1.5)
    ax.add_patch(comida_circ)

    # Desenha o corpo da cobra (retângulos verdes com cantos arredondados)
    for seg_idx, (r, c) in enumerate(snake[1:], start=1):
        seg_rect = patches.Rectangle((c - 0.42, r - 0.42), 0.84, 0.84,
                                     color="#43A047", ec="#2E7D32", lw=1.2)
        ax.add_patch(seg_rect)

    # Desenha a cabeça da cobra (verde escuro)
    hr, hc = snake[0]
    cabeca_rect = patches.Rectangle((hc - 0.45, hr - 0.45), 0.90, 0.90,
                                    color="#1B5E20", ec="#0D3812", lw=1.8)
    ax.add_patch(cabeca_rect)

    ax.set_title(f"Snake DQN  |  Passo: {step}  |  Pontuação: {score}",
                 fontsize=12, fontweight="bold", pad=10)
    ax.axis("off")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    if TEM_PIL:
        return Image.open(buf)
    else:
        return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Gera GIF de uma partida do Snake com DQN")
    parser.add_argument("--run-name", type=str, required=True, help="Nome da execução (subpasta em results/)")
    parser.add_argument("--seed", type=int, default=42, help="Semente da partida (padrão: 42)")
    parser.add_argument("--fps", type=int, default=6, help="Taxa de quadros por segundo no GIF (padrão: 6)")
    args = parser.parse_args()

    results_dir = os.path.join("results", args.run_name)
    config_file = os.path.join(results_dir, "config.json")
    model_file = os.path.join(results_dir, "model.pt")

    if not os.path.exists(results_dir) or not os.path.exists(model_file):
        print(f"[ERRO] Modelo não encontrado em {model_file}!")
        sys.exit(1)

    grid_size = 10
    hidden_size = 256
    double_dqn = False
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            grid_size = cfg.get("grid_size", 10)
            hidden_size = cfg.get("hidden_size", 256)
            double_dqn = cfg.get("double", False)

    # Inicializa ambiente e agente
    env = SnakeEnv(grid_size=grid_size, seed=args.seed)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        hidden_size=hidden_size,
        double_dqn=double_dqn,
        eps_start=0.0,
        eps_min=0.0,
    )
    agent.load(model_file)
    agent.epsilon = 0.0

    print(f"Gravando partida do modelo {args.run_name} na grade {grid_size}×{grid_size} (semente {args.seed})...")

    obs = env.reset()
    done = False
    frames = []

    # Frame inicial
    frame0 = renderizar_frame(env.snake, env.food, grid_size, env.steps, env.score)
    frames.append(frame0)

    while not done:
        action = agent.select_action(obs, training=False)
        obs, reward, done, info = env.step(action)
        frame = renderizar_frame(env.snake, env.food, grid_size, env.steps, env.score)
        frames.append(frame)

    gif_path = os.path.join(results_dir, "game.gif")

    if TEM_PIL:
        duration_ms = int(1000 / args.fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
        )
        print(f"[OK] GIF gerado com sucesso com {len(frames)} frames!")
        print(f"Salvo em: {os.path.abspath(gif_path)}")
        print(f"Pontuação final: {info['score']} | Passos: {info['steps']}")
    else:
        print("[AVISO] Pacote Pillow não instalado. Salvando frames como imagens PNG...")
        frames_dir = os.path.join(results_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for idx, fbytes in enumerate(frames):
            with open(os.path.join(frames_dir, f"frame_{idx:04d}.png"), "wb") as f:
                f.write(fbytes)
        print(f"Frames PNG salvos em: {frames_dir}")


if __name__ == "__main__":
    main()
