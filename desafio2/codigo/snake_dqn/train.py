"""
train.py
========
Loop de treinamento do agente DQN para o ambiente Snake com registro em CSV e salvamento de modelos.
Disciplina GBC063 - Inteligência Artificial (UFU).

Uso:
    python train.py --grid-size 10 --episodes 2000 --run-name g10 --seed 0
    python train.py --smoke-test --run-name smoke
"""

import argparse
import csv
import json
import os
import random
import time
from collections import deque

import numpy as np
import torch

from snake_env import SnakeEnv
from dqn import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Treinamento de DQN no jogo Snake")
    parser.add_argument("--grid-size", type=int, default=10, help="Tamanho da grade (padrão: 10)")
    parser.add_argument("--episodes", type=int, default=2000, help="Total de episódios de treino (padrão: 2000)")
    parser.add_argument("--gamma", type=float, default=0.95, help="Fator de desconto gamma (padrão: 0.95)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado Adam (padrão: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamanho do mini-batch (padrão: 64)")
    parser.add_argument("--eps-decay", type=float, default=0.995, help="Decaimento multiplicativo de epsilon (padrão: 0.995)")
    parser.add_argument("--target-update", type=int, default=500, help="Frequência em passos para atualizar target net (padrão: 500)")
    parser.add_argument("--seed", type=int, default=0, help="Semente para reprodutibilidade (padrão: 0)")
    parser.add_argument("--run-name", type=str, default="run", help="Nome da execução / pasta em results/ (padrão: run)")
    parser.add_argument("--double", action="store_true", default=False, help="Usar Double DQN (padrão: False)")
    parser.add_argument("--smoke-test", action="store_true", default=False, help="Executar modo de teste rápido com 20 episódios")
    parser.add_argument("--hidden-size", type=int, default=256, help="Número de neurônios na camada oculta (padrão: 256)")
    return parser.parse_args()


def set_seed(seed: int):
    """Garante reprodutibilidade fixando sementes em random, numpy e torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    if args.smoke_test:
        args.episodes = 20
        print("[INFO] Modo smoke-test ativado: limitando a 20 episódios.")

    # Fixa sementes
    set_seed(args.seed)

    # Diretório de saída
    results_dir = os.path.join("results", args.run_name)
    os.makedirs(results_dir, exist_ok=True)

    # Salva configurações em config.json
    config_dict = {
        "grid_size": args.grid_size,
        "episodes": args.episodes,
        "gamma": args.gamma,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eps_decay": args.eps_decay,
        "target_update": args.target_update,
        "seed": args.seed,
        "run_name": args.run_name,
        "double": args.double,
        "hidden_size": args.hidden_size,
        "smoke_test": args.smoke_test,
    }
    config_path = os.path.join(results_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=4, ensure_ascii=False)

    print(f"==================================================")
    print(f" Treinamento Snake DQN — Run: {args.run_name}")
    print(f" Grade: {args.grid_size}×{args.grid_size} | Episódios: {args.episodes}")
    print(f" Gamma: {args.gamma} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f" Epsilon Decay: {args.eps_decay} | Double DQN: {args.double}")
    print(f" Semente: {args.seed} | Resultados em: {results_dir}")
    print(f"==================================================")

    # Inicializa ambiente e agente
    env = SnakeEnv(grid_size=args.grid_size, seed=args.seed)
    agent = DQNAgent(
        state_size=11,
        action_size=3,
        hidden_size=args.hidden_size,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=100_000,
        target_update=args.target_update,
        eps_start=1.0,
        eps_min=0.01,
        eps_decay=args.eps_decay,
        double_dqn=args.double,
        warmup=1000 if not args.smoke_test else 50,  # Warmup reduzido no smoke test
    )

    # Configura arquivo de log CSV
    log_path = os.path.join(results_dir, "train_log.csv")
    csv_header = ["episode", "total_reward", "score", "steps", "epsilon", "mean_loss", "truncated"]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    # Histórico de métricas para médias móveis
    recent_rewards = deque(maxlen=100)
    recent_scores = deque(maxlen=100)
    best_mean_score = -float("inf")

    model_last_path = os.path.join(results_dir, "model.pt")
    model_best_path = os.path.join(results_dir, "model_best.pt")

    # Barra de progresso com fallback simples
    try:
        from tqdm import tqdm
        iterator = tqdm(range(1, args.episodes + 1), desc=f"Treino [{args.run_name}]")
    except ImportError:
        iterator = range(1, args.episodes + 1)

    start_time = time.time()

    for ep in iterator:
        obs = env.reset()
        done = False
        total_reward = 0.0
        losses_ep = []

        while not done:
            # Seleção de ação ε-greedy
            action = agent.select_action(obs, training=True)

            # Executa passo no ambiente
            next_obs, reward, done, info = env.step(action)

            # ATENÇÃO CRÍTICA: done_terminal só é True se o episódio terminou por colisão ou vitória,
            # NUNCA por timeout/truncamento!
            done_terminal = done and (not info.get("truncated", False))

            # Guarda transição no replay buffer
            agent.push_transition(obs, action, reward, next_obs, done_terminal)

            # Atualização dos pesos da rede
            loss = agent.update()
            if loss is not None:
                losses_ep.append(loss)

            obs = next_obs
            total_reward += reward

        # Decaimento do epsilon ao final de cada episódio
        agent.decay_epsilon()

        # Métricas do episódio
        ep_score = info.get("score", 0)
        ep_steps = info.get("steps", 0)
        ep_truncated = int(info.get("truncated", False))
        ep_mean_loss = float(np.mean(losses_ep)) if losses_ep else 0.0

        recent_rewards.append(total_reward)
        recent_scores.append(ep_score)

        # Salva linha no CSV
        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                round(total_reward, 4),
                ep_score,
                ep_steps,
                round(agent.epsilon, 6),
                round(ep_mean_loss, 6),
                ep_truncated,
            ])

        # Salva o modelo mais recente
        agent.save(model_last_path)

        # Salva o melhor modelo baseado na média móvel de 100 episódios da pontuação
        mean_score = float(np.mean(recent_scores))
        if mean_score > best_mean_score:
            best_mean_score = mean_score
            agent.save(model_best_path)

        # Impressão de status a cada 50 episódios (ou a cada 5 em smoke test)
        print_interval = 50 if not args.smoke_test else 5
        if ep % print_interval == 0:
            mean_reward = float(np.mean(recent_rewards))
            msg = (f"[Ep {ep:4d}/{args.episodes}] "
                   f"Recomp. média: {mean_reward:6.1f} | "
                   f"Pontuação média: {mean_score:4.2f} | "
                   f"ε: {agent.epsilon:5.3f}")
            if hasattr(iterator, "write"):
                iterator.write(msg)
            else:
                print(msg)

    tempo_total = time.time() - start_time
    print(f"\nTreino concluído em {tempo_total:.1f}s. Resultados em {results_dir}/")
    print(f"Melhor pontuação média observada: {best_mean_score:.2f}")

    if args.smoke_test:
        print("Smoke test OK")


if __name__ == "__main__":
    main()
