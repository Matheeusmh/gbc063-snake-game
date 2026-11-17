"""
compare.py
==========
Compara múltiplas execuções do Snake DQN (ex: 10x10 vs 20x20, gamma 0.9 vs 0.99).
Gera:
- comparison.png com as curvas de média móvel de pontuação sobrepostas.
- Tabela resumo comparando a pontuação na avaliação e no treino.

Uso:
    python compare.py --runs g10 g20
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Paleta de cores nítidas e acessíveis
CORES = [
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"
]


def carregar_dados_run(run_name: str):
    """Carrega config.json, train_log.csv e eval.csv para uma execução."""
    run_dir = os.path.join("results", run_name)
    if not os.path.exists(run_dir):
        print(f"[AVISO] Pasta não encontrada: {run_dir}")
        return None

    # Config
    config_file = os.path.join(run_dir, "config.json")
    config = {}
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)

    # Train log
    train_file = os.path.join(run_dir, "train_log.csv")
    df_train = None
    if os.path.exists(train_file):
        df_train = pd.read_csv(train_file)

    # Eval log
    eval_file = os.path.join(run_dir, "eval.csv")
    df_eval = None
    if os.path.exists(eval_file):
        df_eval = pd.read_csv(eval_file)

    return {
        "name": run_name,
        "config": config,
        "train": df_train,
        "eval": df_eval,
    }


def main():
    parser = argparse.ArgumentParser(description="Comparação de múltiplas execuções do Snake DQN")
    parser.add_argument("--runs", nargs="+", required=True, help="Lista de nomes das execuções (ex: g10 g20)")
    parser.add_argument("--output", type=str, default="comparison.png", help="Nome do arquivo de saída (padrão: comparison.png)")
    args = parser.parse_args()

    dados_runs = []
    for r in args.runs:
        d = carregar_dados_run(r)
        if d is not None and d["train"] is not None and len(d["train"]) > 0:
            dados_runs.append(d)

    if not dados_runs:
        print("[ERRO] Nenhuma execução válida com train_log.csv foi encontrada!")
        sys.exit(1)

    # --- 1. Gráfico comparison.png ---
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200, facecolor="white")

    for i, d in enumerate(dados_runs):
        df = d["train"]
        grid = d["config"].get("grid_size", "?")
        label = f"{d['name']} (grade {grid}×{grid})"

        janela = min(100, len(df))
        ma_score = df["score"].rolling(window=janela, min_periods=1).mean()

        cor = CORES[i % len(CORES)]
        ax.plot(df["episode"], ma_score, label=label, color=cor, linewidth=2.2)

    ax.set_title("Comparação de Execuções — Média Móvel da Pontuação (janela = 100)",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Episódio", fontsize=12, labelpad=8)
    ax.set_ylabel("Pontuação Média (Maçãs)", fontsize=12, labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper left", frameon=True, fontsize=11)

    plt.tight_layout()
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] Gráfico comparativo salvo em: {os.path.abspath(args.output)}")

    # --- 2. Tabela Resumo no Terminal ---
    print("\n" + "=" * 65)
    print(f"{'Comparação de Execuções':^65}")
    print("=" * 65)
    print(f"{'Run Name':<12} | {'Grade':<7} | {'Ep. Treino':>10} | {'Pontuação DQN (avg ± std)':>25}")
    print("-" * 12 + "-+-" + "-" * 7 + "-+-" + "-" * 10 + "-+-" + "-" * 25)

    for d in dados_runs:
        grid = f"{d['config'].get('grid_size', '?')}×{d['config'].get('grid_size', '?')}"
        ep_treino = len(d["train"])

        if d["eval"] is not None:
            sub = d["eval"][d["eval"]["agente"] == "DQN"]["score"]
            if len(sub) > 0:
                score_str = f"{sub.mean():6.2f} ± {sub.std():.2f}"
            else:
                score_str = "Sem dados DQN"
        else:
            # Fallback: média dos últimos 100 episódios do treino
            ultimos = d["train"]["score"].tail(100)
            score_str = f"{ultimos.mean():6.2f} ± {ultimos.std():.2f} (treino)"

        print(f"{d['name']:<12} | {grid:<7} | {ep_treino:>10d} | {score_str:>25}")

    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
