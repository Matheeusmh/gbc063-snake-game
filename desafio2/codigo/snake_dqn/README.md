# Snake DQN — GBC063 Inteligência Artificial (UFU)

Implementação de Deep Q-Network (DQN) para o jogo da cobra, sem dependências de Gym ou stable-baselines. Projeto para a disciplina GBC063 — Aprendizado por Reforço.

---

## Instalação

```bash
pip install -r requirements.txt
```

---

## Sequência de Comandos

```bash
# Treinar na grade 10×10
py train.py --grid-size 10 --episodes 2000 --run-name g10 --seed 0

# Avaliar e gerar gráficos para g10
py evaluate.py --run-name g10

# Treinar na grade 20×20
py train.py --grid-size 20 --episodes 3000 --run-name g20 --seed 0

# Avaliar e gerar gráficos para g20
py evaluate.py --run-name g20

# Comparar as duas execuções
py compare.py --runs g10 g20

# (Opcional) Gerar GIF de uma partida
py play.py --run-name g10

# Teste rápido (20 episódios)
py train.py --smoke-test --run-name smoke
```

---

## Onde Ficam os Arquivos Gerados

```
results/
├── g10/
│   ├── train_log.csv       ← log de treino (1 linha por episódio)
│   ├── eval.csv            ← resultados da avaliação
│   ├── config.json         ← hiperparâmetros
│   ├── model.pt            ← último modelo
│   ├── model_best.pt       ← melhor modelo (pela pontuação)
│   ├── training_curves.png ← Slide 4: curvas de aprendizado
│   ├── training_extra.png  ← Slide 4 extra: ε e loss
│   └── eval_comparison.png ← Slide 5: boxplot comparativo
└── g20/
    └── (mesma estrutura)
comparison.png              ← gerado por compare.py
```

---

## PNGs para os Slides

- **Slide 4**: `results/g10/training_curves.png` — recompensa e pontuação por episódio
- **Slide 5**: `results/g10/eval_comparison.png` — boxplot DQN vs Guloso vs Aleatório
- **Comparação**: `comparison.png` — DQN 10×10 vs 20×20

---

## Como Ler os Gráficos

- Os **pontos finos** são os valores brutos por episódio (ruidosos, mostram a variância natural do treino).
- A **linha forte** é a média móvel de 100 episódios, que revela a tendência real de aprendizado.
- O **ε (epsilon)** decresce ao longo do treino: o agente explora menos e explota mais as ações aprendidas.
- O **boxplot** mostra a distribuição de pontuações em 100 episódios de avaliação sem exploração.
- Uma **caixa (IQR) mais alta e mediana mais alta** indica um agente mais consistente e bem treinado.

---

## Se o Agente Não Aprender

| Sintoma | Solução sugerida |
| --------- | ----------------- |
| Curva plana até o final | Aumentar `--episodes` (ex: 5000 para 10×10) |
| Agente explora pouco | Reduzir `--eps-decay` (ex: `0.990`) para explorar mais tempo |
| Recompensas muito negativas logo de início | Checar a observação com `python snake_env.py --episodes 5` |
| Treino instável ou divergindo | Tentar `--double` para Double DQN |
| Grade 20×20 não converge | Reduzir `--grid-size` para `10` primeiro |

---

## Estrutura do Projeto

| Arquivo | Descrição |
| --------- | ----------- |
| `snake_env.py` | Ambiente do jogo da cobra com observação de 11 features |
| `dqn.py` | Rede neural DQNNetwork, ReplayBuffer e agente DQNAgent |
| `train.py` | Loop de treinamento com logging CSV e salvamento de modelos |
| `evaluate.py` | Avaliação do agente treinado vs. aleatório vs. guloso; gera gráficos |
| `compare.py` | Compara métricas de duas ou mais execuções (ex: g10 vs g20) |
| `play.py` | Roda uma partida e (opcionalmente) salva GIF animado |
| `requirements.txt` | Dependências Python do projeto |
| `README.md` | Este arquivo — guia de uso rápido |
| `DOCUMENTACAO.md` | Documentação técnica completa do projeto |
