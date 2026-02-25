# Meta-World Continual World Runner

Script:
`rl/rl_uniandes/drl/mujoco-drl/metaworld_crl_runner.py`

## Requisitos

```bash
./.venv/bin/python -m pip install metaworld stable-baselines3 gymnasium torch
```

## Problemas comunes

### 1) Warnings `obs ... is not within observation space`

En Meta-World (v3), algunas dimensiones del `observation_space` vienen con `low==high==0` aunque la observación real varía.

Solución en el runner:
- `--disable-env-checker` (por defecto `true`)
- `--relax-obs-bounds` (por defecto `true`)

### 2) Render congelado / kernel crash en notebook

Evita `render_mode='human'` con `gym.make_vec(...)` dentro de Jupyter.
Usa `rgb_array` para visualizar frames en notebook.

## Benchmark Continual World (CW10 / CW20)

### CW10 (recomendado para iteración rápida)

```bash
./.venv/bin/python rl/rl_uniandes/drl/mujoco-drl/metaworld_crl_runner.py \
  --mode continual \
  --algo sac \
  --task-preset cw10 \
  --steps-per-task 1000000 \
  --eval-episodes 10
```

### CW20 (CW10 repetido dos veces)

```bash
./.venv/bin/python rl/rl_uniandes/drl/mujoco-drl/metaworld_crl_runner.py \
  --mode continual \
  --algo sac \
  --task-preset cw20 \
  --steps-per-task 1000000 \
  --eval-episodes 10
```

Nota:
- El paper de Continual World usa SAC y normalmente `1M` steps por tarea.
- En el runner, `--reset-replay-every-task` está activo por defecto (alineado a esa configuración).

## Modo multitarea (referencia)

```bash
./.venv/bin/python rl/rl_uniandes/drl/mujoco-drl/metaworld_crl_runner.py \
  --mode multitask \
  --algo sac \
  --task-preset mt10 \
  --total-steps 1000000 \
  --eval-episodes 10
```

## Tareas personalizadas

```bash
./.venv/bin/python rl/rl_uniandes/drl/mujoco-drl/metaworld_crl_runner.py \
  --mode continual \
  --algo sac \
  --tasks "hammer-v3,push-wall-v3,peg-unplug-side-v3" \
  --steps-per-task 200000
```

## Salidas

Cada corrida crea un directorio en:

`rl/rl_uniandes/drl/mujoco-drl/logs/metaworld_cw/<run_name>/`

Archivos principales:
- `config.json`
- `train.monitor.csv`
- `eval.monitor.csv`
- `eval_metrics.csv`
- `summary.json`
- `model_final.zip`
