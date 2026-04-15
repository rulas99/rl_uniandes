# MORPHIN Thesis Final Scope

Este directorio fue podado para dejar solo el pipeline que respalda la corrida final reportada en la tesis y sus metricas/figuras.

## Se conserva

- Codigo base de entrenamiento y entorno:
  `train_continual.py`, `run_experiments_morphin.sh`, `analysis_metrics.py`,
  `aggregate_morphin_results.py`, `combine_morphin_analyses.py`,
  `build_scratch_refs.py`, `thesis_stats.py`, `thesis_plots.py`,
  `thesis_env_plots.py`, `agents/`, `adapt/`, `envs/`, `gridworld_env/`.
- Pipeline final por shards:
  `run_full_256_shard.sh`, `run_full_256_shard1.sh`,
  `run_full_256_shard2.sh`, `run_full_256_shard3.sh`,
  `combine_full_256_shards.sh`.
- Artefactos finales de tesis:
  `logs/morphin_gridworld/thesis_9x9_full_256_combined/session_20260323_091255`.
- Referencias scratch usadas por los shards finales:
  `logs/morphin_gridworld/thesis_9x9_full_256_parallel/session_20260318_211023/shared_scratch_refs.json`.

## Repetir la corrida final

Desde este directorio:

```bash
bash run_full_256_shard1.sh
bash run_full_256_shard2.sh
bash run_full_256_shard3.sh
bash combine_full_256_shards.sh
```

Los shards quedan en modo `lean`, o sea: guardan solo los artefactos necesarios para recomputar metricas y figuras finales, evitando `update_metrics.csv`, checkpoints finales y rollouts canonicos.

## Si necesitas regenerar scratch refs desde cero

1. Corre `run_experiments_morphin.sh` con `RUN_PROFILE=scratch`.
2. Usa `build_scratch_refs.py` sobre el arbol `runs/` resultante.
3. Exporta `SHARED_SCRATCH_REFS_JSON=/ruta/al/json` antes de lanzar los shards.

## Figuras y tablas de tesis

La tesis enlaza directamente a:

- `logs/morphin_gridworld/thesis_9x9_full_256_combined/session_20260323_091255/analysis/thesis/figures`
- `logs/morphin_gridworld/thesis_9x9_full_256_combined/session_20260323_091255/analysis/thesis/tables`
