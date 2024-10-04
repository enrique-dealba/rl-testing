# RL Testing

Build Docker image:
```sh
docker build -t rl-project .
```

Run Docker container diagnostics:
```sh
docker run --gpus all rl-project:latest diagnostics
```

Run RL training:
```sh
docker run --gpus all -e WANDB_API_KEY=your_wandb_api_key rl-project train \
    --wandb_project your_project_name \
    --wandb_entity your_entity_name \
    --exp_name your_experiment_name \
    --env_id MsPacman-v5 \
    --optimizer adam \
    --learning_rate 0.00025 \
    --weight_decay 0.0001 \
    --total_timesteps 10000000 \
    --anneal_lr

```
