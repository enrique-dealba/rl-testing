import torch.optim as optim


def get_adam_optimizer(params, learning_rate, weight_decay):
    return optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, eps=1e-5)
