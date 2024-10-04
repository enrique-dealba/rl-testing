from kron_torch import Kron


def get_kron_optimizer(params, learning_rate, weight_decay, memory_saving=False):
    return Kron(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
        max_size_triangular=0 if memory_saving else float("inf"),
    )
