from src.optimizers.soap_optimizer_module import SOAP


def get_soap_optimizer(params, learning_rate, weight_decay):
    return SOAP(
        params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
