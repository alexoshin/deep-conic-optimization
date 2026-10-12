"""Neural network implementations."""

import torch
from torch import Tensor, nn


class MLP(nn.Module):
    """Multilayer perceptron feedforward network."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        activation_fn: nn.Module = nn.ReLU(),
        bias: bool = True,
        zero_last_layer: bool = False,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size, bias=bias))
            layers.append(activation_fn)
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size, bias=bias))
        if zero_last_layer:
            layers[-1].weight.data.zero_()
            if bias:
                layers[-1].bias.data.zero_()
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
