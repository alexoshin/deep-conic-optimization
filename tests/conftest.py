import torch

torch.set_default_dtype(torch.float64)


def pytest_configure(config):
    torch.manual_seed(0)
