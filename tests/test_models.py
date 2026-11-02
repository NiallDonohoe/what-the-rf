import pytest
import torch

from wtrf.models.registry import MODELS, build_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_output_shape(name):
    model = build_model(name, n_classes=7, n_samples=128)
    assert model(torch.randn(5, 2, 128)).shape == (5, 7)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_one_step_lowers_loss_on_a_tiny_batch(name):
    torch.manual_seed(0)
    model = build_model(name, n_classes=3, n_samples=64)
    x, y = torch.randn(16, 2, 64), torch.randint(0, 3, (16,))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    before = loss_fn(model(x), y)
    for _ in range(5):
        opt.zero_grad()
        loss_fn(model(x), y).backward()
        opt.step()
    assert loss_fn(model(x), y).item() < before.item()
