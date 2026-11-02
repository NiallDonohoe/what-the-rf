import numpy as np
import pytest

from wtrf.data import augment, synth


@pytest.mark.parametrize("name", augment.ORDER)
def test_each_augmentation_keeps_shape_and_changes_signal(name, rng):
    x = synth.generate("QPSK", 16, 30.0, rng)
    cfg = augment.AugmentConfig(p=1.0)
    y = augment.apply(x, [name], cfg, rng)
    assert y.shape == x.shape and y.dtype == np.float32
    assert not np.allclose(x, y)


def test_cfo_preserves_power(rng):
    z = synth.clean("QPSK", 8, rng)
    out = augment.cfo(z, rng, augment.AugmentConfig())
    assert np.allclose(np.abs(out), np.abs(z))


def test_empty_list_is_identity_and_unknown_raises(rng):
    x = synth.generate("BPSK", 4, 10.0, rng)
    assert augment.apply(x, [], augment.AugmentConfig(), rng) is x
    with pytest.raises(ValueError):
        augment.apply(x, ["teleport"], augment.AugmentConfig(), rng)
