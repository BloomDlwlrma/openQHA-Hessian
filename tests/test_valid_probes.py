"""The per-structure fixed probe set: `--valid_probes_key` -> `Configuration.properties
["valid_probes"]` -> `AtomicData.valid_probes / has_valid_probes`, and how a batch carries
them (openQHA-Hessian, branch openqha-hessian, commit D).

The probes are DRAWN BY THE DATASET, not here (openQHA S0-C-67): a structure that carries
none gets None with weight 0, and a loss that wants them must say so itself. Their
DISTRIBUTION is the dataset's business too (openQHA draws PHL's standard normal, S0-C-68),
so the validation here checks the shape and that the numbers are finite, nothing else.
"""

import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data, tools
from mace.tools import torch_geometric
from mace.tools.default_keys import DefaultKeys

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 6, 8])
K_MAX = 16


def _atoms(n_atoms, seed, with_probes=True, k=K_MAX, length_ok=True, finite=True, declared=None):
    rng = np.random.default_rng(seed)
    symbols = ["C", "O", "H", "H", "H", "C", "H", "H"][:n_atoms]
    at = Atoms(symbols=symbols, positions=rng.standard_normal((n_atoms, 3)) * 1.3)
    at.info["REF_energy"] = float(rng.standard_normal())
    at.arrays["REF_forces"] = rng.standard_normal((n_atoms, 3))
    if with_probes:
        n3 = 3 * n_atoms
        v = rng.standard_normal((k, n3))           # openQHA S0-C-68 draws PHL's normal
        if not finite:
            v[0, 0] = np.nan
        flat = v.reshape(-1) if length_ok else v.reshape(-1)[:-1]
        at.info["REF_valid_probes"] = flat
        at.info["_V"] = v
    if declared is not None:
        at.info["has_valid_probes"] = declared
    return at


def test_default_keys_and_keyspec():
    assert DefaultKeys.VALID_PROBES.value == "REF_valid_probes"
    ks = data.KeySpecification.from_defaults()
    assert ks.info_keys["valid_probes"] == "REF_valid_probes"
    ks2 = data.utils.update_keyspec_from_kwargs(data.KeySpecification(), {"valid_probes_key": "my_probes"})
    assert ks2.info_keys["valid_probes"] == "my_probes"


def test_config_from_atoms_reads_validates_and_masks():
    ks = data.KeySpecification.from_defaults()
    with_v = data.config_from_atoms(_atoms(3, 0), key_specification=ks)
    assert with_v.properties["valid_probes"].shape == (K_MAX, 9)
    assert np.allclose(with_v.properties["valid_probes"], _atoms(3, 0).info["_V"])
    assert with_v.property_weights["valid_probes"] == 1.0

    # k is whatever the dataset stored: the field says nothing about how many a loss takes
    four = data.config_from_atoms(_atoms(3, 1, k=4), key_specification=ks)
    assert four.properties["valid_probes"].shape == (4, 9)

    without = data.config_from_atoms(_atoms(3, 2, with_probes=False), key_specification=ks)
    assert without.properties["valid_probes"] is None
    assert without.property_weights["valid_probes"] == 0.0

    declared_absent = data.config_from_atoms(_atoms(3, 3, declared=False), key_specification=ks)
    assert declared_absent.properties["valid_probes"] is None

    with pytest.raises(ValueError, match="not a multiple of 3N"):
        data.config_from_atoms(_atoms(3, 4, length_ok=False), key_specification=ks)
    with pytest.raises(ValueError, match="non-finite"):
        data.config_from_atoms(_atoms(3, 5, finite=False), key_specification=ks)


def test_atomic_data_fields_and_batching():
    ks = data.KeySpecification.from_defaults()
    frames = [_atoms(3, 10), _atoms(4, 11, with_probes=False), _atoms(2, 12)]
    ads = [
        data.AtomicData.from_config(data.config_from_atoms(a, key_specification=ks), z_table=TABLE, cutoff=5.0)
        for a in frames
    ]
    assert [int(ad.valid_probes.numel()) for ad in ads] == [K_MAX * 9, 0, K_MAX * 6]
    assert [bool(ad.has_valid_probes) for ad in ads] == [True, False, True]

    loader = torch_geometric.dataloader.DataLoader(dataset=ads, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert int(batch.valid_probes.numel()) == K_MAX * 9 + K_MAX * 6
    assert list(batch.has_valid_probes) == [True, False, True]

    # the rows come back in order, graph by graph: the slice a loss takes
    off, n_k = 0, [3, 4, 2]
    flat = batch.valid_probes.numpy()
    for g, n in enumerate(n_k):
        if not bool(batch.has_valid_probes[g]):
            continue
        size = K_MAX * 3 * n
        got = flat[off:off + size].reshape(K_MAX, 3 * n)
        assert np.allclose(got, frames[g].info["_V"])
        off += size
    assert off == flat.size


def test_the_default_path_is_untouched():
    """A dataset with no probe key loads exactly as before: empty tensors, no exception."""
    ks = data.KeySpecification.from_defaults()
    ad = data.AtomicData.from_config(
        data.config_from_atoms(_atoms(3, 20, with_probes=False), key_specification=ks),
        z_table=TABLE, cutoff=5.0)
    assert int(ad.valid_probes.numel()) == 0 and not bool(ad.has_valid_probes)
