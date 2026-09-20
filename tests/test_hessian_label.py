"""The per-structure Hessian label: `--hessian_key` -> `Configuration.properties["hessian"]`
-> `AtomicData.hessian / has_hessian / hessian_weight / sqrt_masses`, and how a batch
carries them (openQHA-Hessian, branch openqha-hessian, commit A)."""

import ase.data
import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from mace import data, tools
from mace.tools import torch_geometric
from mace.tools.default_keys import DefaultKeys

torch.set_default_dtype(torch.float64)

TABLE = tools.AtomicNumberTable([1, 6, 8])


def _atoms(n_atoms, seed, with_hessian=True, symmetric=True, length_ok=True, declared=None):
    rng = np.random.default_rng(seed)
    symbols = ["C", "O", "H", "H", "H", "C", "H", "H"][:n_atoms]
    at = Atoms(symbols=symbols, positions=rng.standard_normal((n_atoms, 3)) * 1.3)
    at.info["REF_energy"] = float(rng.standard_normal())
    at.arrays["REF_forces"] = rng.standard_normal((n_atoms, 3))
    if with_hessian:
        n3 = 3 * n_atoms
        h = rng.standard_normal((n3, n3))
        h = h + h.T if symmetric else h
        flat = h.reshape(-1) if length_ok else h.reshape(-1)[:-1]
        at.info["REF_hessian"] = flat
        at.info["_H"] = h
    if declared is not None:
        at.info["has_hessian"] = declared
    return at


def test_default_keys_and_keyspec():
    assert DefaultKeys.HESSIAN.value == "REF_hessian"
    ks = data.KeySpecification.from_defaults()
    assert ks.info_keys["hessian"] == "REF_hessian"
    ks2 = data.utils.update_keyspec_from_kwargs(data.KeySpecification(), {"hessian_key": "my_hess"})
    assert ks2.info_keys["hessian"] == "my_hess"


def test_config_from_atoms_reads_validates_and_masks():
    ks = data.KeySpecification.from_defaults()
    with_h = data.config_from_atoms(_atoms(3, 0), key_specification=ks)
    assert with_h.properties["hessian"].shape == (9, 9)
    assert np.allclose(with_h.properties["hessian"], _atoms(3, 0).info["_H"])
    assert with_h.property_weights["hessian"] == 1.0

    without = data.config_from_atoms(_atoms(3, 1, with_hessian=False), key_specification=ks)
    assert without.properties["hessian"] is None
    assert without.property_weights["hessian"] == 0.0

    declared_absent = data.config_from_atoms(_atoms(3, 2, declared=False), key_specification=ks)
    assert declared_absent.properties["hessian"] is None
    assert declared_absent.property_weights["hessian"] == 0.0

    with pytest.raises(ValueError, match="not symmetric"):
        data.config_from_atoms(_atoms(3, 3, symmetric=False), key_specification=ks)
    with pytest.raises(ValueError, match=r"numbers, not \(3N\)\^2"):
        data.config_from_atoms(_atoms(3, 4, length_ok=False), key_specification=ks)

    # a config_hessian_weight in info is the per-structure weight
    at = _atoms(3, 5)
    at.info["config_hessian_weight"] = 0.25
    assert data.config_from_atoms(at, key_specification=ks).property_weights["hessian"] == 0.25


def test_atomic_data_fields_and_batching():
    ks = data.KeySpecification.from_defaults()
    frames = [_atoms(3, 10), _atoms(4, 11, with_hessian=False), _atoms(2, 12)]
    ads = [
        data.AtomicData.from_config(data.config_from_atoms(a, key_specification=ks), z_table=TABLE, cutoff=5.0)
        for a in frames
    ]
    assert [int(ad.hessian.numel()) for ad in ads] == [81, 0, 36]
    assert [bool(ad.has_hessian) for ad in ads] == [True, False, True]
    assert [float(ad.hessian_weight) for ad in ads] == [1.0, 0.0, 1.0]  # absent -> weight 0, as mace does for absent forces
    for ad, a in zip(ads, frames):
        assert torch.allclose(ad.sqrt_masses, torch.tensor(np.sqrt(ase.data.atomic_masses[a.numbers])))

    loader = torch_geometric.dataloader.DataLoader(dataset=ads, batch_size=3, shuffle=False)
    batch = next(iter(loader))
    assert batch.hessian.shape == (81 + 36,)
    assert batch.has_hessian.tolist() == [True, False, True]
    assert batch.hessian_weight.shape == (3,)
    assert batch.sqrt_masses.shape == (9,)
    # slicing graph k's block by cumsum(9 n_k^2) over the labelled graphs
    n_k = (batch.ptr[1:] - batch.ptr[:-1]).tolist()
    off = 0
    for g, (n, has) in enumerate(zip(n_k, batch.has_hessian.tolist())):
        if not has:
            continue
        block = batch.hessian[off:off + 9 * n * n].reshape(3 * n, 3 * n).numpy()
        assert np.allclose(block, frames[g].info["_H"])
        off += 9 * n * n
    assert off == batch.hessian.numel()


def test_default_path_unchanged():
    """Without the key the fields exist, empty, and nothing else moves."""
    at = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    at.info["REF_energy"] = -1.0
    at.arrays["REF_forces"] = np.zeros((3, 3))
    ad = data.AtomicData.from_config(data.config_from_atoms(at, key_specification=data.KeySpecification.from_defaults()), z_table=TABLE, cutoff=5.0)
    assert ad.hessian.numel() == 0 and bool(ad.has_hessian) is False
    assert ad.forces.shape == (3, 3) and float(ad.energy) == -1.0
    # the constructor also accepts a call without the new keywords (calculator path)
    ad2 = data.AtomicData(
        edge_index=ad.edge_index, positions=ad.positions, shifts=ad.shifts, unit_shifts=ad.unit_shifts,
        cell=ad.cell, node_attrs=ad.node_attrs, weight=ad.weight, head=ad.head,
        energy_weight=ad.energy_weight, forces_weight=ad.forces_weight, stress_weight=ad.stress_weight,
        virials_weight=ad.virials_weight, dipole_weight=ad.dipole_weight, charges_weight=ad.charges_weight,
        polarizability_weight=ad.polarizability_weight, forces=ad.forces, energy=ad.energy, stress=ad.stress,
        virials=ad.virials, dipole=ad.dipole, charges=ad.charges, polarizability=ad.polarizability,
        elec_temp=ad.elec_temp, total_charge=ad.total_charge, total_spin=ad.total_spin,
    )
    assert ad2.hessian.numel() == 0 and bool(ad2.has_hessian) is False and float(ad2.hessian_weight) == 1.0
    assert ad2.sqrt_masses is None


def test_arg_parser_has_hessian_key():
    parser = tools.build_default_arg_parser()
    args = parser.parse_args(["--name", "x", "--train_file", "a.xyz"])
    assert args.hessian_key == "REF_hessian"
    args = parser.parse_args(["--name", "x", "--train_file", "a.xyz", "--hessian_key", "H"])
    assert args.hessian_key == "H"
