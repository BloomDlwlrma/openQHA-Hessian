"""`--loss external --loss_module package.module:factory`: the hook, the flags it reads
back, the full Hessian at evaluation, and Stage Two (openQHA-Hessian, commit B).

The loss module here is defined in this file and imported by name, so nothing about any
particular loss lives in mace.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms

from mace import tools
from mace.tools.scripts_utils import get_loss_fn, load_external_loss

CALLS = []


class FakeExternalLoss(torch.nn.Module):
    """Counts its calls and reports whether it was given a full Hessian."""

    wants_hessian_at_eval = True

    def __init__(self, args):
        super().__init__()
        self.energy_weight = float(args.energy_weight)
        self.forces_weight = float(args.forces_weight)
        self.hessian_weight = float(args.hessian_weight)
        self.n_probes = int(args.n_hessian_probes)
        self.probe = args.hessian_probe
        self.mode_weighting = args.hessian_mode_weighting
        self.saw_hessian = []

    def forward(self, ref, pred, ddp=None):
        from mace.modules.loss import (
            mean_squared_error_forces,
            weighted_mean_squared_error_energy,
        )

        self.saw_hessian.append(pred.get("hessian") is not None)
        loss = self.energy_weight * weighted_mean_squared_error_energy(ref, pred, ddp)
        loss = loss + self.forces_weight * mean_squared_error_forces(ref, pred, ddp)
        # a term that reads the Hessian label so the fields of commit A are exercised
        if ref.hessian.numel() > 0:
            loss = loss + self.hessian_weight * 0.0 * ref.hessian.sum()
        return loss

    def __repr__(self):
        return f"FakeExternalLoss(hessian_weight={self.hessian_weight:.3f}, probe={self.probe!r})"


class NotAModule:
    pass


def build(args):
    CALLS.append(args)
    return FakeExternalLoss(args)


def build_not_a_module(args):  # pylint: disable=unused-argument
    return NotAModule()


def _args(**kwargs):
    parser = tools.build_default_arg_parser()
    argv = ["--name", "x", "--train_file", "a.xyz"]
    for key, value in kwargs.items():
        argv += [f"--{key}", str(value)]
    return parser.parse_args(argv)


def test_flags_and_defaults():
    args = _args()
    assert args.loss_module is None
    assert (args.hessian_weight, args.n_hessian_probes) == (1.0, 4)
    assert (args.hessian_probe, args.hessian_mode_weighting) == ("rademacher", "entropy")
    assert args.swa_hessian_weight == 1.0
    args = _args(loss="external", loss_module="m:f", hessian_weight=7.5, n_hessian_probes=2,
                 hessian_probe="modes", hessian_mode_weighting="none")
    assert args.loss == "external" and args.loss_module == "m:f"
    assert (args.hessian_weight, args.n_hessian_probes) == (7.5, 2)
    assert (args.hessian_probe, args.hessian_mode_weighting) == ("modes", "none")
    with pytest.raises(SystemExit):
        _args(hessian_probe="hutchinson")


def test_get_loss_fn_calls_the_factory_with_the_args():
    CALLS.clear()
    args = _args(loss="external", loss_module=f"{__name__}:build", hessian_weight=3.0,
                 n_hessian_probes=2, hessian_probe="modes", forces_weight=100.0)
    loss_fn = get_loss_fn(args, dipole_only=False, compute_dipole=False)
    assert isinstance(loss_fn, FakeExternalLoss)
    assert len(CALLS) == 1 and CALLS[0] is args
    assert (loss_fn.hessian_weight, loss_fn.n_probes, loss_fn.probe) == (3.0, 2, "modes")
    assert loss_fn.forces_weight == 100.0
    assert "hessian_weight=3.000" in repr(loss_fn)


def test_the_hook_refuses_what_it_cannot_use():
    with pytest.raises(ValueError, match="needs --loss_module"):
        load_external_loss(_args(loss="external"))
    with pytest.raises(ModuleNotFoundError):
        load_external_loss(_args(loss="external", loss_module="no_such_module:build"))
    with pytest.raises(AttributeError):
        load_external_loss(_args(loss="external", loss_module=f"{__name__}:no_such_factory"))
    with pytest.raises(TypeError, match="not a torch.nn.Module"):
        load_external_loss(_args(loss="external", loss_module=f"{__name__}:build_not_a_module"))


def test_default_factory_name_is_build():
    loss_fn = load_external_loss(_args(loss="external", loss_module=__name__))
    assert isinstance(loss_fn, FakeExternalLoss)


def test_stage_two_rebuilds_with_the_swa_weights():
    """get_swa's external branch: the factory sees the Stage Two weights."""
    from mace.tools.scripts_utils import get_swa

    args = _args(loss="external", loss_module=f"{__name__}:build", hessian_weight=1.0,
                 forces_weight=100.0, swa_hessian_weight=0.1, swa_forces_weight=10.0,
                 max_num_epochs=8, start_swa=4)
    model = torch.nn.Linear(1, 1)
    swa, _swas = get_swa(args, model, torch.optim.Adam(model.parameters()), swas=[])
    assert isinstance(swa.loss_fn, FakeExternalLoss)
    assert (swa.loss_fn.hessian_weight, swa.loss_fn.forces_weight) == (0.1, 10.0)
    assert CALLS[-1] is not args  # a copy, so the training-stage loss is untouched


def _write_frames(path, n_frames=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_frames):
        at = Atoms("H2O", positions=np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
                   + rng.standard_normal((3, 3)) * 0.02)
        at.info["REF_energy"] = float(-10.0 + 0.1 * i)
        at.arrays["REF_forces"] = rng.standard_normal((3, 3)) * 0.1
        h = rng.standard_normal((9, 9))
        at.info["REF_hessian"] = (h + h.T).reshape(-1)
        at.cell = np.eye(3) * 20.0
        out.append(at)
    ase.io.write(str(path), out, format="extxyz")


@pytest.mark.parametrize("loss_flags", [
    ["--loss", "external", "--loss_module", "tests.test_external_loss:build"],
    ["--loss", "weighted"],
])
def test_run_train_end_to_end(tmp_path, loss_flags):
    """mace_run_train with the external loss trains; with the stock loss it is unchanged."""
    train = tmp_path / "train.xyz"
    _write_frames(train)
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    cmd = [
        sys.executable, "-m", "mace.cli.run_train",
        "--name", "ext", "--train_file", str(train), "--valid_fraction", "0.34",
        "--E0s", "average", "--model", "MACE", "--hidden_irreps", "8x0e", "--r_max", "4.0",
        "--num_channels", "8", "--max_L", "0", "--num_interactions", "1",
        "--batch_size", "2", "--valid_batch_size", "2", "--max_num_epochs", "2",
        "--seed", "3", "--device", "cpu", "--default_dtype", "float64",
        "--energy_weight", "1", "--forces_weight", "10", "--error_table", "PerAtomRMSE",
    ] + loss_flags
    proc = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    log = "\n".join((tmp_path / "logs").glob("ext*.log") and
                    [p.read_text() for p in (tmp_path / "logs").glob("ext*.log")])
    assert "Epoch" in log
    if "external" in loss_flags:
        assert "FakeExternalLoss" in log


class RecordingModel(torch.nn.Module):
    """A stand-in model: records the forward's keyword arguments and answers with a
    full Hessian when asked for one (mace's [3 n_nodes, n_nodes, 3] layout)."""

    def __init__(self):
        super().__init__()
        self.kwargs = []
        self.p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))

    def forward(self, batch_dict, **kwargs):
        self.kwargs.append(kwargs)
        n = batch_dict["positions"].shape[0]
        out = {
            "energy": torch.zeros(int(batch_dict["ptr"].numel() - 1), dtype=torch.float64) + self.p,
            "forces": torch.zeros_like(batch_dict["positions"]),
        }
        if kwargs.get("compute_hessian"):
            out["hessian"] = torch.zeros(3 * n, n, 3, dtype=torch.float64)
        return out


def _one_batch():
    from mace import data as mdata
    from mace.tools import torch_geometric

    at = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]])
    at.info["REF_energy"] = -1.0
    at.arrays["REF_forces"] = np.zeros((3, 3))
    h = np.eye(9)
    at.info["REF_hessian"] = h.reshape(-1)
    config = mdata.config_from_atoms(at, key_specification=mdata.KeySpecification.from_defaults())
    ad = mdata.AtomicData.from_config(config, z_table=tools.AtomicNumberTable([1, 8]), cutoff=5.0)
    return torch_geometric.dataloader.DataLoader(dataset=[ad], batch_size=1, shuffle=False)


def test_evaluate_asks_for_the_full_hessian_only_when_the_loss_wants_it():
    """The one line of commit B in `train.evaluate`: `compute_hessian` follows
    `output_args["hessian"]`, which `run_train` sets from the loss."""
    from mace.tools.train import evaluate

    torch.set_default_dtype(torch.float64)
    loader = _one_batch()
    loss_fn = load_external_loss(_args(loss="external", loss_module=f"{__name__}:build"))
    output_args = {"energy": True, "forces": True, "virials": False, "stress": False}

    model = RecordingModel()
    evaluate(model=model, loss_fn=loss_fn, data_loader=loader, output_args=output_args,
             device=torch.device("cpu"))
    assert model.kwargs[0]["compute_hessian"] is False        # absent key -> off, the default path
    assert loss_fn.saw_hessian and not any(loss_fn.saw_hessian)   # torchmetrics calls update twice per batch

    model = RecordingModel()
    loss_fn.saw_hessian.clear()
    evaluate(model=model, loss_fn=loss_fn, data_loader=loader,
             output_args=dict(output_args, hessian=True), device=torch.device("cpu"))
    assert model.kwargs[0]["compute_hessian"] is True
    assert loss_fn.saw_hessian and all(loss_fn.saw_hessian)       # the loss received the full matrix


def test_run_train_sets_output_args_from_the_loss():
    """`run_train` reads `wants_hessian_at_eval` off the loss -- as source, since running
    the whole driver twice to see one flag is not worth the minutes."""
    source = (Path(__file__).resolve().parents[1] / "mace" / "cli" / "run_train.py").read_text()
    assert 'output_args["hessian"] = bool(getattr(loss_fn, "wants_hessian_at_eval", False))' in source
