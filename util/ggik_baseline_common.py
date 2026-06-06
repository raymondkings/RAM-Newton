from __future__ import annotations

import importlib.util
import inspect
import json
import math
import sys
import zipfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from interface import Morphology, Task
from optim.direct_ik_baseline import (
    EPS,
    _build_morphology_tensors,
    _collision_critical_distance,
    _pose_se3_distance,
    _run_validation,
)
from optim.nrm_alpha_random_selection import (
    DEFAULT_DISTRIBUTION_BATCH_SIZE as NRM_DEFAULT_DISTRIBUTION_BATCH_SIZE,
    DEFAULT_NUM_ALPHA_CANDIDATES as NRM_DEFAULT_NUM_ALPHA_CANDIDATES,
    SE3_TIE_EPS,
    TOP_PROBABILITY_FRACTION as NRM_TOP_PROBABILITY_FRACTION,
)
from util.fixed_alpha_morphology_candidates import (
    DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
    DEFAULT_FIXED_ALPHA_RANDOM_TRIES,
    generate_alpha_candidates,
    sample_fixed_alpha_morphology_candidates_by_dof,
    _sample_one_chunk_with_fixed_alpha,
    _set_sampler_seed,
)
from util.kinematics import forward_kinematics
from util.optimization_csv_logger import OptimizationCSVLogger
from validation.optimization_validation import build_optimization_validation_context


PROJECT_ROOT = Path(__file__).parent.parent

# Legacy fallback only. The real path is resolved from the installed
# generative_graphik package; see _default_ggik_repo_path.
_LEGACY_GGIK_REPO_PATH = Path("/tmp/generative-graphik")


def _default_ggik_repo_path() -> Path:
    """Locate the generative-graphik checkout that ships paper_models.

    generative_graphik is a normal (editable) project dependency now, so prefer
    the repo root of the installed package: its parent directory holds
    paper_models.zip. find_spec is used instead of importing the package so this
    stays cheap and does not pull torch_geometric/graphik in at import time.
    Falls back to the legacy /tmp clone when the package is not an editable
    checkout (e.g. installed as a plain wheel).
    """
    try:
        spec = importlib.util.find_spec("generative_graphik")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        repo_root = Path(spec.origin).resolve().parent.parent
        if (repo_root / "paper_models").is_dir() or (
            repo_root / "paper_models.zip"
        ).is_file():
            return repo_root
    return _LEGACY_GGIK_REPO_PATH


DEFAULT_GGIK_MODEL_NAME = "67_4mil_480-80_checkpoint_model"
DEFAULT_CANDIDATE_DOFS = (7, 6, 5)
DEFAULT_NUM_ALPHA_CANDIDATES: int | str = NRM_DEFAULT_NUM_ALPHA_CANDIDATES
DEFAULT_CANDIDATE_BATCH_SIZE: int | str = "GGIK_FORWARD"
DEFAULT_DISTRIBUTION_BATCH_SIZE = NRM_DEFAULT_DISTRIBUTION_BATCH_SIZE
DEFAULT_TOP_FRACTION = NRM_TOP_PROBABILITY_FRACTION
# Number of morphologies optimized together. Internally this becomes
# morphology batch size * number of poses GGIK graphs per forward chunk.
DEFAULT_GGIK_MORPHOLOGY_BATCH_SIZE = 325
ZERO_ALPHA_RUN_EXCLUSION_LENGTH = 3


class GGIKUnavailableError(RuntimeError):
    pass


def _patch_networkx_shortest_path_for_graphik() -> None:
    import networkx as nx

    if getattr(nx.shortest_path, "_nrm_graphik_compat", False):
        return

    original_shortest_path = nx.shortest_path

    def shortest_path_compat(*args, **kwargs):
        result = original_shortest_path(*args, **kwargs)
        if not isinstance(result, (dict, list)):
            try:
                return dict(result)
            except TypeError:
                return result
        return result

    shortest_path_compat._nrm_graphik_compat = True
    nx.shortest_path = shortest_path_compat


@dataclass
class GGIKMetrics:
    loss_per_candidate: Tensor
    pose_success_rate: Tensor
    best_se3_mean: Tensor
    self_collision_pose_rate: Tensor
    processed_morphologies: Tensor
    raw_morphologies: Tensor


class TimGenerativeGraphIKSolver:
    """Thin adapter around Tim/UTIAS generative-graphIK.

    This adapter intentionally imports GGIK dependencies lazily. The current
    project environment does not ship torch_geometric, graphik, or liegroups, so
    importing this module must remain cheap and compile-safe.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        repo_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        model_name: str = DEFAULT_GGIK_MODEL_NAME,
        num_samples: int = 8,
        torch_postprocess: bool = True,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.repo_path = (
            Path(repo_path).expanduser() if repo_path else _default_ggik_repo_path()
        )
        self.num_samples = int(num_samples)
        self.torch_postprocess = bool(torch_postprocess)

        if self.num_samples <= 0:
            raise ValueError("num_ggik_samples must be positive.")

        self._ensure_import_path()
        self._import_dependencies()

        resolved_model_dir = self._resolve_model_dir(model_dir, model_name)
        self.model = self._load_model(resolved_model_dir)

    def _ensure_import_path(self) -> None:
        if not self.repo_path.is_dir():
            raise GGIKUnavailableError(
                f"generative-graphIK repo was not found at {self.repo_path}. "
                "generative-graphik is a project dependency, so install it as an "
                "editable checkout (uv sync), or pass 'ggik_repo_path' pointing "
                "at a clone that contains paper_models.zip."
            )
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    def _import_dependencies(self) -> None:
        try:
            _patch_networkx_shortest_path_for_graphik()
            from graphik.graphs import ProblemGraphRevolute
            from graphik.robots import RobotRevolute
            from graphik.utils.dgp import graph_from_pos
            from liegroups.numpy import SE3
            from torch_geometric.data import Batch
            from generative_graphik.utils.dataset_generation import (
                generate_data_point_from_pose,
            )
            from generative_graphik.utils.torch_utils import (
                batchIKmultiDOF,
                repeat_offset_index,
            )
        except ModuleNotFoundError as exc:
            raise GGIKUnavailableError(
                "GGIK dependencies are missing. Required packages include "
                "torch_geometric, graphik, liegroups, and generative_graphik. "
                "Tim's repo pins graphik/liegroups from git branches, so this "
                "is not installed in the current nrm-newton environment."
            ) from exc

        self.ProblemGraphRevolute = ProblemGraphRevolute
        self.RobotRevolute = RobotRevolute
        self.graph_from_pos = graph_from_pos
        self.SE3 = SE3
        self.Batch = Batch
        self.generate_data_point_from_pose = generate_data_point_from_pose
        self.batchIKmultiDOF = batchIKmultiDOF
        self.repeat_offset_index = repeat_offset_index

    def _resolve_model_dir(
        self,
        model_dir: str | Path | None,
        model_name: str,
    ) -> Path:
        if model_dir is not None:
            resolved = Path(model_dir).expanduser()
            if not resolved.is_dir():
                raise GGIKUnavailableError(f"ggik_model_dir does not exist: {resolved}")
            return resolved

        paper_models_dir = self.repo_path / "paper_models"
        if not paper_models_dir.is_dir():
            zip_path = self.repo_path / "paper_models.zip"
            if zip_path.is_file():
                with zipfile.ZipFile(zip_path) as archive:
                    archive.extractall(self.repo_path)

        resolved = paper_models_dir / model_name
        if not resolved.is_dir():
            raise GGIKUnavailableError(
                f"Could not find GGIK model directory: {resolved}. "
                "Pass 'ggik_model_dir' explicitly."
            )
        return resolved

    def _load_model(self, model_dir: Path):
        model_py = model_dir / "model.py"
        hparams_path = model_dir / "hyperparameters.txt"
        checkpoint_path = model_dir / "checkpoints" / "checkpoint.pth"
        net_path = model_dir / "net.pth"

        if not model_py.is_file():
            raise GGIKUnavailableError(f"Missing GGIK model.py: {model_py}")
        if not hparams_path.is_file():
            raise GGIKUnavailableError(f"Missing GGIK hyperparameters: {hparams_path}")
        if not checkpoint_path.is_file() and not net_path.is_file():
            raise GGIKUnavailableError(
                f"Missing GGIK weights under {model_dir}; expected checkpoint.pth or net.pth."
            )

        spec = importlib.util.spec_from_file_location(
            f"ggik_model_{abs(hash(model_dir))}",
            model_py,
        )
        if spec is None or spec.loader is None:
            raise GGIKUnavailableError(f"Could not import GGIK model from {model_py}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model_args = Namespace(**json.loads(hparams_path.read_text()))
        model = module.Model(model_args).to(self.device)

        weights_path = checkpoint_path if checkpoint_path.is_file() else net_path
        state = torch.load(weights_path, map_location=self.device)
        if isinstance(state, dict) and "net" in state:
            state = state["net"]
        model.load_state_dict(state)
        model.eval()
        self._forward_eval_uses_data_api = (
            "data" in inspect.signature(model.forward_eval).parameters
        )
        return model

    def _graph_from_morphology(self, morphology: Tensor):
        detached = morphology.detach().to("cpu", dtype=torch.float32)
        seq_len = detached.shape[0]
        params = {
            "alpha": detached[:, 0].tolist(),
            "a": detached[:, 1].tolist(),
            "d": detached[:, 2].tolist(),
            "theta": [0.0] * seq_len,
            "num_joints": seq_len,
            "modified_dh": True,
        }
        robot = self.RobotRevolute(params)
        return robot, self.ProblemGraphRevolute(robot)

    def _differentiable_t0(self, morphology: Tensor) -> Tensor:
        seq_len = morphology.shape[0]
        joints = torch.zeros(
            1,
            seq_len,
            1,
            device=morphology.device,
            dtype=morphology.dtype,
        )
        poses = forward_kinematics(morphology.unsqueeze(0), joints)[0]
        identity = torch.eye(
            4,
            device=morphology.device,
            dtype=morphology.dtype,
        ).unsqueeze(0)
        return torch.cat([identity, poses], dim=0)

    def _generate_data_point_from_pose_cpu(self, graph, target_pose_se3):
        default_device = torch.get_default_device()
        torch.set_default_device("cpu")
        try:
            return self.generate_data_point_from_pose(graph, target_pose_se3)
        finally:
            torch.set_default_device(default_device)

    def _forward_eval_core(
        self,
        *,
        x: Tensor,
        h: Tensor,
        edge_attr: Tensor,
        edge_attr_partial_full: Tensor,
        edge_index: Tensor,
        partial_goal_mask: Tensor,
        nodes_per_single_graph: int,
        batch_size: int,
    ) -> Tensor:
        nodes = partial_goal_mask[:, None] * x
        edges = edge_attr_partial_full
        data_index = edge_index
        h_current = h

        for ii in range(self.model.max_num_iterations):
            with torch.no_grad():
                z_goal_partial = self.model.goal_partial_config_encoder(
                    x=nodes,
                    h=h_current,
                    edge_attr=edges,
                    edge_index=data_index,
                )

                if self.model.train_prior:
                    params = self.model.prior_encoder(z_goal_partial)
                    pz_c = self.model.pz_c_dist(*params)
                else:
                    pz_c = self.model.pz_c_dist(
                        loc=torch.zeros(
                            (
                                batch_size * nodes_per_single_graph,
                                z_goal_partial.shape[-1],
                            ),
                            device=z_goal_partial.device,
                        ),
                        scale=torch.ones(
                            (
                                batch_size * nodes_per_single_graph,
                                z_goal_partial.shape[-1],
                            ),
                            device=z_goal_partial.device,
                        ),
                    )

                if ii == 0:
                    z_prior = pz_c.sample([self.num_samples])
                    z_prior = z_prior.reshape(-1, z_prior.shape[-1])
                    z_goal_partial = z_goal_partial.unsqueeze(0).repeat(
                        self.num_samples,
                        1,
                        1,
                    )
                    z_goal_partial = z_goal_partial.reshape(
                        -1,
                        z_goal_partial.shape[-1],
                    )
                    h_current = h_current.unsqueeze(0).repeat(
                        self.num_samples,
                        1,
                        1,
                    )
                    h_current = h_current.reshape(-1, h_current.shape[-1])
                    data_index = self.repeat_offset_index(
                        edge_index,
                        self.num_samples,
                        batch_size * nodes_per_single_graph,
                    )
                    data_index = data_index.reshape(data_index.shape[0], -1)
                    data_edge_attr = edge_attr.unsqueeze(0).repeat(
                        self.num_samples,
                        1,
                        1,
                    )
                    data_edge_attr = data_edge_attr.reshape(
                        -1,
                        data_edge_attr.shape[-1],
                    )
                else:
                    z_prior = pz_c.sample()

                inp_decoder_prior = torch.cat(
                    (
                        z_prior,
                        z_goal_partial,
                    ),
                    dim=-1,
                )
                mu_x_sample = self.model.decoder(
                    x=inp_decoder_prior,
                    h=h_current,
                    edge_attr=0.0 * data_edge_attr,
                    edge_index=data_index,
                )

                if self.model.max_num_iterations > 1 and self.model.train_prior:
                    nodes = mu_x_sample
                    src, dst = data_index
                    edges = ((nodes[src] - nodes[dst]) ** 2).sum(dim=-1).sqrt()
                    edges = edges.unsqueeze(-1)

        return mu_x_sample.reshape(
            self.num_samples,
            batch_size * nodes_per_single_graph,
            -1,
        )

    def _forward_eval_preprocessed_data(self, data) -> Tensor:
        batch_size = int(data.num_graphs)
        nodes_per_single_graph = int(data.num_nodes / batch_size)

        if self._forward_eval_uses_data_api:
            if batch_size == 1:
                return self.model.forward_eval(data, num_samples=self.num_samples).to(
                    device=self.device,
                    dtype=self.dtype,
                )

            t_ee = data.T_ee
            if t_ee.dim() == 1:
                t_ee = t_ee.reshape(batch_size, -1)
            dim = t_ee.shape[-1] // 3 + 1
            goal_data_repeated_per_node = torch.repeat_interleave(
                t_ee,
                (dim - 1) * data.num_joints + self.model.num_anchor_nodes,
                dim=0,
            )
            return self._forward_eval_core(
                x=data.pos,
                h=torch.cat((data.type, goal_data_repeated_per_node), dim=-1),
                edge_attr=data.edge_attr,
                edge_attr_partial_full=data.edge_attr_partial_full,
                edge_index=data.edge_index_full,
                partial_goal_mask=data.partial_goal_mask,
                nodes_per_single_graph=nodes_per_single_graph,
                batch_size=batch_size,
            ).to(device=self.device, dtype=self.dtype)

        return self._forward_eval_core(
            x=data.pos,
            h=torch.cat((data.type, data.goal_data_repeated_per_node), dim=-1),
            edge_attr=data.edge_attr,
            edge_attr_partial_full=data.edge_attr_partial,
            edge_index=data.edge_index_full,
            partial_goal_mask=data.partial_goal_mask,
            nodes_per_single_graph=nodes_per_single_graph,
            batch_size=batch_size,
        ).to(device=self.device, dtype=self.dtype)

    def _predict_one_pose(
        self,
        *,
        morphology: Tensor,
        graph,
        target_pose: Tensor,
    ) -> Tensor:
        seq_len = morphology.shape[0]
        target_pose_np = target_pose.detach().cpu().numpy()
        target_pose_se3 = self.SE3.from_matrix(target_pose_np, normalize=True)
        data = self._generate_data_point_from_pose_cpu(graph, target_pose_se3).to(
            self.device
        )
        data.num_graphs = 1
        data = self.model.preprocess(data)

        p_all = self._forward_eval_preprocessed_data(data)

        if self.torch_postprocess:
            t0 = self._differentiable_t0(morphology)
            num_joints = torch.full(
                (self.num_samples,),
                seq_len,
                device=self.device,
                dtype=torch.long,
            )
            q_raw = self.batchIKmultiDOF(
                p_all.reshape(-1, 3),
                t0.repeat(self.num_samples, 1, 1),
                num_joints,
                T_final=target_pose[None].expand(self.num_samples, -1, -1),
            ).reshape(self.num_samples, seq_len)
        else:
            q_rows = []
            for sample_idx in range(self.num_samples):
                p_sample = p_all[sample_idx].detach().cpu().numpy()
                graph_pos = self.graph_from_pos(p_sample, graph.node_ids)
                q_dict = graph.joint_variables(
                    graph_pos,
                    T_final={f"p{seq_len}": target_pose_se3},
                )
                q_rows.append(
                    torch.tensor(
                        list(q_dict.values()),
                        device=self.device,
                        dtype=self.dtype,
                    )
                )
            q_raw = torch.stack(q_rows, dim=0)

        zeros = torch.zeros(
            self.num_samples,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        return torch.cat([q_raw[:, 1:], zeros], dim=1).unsqueeze(-1)

    def _predict_pose_batch(self, pending: list[tuple[int, int, Tensor, Any, Tensor]]):
        if not self.torch_postprocess:
            return [
                (
                    morph_idx,
                    pose_idx,
                    self._predict_one_pose(
                        morphology=morphology,
                        graph=graph,
                        target_pose=target_pose,
                    ),
                )
                for morph_idx, pose_idx, morphology, graph, target_pose in pending
            ]

        data_list = []
        morphologies = []
        target_poses = []
        for _morph_idx, _pose_idx, morphology, graph, target_pose in pending:
            target_pose_np = target_pose.detach().cpu().numpy()
            target_pose_se3 = self.SE3.from_matrix(target_pose_np, normalize=True)
            data_single = self._generate_data_point_from_pose_cpu(
                graph,
                target_pose_se3,
            ).to(self.device)
            data_single.num_graphs = 1
            data_list.append(self.model.preprocess(data_single))
            morphologies.append(morphology)
            target_poses.append(target_pose)

        data = self.Batch.from_data_list(data_list)
        p_all = self._forward_eval_preprocessed_data(data)

        batch_size = len(pending)
        seq_len = morphologies[0].shape[0]
        t0 = torch.stack(
            [self._differentiable_t0(morphology) for morphology in morphologies],
            dim=0,
        )
        t0 = (
            t0.unsqueeze(0)
            .expand(self.num_samples, -1, -1, -1, -1)
            .reshape(self.num_samples * batch_size * (seq_len + 1), 4, 4)
        )
        num_joints = torch.full(
            (self.num_samples * batch_size,),
            seq_len,
            device=self.device,
            dtype=torch.long,
        )
        target_batch = torch.stack(target_poses, dim=0)
        target_batch = (
            target_batch.unsqueeze(0)
            .expand(self.num_samples, -1, -1, -1)
            .reshape(self.num_samples * batch_size, 4, 4)
        )

        q_raw = self.batchIKmultiDOF(
            p_all.reshape(-1, 3),
            t0,
            num_joints,
            T_final=target_batch,
        ).reshape(self.num_samples, batch_size, seq_len)
        q_raw = q_raw.permute(1, 0, 2)

        zeros = torch.zeros(
            batch_size,
            self.num_samples,
            1,
            device=self.device,
            dtype=self.dtype,
        )
        joints = torch.cat([q_raw[:, :, 1:], zeros], dim=-1).unsqueeze(-1)

        return [
            (morph_idx, pose_idx, joints[local_idx])
            for local_idx, (morph_idx, pose_idx, *_rest) in enumerate(pending)
        ]

    def predict_joints(
        self,
        morphologies: Tensor,
        target_poses_local: Tensor,
    ) -> Tensor:
        flat_results: list[Tensor | None] = [
            None
            for _ in range(
                int(morphologies.shape[0]) * int(target_poses_local.shape[0])
            )
        ]
        pending: list[tuple[int, int, Tensor, Any, Tensor]] = []
        graph_forward_batch_size = max(
            1,
            int(morphologies.shape[0]) * int(target_poses_local.shape[0]),
        )

        def flush_pending() -> None:
            nonlocal pending
            if not pending:
                return
            for morph_idx, pose_idx, joints in self._predict_pose_batch(pending):
                flat_results[morph_idx * target_poses_local.shape[0] + pose_idx] = (
                    joints
                )
            pending = []

        for morph_idx, morphology in enumerate(morphologies):
            _robot, graph = self._graph_from_morphology(morphology)
            for pose_idx, pose in enumerate(target_poses_local):
                pending.append(
                    (
                        int(morph_idx),
                        int(pose_idx),
                        morphology,
                        graph,
                        pose,
                    )
                )
                if len(pending) >= graph_forward_batch_size:
                    flush_pending()
        flush_pending()

        if any(result is None for result in flat_results):
            raise RuntimeError("GGIK joint prediction produced incomplete results.")

        flat_joints = torch.stack(
            [result for result in flat_results if result is not None]
        )
        return flat_joints.reshape(
            morphologies.shape[0],
            target_poses_local.shape[0],
            self.num_samples,
            morphologies.shape[1],
            1,
        )


def _quantize_alpha(alpha: Tensor) -> Tensor:
    levels = alpha.new_tensor([-math.pi / 2.0, 0.0, math.pi / 2.0])
    idx = (alpha.unsqueeze(-1) - levels).abs().argmin(dim=-1)
    return levels[idx]


def _ggik_parameters(optimization_parameters: dict) -> dict[str, Any]:
    lr_fallback = float(optimization_parameters.get("learning_rate", 0.01))
    return {
        "num_iterations": int(optimization_parameters.get("num_iterations", 100)),
        "learning_rate_length": float(
            optimization_parameters.get("learning_rate_length", lr_fallback)
        ),
        "learning_rate_alpha": float(
            optimization_parameters.get("learning_rate_alpha", lr_fallback)
        ),
        "collision_weight": float(
            optimization_parameters.get("collision_weight", 10.0)
        ),
        "collision_margin": float(optimization_parameters.get("collision_margin", 0.0)),
        "success_eps": float(optimization_parameters.get("success_eps", EPS)),
        "softmin_temperature": float(
            optimization_parameters.get("softmin_temperature", 0.01)
        ),
    }


def _build_ggik_solver(
    optimization_parameters: dict,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> TimGenerativeGraphIKSolver:
    return TimGenerativeGraphIKSolver(
        device=device,
        dtype=dtype,
        repo_path=optimization_parameters.get("ggik_repo_path"),
        model_dir=optimization_parameters.get("ggik_model_dir"),
        model_name=str(
            optimization_parameters.get("ggik_model_name", DEFAULT_GGIK_MODEL_NAME)
        ),
        num_samples=int(optimization_parameters.get("num_ggik_samples", 8)),
        torch_postprocess=bool(
            optimization_parameters.get("ggik_torch_postprocess", True)
        ),
    )


def _ggik_metrics(
    *,
    alpha: Tensor,
    lengths: Tensor,
    solver: TimGenerativeGraphIKSolver,
    target_poses_local: Tensor,
    link_radius: float,
    collision_weight: float,
    collision_margin: float,
    success_eps: float,
    softmin_temperature: float,
) -> GGIKMetrics:
    raw_morphologies, processed_morphologies = _build_morphology_tensors(
        alpha,
        lengths,
        link_radius,
    )

    joints = solver.predict_joints(processed_morphologies, target_poses_local)
    num_candidates, num_poses, num_samples = joints.shape[:3]
    seq_len = processed_morphologies.shape[-2]

    morph_expanded = processed_morphologies[:, None, None].expand(
        num_candidates,
        num_poses,
        num_samples,
        seq_len,
        3,
    )
    flat_morph = morph_expanded.reshape(
        num_candidates * num_poses * num_samples,
        seq_len,
        3,
    )
    flat_joints = joints.reshape(num_candidates * num_poses * num_samples, seq_len, 1)

    reached_poses = forward_kinematics(flat_morph, flat_joints)
    ee_poses = reached_poses[:, -1].reshape(
        num_candidates,
        num_poses,
        num_samples,
        4,
        4,
    )

    target = target_poses_local[None, :, None].expand_as(ee_poses)
    se3 = _pose_se3_distance(ee_poses, target)

    critical_distance = _collision_critical_distance(
        flat_morph,
        reached_poses,
        radius=link_radius,
    ).reshape(num_candidates, num_poses, num_samples)

    collision_penalty = F.relu(collision_margin - critical_distance)
    sample_score = se3 + collision_weight * collision_penalty

    if softmin_temperature > 0.0:
        weights = torch.softmax(-sample_score / softmin_temperature, dim=-1)
        best_score = (weights * sample_score).sum(dim=-1)
        best_se3 = (weights * se3).sum(dim=-1)
    else:
        best_score, best_idx = sample_score.min(dim=-1)
        best_se3 = se3.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

    success_per_sample = (se3 <= success_eps) & (critical_distance >= collision_margin)

    return GGIKMetrics(
        loss_per_candidate=best_score.mean(dim=-1),
        pose_success_rate=success_per_sample.any(dim=-1).float().mean(dim=-1),
        best_se3_mean=best_se3.mean(dim=-1),
        self_collision_pose_rate=(critical_distance < 0.0)
        .any(dim=-1)
        .float()
        .mean(dim=-1),
        processed_morphologies=processed_morphologies,
        raw_morphologies=raw_morphologies,
    )


def run_ggik_morphology_optimization(
    *,
    initial_morphologies: Tensor,
    target_poses_local: Tensor,
    link_radius: float,
    solver: TimGenerativeGraphIKSolver,
    ggik_params: dict[str, Any],
    random_seed: int,
    optimize_alpha: bool,
    quantize_alpha_at_end: bool,
    logging: bool,
    desc: str,
) -> GGIKMetrics:
    del random_seed
    if ggik_params["num_iterations"] <= 0:
        raise ValueError("num_iterations must be positive.")

    alpha = initial_morphologies[..., 0:1].detach().clone()
    lengths = initial_morphologies[..., 1:].detach().clone().requires_grad_(True)

    params = [{"params": [lengths], "lr": ggik_params["learning_rate_length"]}]
    if optimize_alpha:
        alpha = alpha.requires_grad_(True)
        params.append({"params": [alpha], "lr": ggik_params["learning_rate_alpha"]})

    optimizer = torch.optim.AdamW(params, weight_decay=0.0)
    iterator = tqdm(
        range(ggik_params["num_iterations"]),
        desc=desc,
        disable=not logging,
        dynamic_ncols=True,
    )

    final_metrics: GGIKMetrics | None = None
    for _ in iterator:
        optimizer.zero_grad(set_to_none=True)
        metrics = _ggik_metrics(
            alpha=alpha,
            lengths=lengths,
            solver=solver,
            target_poses_local=target_poses_local,
            link_radius=link_radius,
            collision_weight=ggik_params["collision_weight"],
            collision_margin=ggik_params["collision_margin"],
            success_eps=ggik_params["success_eps"],
            softmin_temperature=ggik_params["softmin_temperature"],
        )
        loss = metrics.loss_per_candidate.mean()
        loss.backward()
        optimizer.step()
        final_metrics = metrics

        if logging:
            iterator.set_postfix(
                loss=f"{loss.item():.4f}",
                success=f"{metrics.pose_success_rate.mean().item():.3f}",
                se3=f"{metrics.best_se3_mean.mean().item():.4f}",
            )

    final_alpha = alpha.detach()
    if quantize_alpha_at_end:
        final_alpha = _quantize_alpha(final_alpha)

    final_metrics = _ggik_metrics(
        alpha=final_alpha,
        lengths=lengths.detach(),
        solver=solver,
        target_poses_local=target_poses_local,
        link_radius=link_radius,
        collision_weight=ggik_params["collision_weight"],
        collision_margin=ggik_params["collision_margin"],
        success_eps=ggik_params["success_eps"],
        softmin_temperature=ggik_params["softmin_temperature"],
    )
    return final_metrics


def optimize_single_morphology(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
    *,
    optimize_alpha: bool = False,
    quantize_alpha_at_end: bool = False,
    validate_with_curobo: bool = False,
    label: str = "GGIK baseline",
) -> tuple[Morphology, Path]:
    ggik_params = _ggik_parameters(optimization_parameters)
    logging = bool(optimization_parameters.get("logging", True))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    number_random_seed = int(optimization_parameters.get("number_random_seed", 32))
    percentage_poses = float(optimization_parameters.get("percentage_poses", 1))
    ignore_ground = bool(optimization_parameters.get("ignore_ground", False))
    ignore_obstacles = bool(optimization_parameters.get("ignore_obstacles", False))

    device = morph.params.device
    dtype = morph.params.dtype
    target_poses_local = task.goal_poses.to(device=device, dtype=dtype)
    solver = _build_ggik_solver(optimization_parameters, device=device, dtype=dtype)
    csv_logger = OptimizationCSVLogger(root_dir=PROJECT_ROOT)

    scene = None
    if validate_with_curobo:
        scene = build_optimization_validation_context(
            task=task,
            device=device,
            ignore_ground=ignore_ground,
            ignore_obstacles=ignore_obstacles,
        )

    if logging:
        print(f"[Info] Starting {label} on device {device}.")
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    try:
        metrics = run_ggik_morphology_optimization(
            initial_morphologies=morph.params.detach().clone().unsqueeze(0),
            target_poses_local=target_poses_local,
            link_radius=morph.link_radius,
            solver=solver,
            ggik_params=ggik_params,
            random_seed=random_seed,
            optimize_alpha=optimize_alpha,
            quantize_alpha_at_end=quantize_alpha_at_end,
            logging=logging,
            desc=label,
        )

        validation_data = None
        if validate_with_curobo:
            validation_data = _run_validation(
                processed_morphology=metrics.processed_morphologies[0],
                morph=morph,
                task=task,
                scene=scene,
                device=device,
                percentage_poses=percentage_poses,
                number_random_seed=number_random_seed,
                random_seed=random_seed,
            )
        else:
            validation_data = {
                "ik_success_pose_rate": metrics.pose_success_rate.mean(),
                "best_se3_dist_mean": metrics.best_se3_mean.mean(),
            }

        csv_logger.log_iteration(
            iteration=ggik_params["num_iterations"],
            loss=metrics.loss_per_candidate.mean(),
            reachability_probability=metrics.pose_success_rate.mean(),
            raw_morphology=metrics.raw_morphologies[0],
            processed_morphology=metrics.processed_morphologies[0],
            validation_data=validation_data,
        )

        print(
            f"[Final {label}] "
            f"loss={metrics.loss_per_candidate.mean().item():.6f}, "
            f"ggik_success={metrics.pose_success_rate.mean().item() * 100.0:.2f}%, "
            f"ggik_se3={metrics.best_se3_mean.mean().item():.6f}"
        )

        return (
            Morphology(
                params=metrics.processed_morphologies[0].detach(),
                link_radius=morph.link_radius,
            ),
            csv_logger.csv_path,
        )
    finally:
        csv_logger.close()


def _resolve_candidate_dofs(value: Any) -> list[int]:
    if value is None:
        return list(DEFAULT_CANDIDATE_DOFS)
    if isinstance(value, str):
        dofs = [int(part) for part in value.replace(",", " ").split()]
    else:
        dofs = [int(dof) for dof in value]
    if not dofs:
        raise ValueError("candidate_dofs must not be empty.")
    if any(dof <= 0 for dof in dofs):
        raise ValueError(f"candidate_dofs must be positive, got {dofs}.")
    seen: set[int] = set()
    ordered_dofs: list[int] = []
    for dof in dofs:
        if dof not in seen:
            ordered_dofs.append(dof)
            seen.add(dof)
    return ordered_dofs


def _resolve_candidate_batch_size(
    value: Any,
    *,
    total_candidates: int,
    num_poses: int,
) -> int:
    if total_candidates <= 0:
        raise ValueError("total_candidates must be positive.")

    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized == "ALL":
            return total_candidates
        if normalized in {"AUTO", "GGIK_FORWARD", "FORWARD"}:
            return min(
                total_candidates,
                max(1, int(DEFAULT_GGIK_MORPHOLOGY_BATCH_SIZE)),
            )

    batch_size = int(value)
    if batch_size <= 0:
        raise ValueError(
            "candidate_batch_size must be positive, 'ALL', or 'GGIK_FORWARD'."
        )
    return min(batch_size, total_candidates)


def _generate_alpha_candidates_by_dof(
    *,
    dofs: list[int],
    requested_num_candidates: int | str | None,
    device: torch.device,
    generator: torch.Generator,
    logging: bool,
) -> dict[int, Tensor]:
    alpha_candidates_by_dof: dict[int, Tensor] = {}
    for dof in dofs:
        alpha_candidates, max_alpha_candidates, using_all = generate_alpha_candidates(
            requested_num_candidates=requested_num_candidates,
            seq_len=dof + 1,
            device=device,
            generator=generator,
            forbidden_zero_run_length=ZERO_ALPHA_RUN_EXCLUSION_LENGTH,
        )
        alpha_candidates_by_dof[dof] = alpha_candidates
        if logging:
            print(
                f"[Info] DOF{dof} alpha candidates generated: "
                f"{alpha_candidates.shape[0]} / max_valid={max_alpha_candidates} "
                f"(using_all={using_all})."
            )
    return alpha_candidates_by_dof


def _sample_fixed_alpha_morphology_candidates_by_dof_no_cache(
    *,
    alpha_candidates_by_dof: dict[int, Tensor],
    seed: int | None,
    link_radius: float,
    batch_size: int,
    max_attempts_per_alpha: int,
    dynamic_rejection_batch_size: int,
    logging: bool,
) -> dict[int, Tensor]:
    if not alpha_candidates_by_dof:
        raise ValueError("alpha_candidates_by_dof must not be empty.")

    first_alpha = next(iter(alpha_candidates_by_dof.values()))
    _set_sampler_seed(seed, first_alpha.device)

    sampled_by_dof: dict[int, Tensor] = {}
    for dof, alpha_candidates in sorted(alpha_candidates_by_dof.items()):
        chunks: list[Tensor] = []
        total_failed = 0
        iterator = tqdm(
            range(0, alpha_candidates.shape[0], batch_size),
            desc=f"sampling uncached DOF{dof} fixed-alpha morphologies",
            disable=not logging,
            dynamic_ncols=True,
        )
        for start in iterator:
            end = min(start + batch_size, alpha_candidates.shape[0])
            sampled, failed_count = _sample_one_chunk_with_fixed_alpha(
                alpha_candidates[start:end].detach(),
                link_radius=link_radius,
                max_attempts_per_alpha=max_attempts_per_alpha,
                dynamic_rejection_batch_size=dynamic_rejection_batch_size,
            )
            if sampled.numel() > 0:
                chunks.append(sampled)
            total_failed += failed_count

        if not chunks:
            raise RuntimeError(
                f"Fixed-alpha sampler failed to generate any DOF{dof} morphology."
            )

        sampled_by_dof[int(dof)] = torch.cat(chunks, dim=0)
        if logging:
            print(
                f"[Info] Uncached DOF{dof} fixed-alpha sampling: "
                f"sampled {sampled_by_dof[int(dof)].shape[0]}/"
                f"{alpha_candidates.shape[0]} candidates, failed={total_failed}."
            )

    return sampled_by_dof


def _distribution_valid_mask(
    processed_morphologies: Tensor,
    batch_size: int,
    logging: bool,
    desc: str,
) -> Tensor:
    from util.distribution_checker import check_morphology_distribution

    reports = []
    for start in tqdm(
        range(0, processed_morphologies.shape[0], batch_size),
        desc=desc,
        disable=not logging,
        dynamic_ncols=True,
    ):
        end = min(start + batch_size, processed_morphologies.shape[0])
        reports.extend(
            check_morphology_distribution(
                processed_morphologies[start:end],
                num_joint_samples=1000,
                seed=0,
            )
        )

    return torch.tensor(
        [report.valid for report in reports],
        dtype=torch.bool,
        device=processed_morphologies.device,
    )


def _tie_score(
    processed_morphology: Tensor, link_radius: float
) -> tuple[Tensor, Tensor]:
    ad_abs = processed_morphology[..., 1:].abs()
    link_mag = torch.linalg.norm(processed_morphology[..., 1:], dim=-1)

    eps_zero = 1e-6
    min_good_nonzero = 4.0 * link_radius

    length_sum = ad_abs.sum()
    zero_reward = (ad_abs <= eps_zero).float().sum()

    is_tiny_nonzero = (ad_abs > eps_zero) & (ad_abs < min_good_nonzero)
    tiny_penalty = (min_good_nonzero - ad_abs).clamp_min(0.0)
    tiny_penalty = (tiny_penalty * is_tiny_nonzero.float()).sum()

    active_link = link_mag > eps_zero
    mean_link = (
        link_mag * active_link.float()
    ).sum() / active_link.float().sum().clamp_min(1.0)
    balance_penalty = ((link_mag - mean_link) ** 2 * active_link.float()).sum()

    score = (
        10.0 * tiny_penalty
        + 2.0 * balance_penalty
        + 0.02 * length_sum
        - 0.003 * zero_reward
    )
    return score, length_sum


def _candidate_sort_key(record: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(record["ggik_se3"].detach().cpu().item()),
        -float(record["ggik_success"].detach().cpu().item()),
        float(record["loss"].detach().cpu().item()),
    )


def _optimize_one_dof_group(
    *,
    dof: int,
    initial_candidate_morphologies: Tensor,
    target_poses_local: Tensor,
    link_radius: float,
    solver: TimGenerativeGraphIKSolver,
    ggik_params: dict[str, Any],
    candidate_batch_size: Any,
    distribution_batch_size: int,
    random_seed: int,
    logging: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    resolved_candidate_batch_size = _resolve_candidate_batch_size(
        candidate_batch_size,
        total_candidates=initial_candidate_morphologies.shape[0],
        num_poses=target_poses_local.shape[0],
    )
    if logging:
        print(
            f"[Info] DOF{dof} GGIK optimization batching: "
            f"candidate_batch_size={resolved_candidate_batch_size}, "
            f"ggik_morphology_batch_size={DEFAULT_GGIK_MORPHOLOGY_BATCH_SIZE}, "
            f"ggik_graph_forward_batch_size="
            f"{resolved_candidate_batch_size * target_poses_local.shape[0]}, "
            f"num_candidates={initial_candidate_morphologies.shape[0]}, "
            f"num_poses={target_poses_local.shape[0]}."
        )

    for start in range(
        0,
        initial_candidate_morphologies.shape[0],
        resolved_candidate_batch_size,
    ):
        end = min(
            start + resolved_candidate_batch_size,
            initial_candidate_morphologies.shape[0],
        )
        metrics = run_ggik_morphology_optimization(
            initial_morphologies=initial_candidate_morphologies[start:end],
            target_poses_local=target_poses_local,
            link_radius=link_radius,
            solver=solver,
            ggik_params=ggik_params,
            random_seed=random_seed + start,
            optimize_alpha=False,
            quantize_alpha_at_end=False,
            logging=logging,
            desc=f"GGIK DOF{dof} candidates {start}-{end}",
        )

        valid_mask = _distribution_valid_mask(
            metrics.processed_morphologies.detach(),
            batch_size=distribution_batch_size,
            logging=logging,
            desc=f"post-checking DOF{dof} GGIK candidates {start}-{end}",
        )

        valid_indices = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
        for local_idx in valid_indices.tolist():
            records.append(
                {
                    "dof": dof,
                    "loss": metrics.loss_per_candidate[local_idx].detach(),
                    "ggik_success": metrics.pose_success_rate[local_idx].detach(),
                    "ggik_se3": metrics.best_se3_mean[local_idx].detach(),
                    "raw_morphology": metrics.raw_morphologies[local_idx].detach(),
                    "processed_morphology": metrics.processed_morphologies[
                        local_idx
                    ].detach(),
                }
            )

    return records


def optimize_candidate_selection(
    morph: Morphology,
    task: Task,
    optimization_parameters: dict,
    *,
    default_candidate_dofs: tuple[int, ...] = DEFAULT_CANDIDATE_DOFS,
    label: str = "GGIK candidate baseline",
) -> tuple[Morphology, Path]:
    ggik_params = _ggik_parameters(optimization_parameters)
    logging = bool(optimization_parameters.get("logging", True))
    random_seed = int(optimization_parameters.get("random_seed", 42))
    candidate_dofs = _resolve_candidate_dofs(
        optimization_parameters.get("candidate_dofs", default_candidate_dofs)
    )
    num_alpha_candidates = optimization_parameters.get(
        "num_alpha_candidates",
        DEFAULT_NUM_ALPHA_CANDIDATES,
    )
    candidate_batch_size = optimization_parameters.get(
        "candidate_batch_size",
        DEFAULT_CANDIDATE_BATCH_SIZE,
    )
    distribution_batch_size = int(
        optimization_parameters.get(
            "distribution_batch_size",
            DEFAULT_DISTRIBUTION_BATCH_SIZE,
        )
    )
    top_fraction = float(
        optimization_parameters.get("top_fraction", DEFAULT_TOP_FRACTION)
    )
    max_attempts_per_alpha = int(
        optimization_parameters.get(
            "max_attempts_per_alpha",
            DEFAULT_FIXED_ALPHA_RANDOM_TRIES,
        )
    )
    dynamic_rejection_batch_size = int(
        optimization_parameters.get(
            "dynamic_rejection_batch_size",
            DEFAULT_DYNAMIC_REJECTION_BATCH_SIZE,
        )
    )
    use_cached_initial_candidates = bool(
        optimization_parameters.get("use_cached_initial_candidates", True)
    )

    if isinstance(candidate_batch_size, str):
        normalized_candidate_batch_size = candidate_batch_size.strip().upper()
        if normalized_candidate_batch_size not in {
            "ALL",
            "AUTO",
            "GGIK_FORWARD",
            "FORWARD",
        }:
            candidate_batch_size = int(candidate_batch_size)
    else:
        candidate_batch_size = int(candidate_batch_size)
    if isinstance(candidate_batch_size, int) and candidate_batch_size <= 0:
        raise ValueError(
            "candidate_batch_size must be positive, 'ALL', or 'GGIK_FORWARD'."
        )
    if distribution_batch_size <= 0:
        raise ValueError("distribution_batch_size must be positive.")
    if not (0.0 < top_fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1].")

    device = morph.params.device
    dtype = morph.params.dtype
    target_poses_local = task.goal_poses.to(device=device, dtype=dtype)
    solver = _build_ggik_solver(optimization_parameters, device=device, dtype=dtype)
    csv_logger = OptimizationCSVLogger(root_dir=PROJECT_ROOT)

    alpha_generator = torch.Generator(device=device)
    alpha_generator.manual_seed(random_seed)

    if logging:
        print(f"[Info] Starting {label} on device {device}.")
        print(f"[Info] Writing CSV log to: {csv_logger.csv_path}")

    try:
        alpha_candidates_by_dof = _generate_alpha_candidates_by_dof(
            dofs=candidate_dofs,
            requested_num_candidates=num_alpha_candidates,
            device=device,
            generator=alpha_generator,
            logging=logging,
        )

        if use_cached_initial_candidates:
            initial_candidate_morphologies_by_dof = (
                sample_fixed_alpha_morphology_candidates_by_dof(
                    alpha_candidates_by_dof=alpha_candidates_by_dof,
                    seed=random_seed,
                    link_radius=morph.link_radius,
                    batch_size=distribution_batch_size,
                    max_attempts_per_alpha=max_attempts_per_alpha,
                    dynamic_rejection_batch_size=dynamic_rejection_batch_size,
                    logging=logging,
                )
            )
        else:
            initial_candidate_morphologies_by_dof = (
                _sample_fixed_alpha_morphology_candidates_by_dof_no_cache(
                    alpha_candidates_by_dof=alpha_candidates_by_dof,
                    seed=random_seed,
                    link_radius=morph.link_radius,
                    batch_size=distribution_batch_size,
                    max_attempts_per_alpha=max_attempts_per_alpha,
                    dynamic_rejection_batch_size=dynamic_rejection_batch_size,
                    logging=logging,
                )
            )

        records: list[dict[str, Any]] = []
        for dof in candidate_dofs:
            records.extend(
                _optimize_one_dof_group(
                    dof=dof,
                    initial_candidate_morphologies=initial_candidate_morphologies_by_dof[
                        dof
                    ],
                    target_poses_local=target_poses_local,
                    link_radius=morph.link_radius,
                    solver=solver,
                    ggik_params=ggik_params,
                    candidate_batch_size=candidate_batch_size,
                    distribution_batch_size=distribution_batch_size,
                    random_seed=random_seed,
                    logging=logging,
                )
            )

        if not records:
            raise RuntimeError(
                "All GGIK candidate morphologies were rejected by the "
                "post-optimization distribution checker."
            )

        records = sorted(records, key=_candidate_sort_key)
        top_k = max(1, int(math.ceil(len(records) * top_fraction)))
        top_records = records[:top_k]

        tie_scores = []
        length_sums = []
        for record in top_records:
            tie_score, length_sum = _tie_score(
                record["processed_morphology"],
                morph.link_radius,
            )
            tie_scores.append(tie_score)
            length_sums.append(length_sum)

        length_sums_tensor = torch.stack(length_sums).to(device)
        ggik_se3_tensor = torch.stack(
            [record["ggik_se3"].to(device) for record in top_records]
        )
        best_ggik_se3 = ggik_se3_tensor.min()
        final_tier_mask = (ggik_se3_tensor - best_ggik_se3).abs() <= SE3_TIE_EPS
        tied_indices = torch.nonzero(final_tier_mask, as_tuple=False).squeeze(1)

        tie_scores_tensor = torch.stack(tie_scores).to(device)
        final_local_in_tied = torch.argmin(tie_scores_tensor[tied_indices])
        final_idx = int(tied_indices[final_local_in_tied].item())

        for idx, record in enumerate(top_records):
            if idx == final_idx:
                marker = 2
            elif bool(final_tier_mask[idx]):
                marker = 1
            else:
                marker = 0

            csv_logger.log_iteration(
                iteration=marker,
                loss=record["loss"],
                reachability_probability=record["ggik_success"],
                raw_morphology=record["raw_morphology"],
                processed_morphology=record["processed_morphology"],
                validation_data={
                    "ik_success_pose_rate": record["ggik_success"],
                    "best_se3_dist_mean": record["ggik_se3"],
                },
            )

        final_record = top_records[final_idx]
        print(
            f"[Final {label}] "
            f"dof={final_record['dof']}, "
            f"loss={final_record['loss'].item():.6f}, "
            f"ggik_success={final_record['ggik_success'].item() * 100.0:.2f}%, "
            f"ggik_se3={final_record['ggik_se3'].item():.6f}, "
            f"length_sum={length_sums_tensor[final_idx].item():.6f}, "
            f"top_k={top_k}"
        )

        return (
            Morphology(
                params=final_record["processed_morphology"].detach(),
                link_radius=morph.link_radius,
            ),
            csv_logger.csv_path,
        )
    finally:
        csv_logger.close()
