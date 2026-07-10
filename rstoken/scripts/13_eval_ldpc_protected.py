"""E23 LDPC-protected RS-Token and classic compressed-bitstream baselines.

Main comparison uses fixed transmitted bits per image. For an LDPC rate R=k/n,
source_bits = total_bits * R. RS-Token uses source_bits/2560 RVQ layers, while
WebP/JPEG2000 are encoded to the same source-bit budget before LDPC protection.

The LDPC implementation is a deterministic systematic sparse parity-check code
H=[P I]. It is intended for reproducible paper experiments without external
communication-toolbox dependencies.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import io
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_script_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    old_platform = sys.platform
    try:
        if relative_path == "scripts/10_eval_classic_baselines.py":
            sys.platform = "linux"
        spec.loader.exec_module(module)
    finally:
        sys.platform = old_platform
    return module


classic = None
rvq_eval = None


def ensure_eval_modules():
    global classic, rvq_eval
    if classic is None:
        classic = load_script_module("rstoken_classic_baselines", "scripts/10_eval_classic_baselines.py")
    if rvq_eval is None:
        rvq_eval = load_script_module("rstoken_recon_task_split", "scripts/09_eval_rvqs_recon_task_split.py")
    return classic, rvq_eval


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ber_from_snr(snr_db: float, channel: str) -> float:
    if channel == "none":
        return 0.0
    snr_lin = 10 ** (snr_db / 10.0)
    if channel == "awgn":
        return 0.5 * math.erfc(math.sqrt(snr_lin))
    if channel == "rayleigh":
        return 0.5 * (1.0 - math.sqrt(snr_lin / (1.0 + snr_lin)))
    raise ValueError(f"unknown channel: {channel}")


def bytes_to_bits(payload: bytes) -> np.ndarray:
    if not payload:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(payload, dtype=np.uint8)
    return np.unpackbits(arr, bitorder="big").astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    if bits.size % 8:
        pad = 8 - bits.size % 8
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits, bitorder="big").tobytes()


@dataclass
class PackedPayload:
    bits: np.ndarray
    original_num_bytes: int
    pad_bits: int


def pack_payload_bits(payload: bytes, source_bits: int) -> PackedPayload:
    bits = bytes_to_bits(payload)
    if bits.size > source_bits:
        raise ValueError(f"payload has {bits.size} bits, exceeds source_bits={source_bits}")
    pad_bits = source_bits - bits.size
    if pad_bits:
        bits = np.concatenate([bits, np.zeros(pad_bits, dtype=np.uint8)])
    return PackedPayload(bits=bits.astype(np.uint8), original_num_bytes=len(payload), pad_bits=pad_bits)


def unpack_payload_bits(bits: np.ndarray, original_num_bytes: int) -> bytes:
    needed = original_num_bytes * 8
    return bits_to_bytes(np.asarray(bits[:needed], dtype=np.uint8))[:original_num_bytes]


class SystematicLDPC:
    """Sparse systematic LDPC code with H=[P I]."""

    def __init__(self, source_bits: int, rate_num: int = 1, rate_den: int = 2, col_weight: int = 3, seed: int = 0):
        if rate_num <= 0 or rate_den <= rate_num:
            raise ValueError("LDPC rate must be between 0 and 1")
        if source_bits % rate_num != 0:
            raise ValueError(f"source_bits={source_bits} must be divisible by rate numerator={rate_num}")
        self.k = int(source_bits)
        self.n = int(source_bits * rate_den // rate_num)
        self.m = self.n - self.k
        self.rate_num = rate_num
        self.rate_den = rate_den
        self.col_weight = int(col_weight)
        rng = np.random.default_rng(seed)
        self.data_checks: list[np.ndarray] = []
        self.check_vars: list[list[int]] = [[] for _ in range(self.m)]
        for var in range(self.k):
            degree = min(self.col_weight, self.m)
            checks = np.sort(rng.choice(self.m, size=degree, replace=False)).astype(np.int32)
            self.data_checks.append(checks)
            for check in checks:
                self.check_vars[int(check)].append(var)
        for check in range(self.m):
            self.check_vars[check].append(self.k + check)
        self.var_edges: list[list[tuple[int, int]]] = [[] for _ in range(self.n)]
        self.edge_check: list[int] = []
        self.edge_var: list[int] = []
        for check, variables in enumerate(self.check_vars):
            for local_idx, var in enumerate(variables):
                edge_idx = len(self.edge_check)
                self.edge_check.append(check)
                self.edge_var.append(var)
                self.var_edges[var].append((edge_idx, check))
        self.num_edges = len(self.edge_check)

    def encode(self, data_bits: np.ndarray) -> np.ndarray:
        data = np.asarray(data_bits, dtype=np.uint8)
        if data.size != self.k:
            raise ValueError(f"expected {self.k} source bits, got {data.size}")
        parity = np.zeros(self.m, dtype=np.uint8)
        one_positions = np.flatnonzero(data)
        for var in one_positions:
            parity[self.data_checks[int(var)]] ^= 1
        return np.concatenate([data, parity])

    def syndrome_ok(self, bits: np.ndarray) -> bool:
        bits = np.asarray(bits, dtype=np.uint8)
        if bits.size != self.n:
            return False
        for check, variables in enumerate(self.check_vars):
            if int(np.bitwise_xor.reduce(bits[variables])) != 0:
                return False
        return True

    def decode(self, llr: np.ndarray, max_iter: int = 30, damping: float = 0.0) -> tuple[np.ndarray, bool, int]:
        llr = np.asarray(llr, dtype=np.float64)
        if llr.size != self.n:
            raise ValueError(f"expected {self.n} LLRs, got {llr.size}")
        q = np.zeros(self.num_edges, dtype=np.float64)
        r = np.zeros(self.num_edges, dtype=np.float64)
        for edge_idx, var in enumerate(self.edge_var):
            q[edge_idx] = llr[var]
        hard = (llr < 0).astype(np.uint8)
        if self.syndrome_ok(hard):
            return hard[: self.k].copy(), True, 0
        for iteration in range(1, max_iter + 1):

            for check, variables in enumerate(self.check_vars):
                edge_indices = []
                for var in variables:
                    for edge_idx, edge_check in self.var_edges[var]:
                        if edge_check == check:
                            edge_indices.append(edge_idx)
                            break
                values = q[edge_indices]
                signs = np.where(values >= 0, 1.0, -1.0)
                abs_values = np.abs(values)
                if len(abs_values) == 1:
                    outgoing = np.array([0.0])
                else:
                    min1_idx = int(np.argmin(abs_values))
                    min1 = abs_values[min1_idx]
                    tmp = abs_values.copy()
                    tmp[min1_idx] = np.inf
                    min2 = float(np.min(tmp))
                    sign_product = float(np.prod(signs))
                    outgoing = np.empty_like(values)
                    for pos in range(len(values)):
                        mag = min2 if pos == min1_idx else min1
                        outgoing[pos] = sign_product * signs[pos] * mag
                if damping:
                    r[edge_indices] = (1.0 - damping) * outgoing + damping * r[edge_indices]
                else:
                    r[edge_indices] = outgoing
            posterior = llr.copy()
            for var in range(self.n):
                for edge_idx, _ in self.var_edges[var]:
                    posterior[var] += r[edge_idx]
            hard = (posterior < 0).astype(np.uint8)
            if self.syndrome_ok(hard):
                return hard[: self.k].copy(), True, iteration
            for var in range(self.n):
                total = posterior[var]
                for edge_idx, _ in self.var_edges[var]:
                    q[edge_idx] = total - r[edge_idx]
        return hard[: self.k].copy(), False, max_iter


def bits_to_llr(bits: np.ndarray, error_probability: float) -> np.ndarray:
    p = min(max(float(error_probability), 1e-9), 1.0 - 1e-9)
    magnitude = math.log((1.0 - p) / p)
    return np.where(np.asarray(bits, dtype=np.uint8) == 0, magnitude, -magnitude).astype(np.float64)


def transmit_bpsk_llr(codeword_bits: np.ndarray, channel: str, snr: str, seed: int) -> tuple[np.ndarray, float]:
    if channel == "none" or snr == "inf":
        raw_ber = 0.0
    else:
        raw_ber = ber_from_snr(float(snr), channel)
    rng = np.random.default_rng(seed)
    received = np.asarray(codeword_bits, dtype=np.uint8).copy()
    if raw_ber > 0:
        received ^= (rng.random(received.shape) < raw_ber).astype(np.uint8)
    return bits_to_llr(received, max(raw_ber, 1e-9)), raw_ber



def stratified_subset_indices(dataset, max_samples: int) -> list[int]:
    if max_samples >= len(dataset):
        return list(range(len(dataset)))
    labels = []
    for idx in range(len(dataset)):
        item = getattr(dataset, "items", None)
        if item is not None:
            labels.append(int(item[idx][1]))
        else:
            _, label = dataset[idx]
            labels.append(int(label))
    by_label: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_label.setdefault(label, []).append(idx)
    chosen: list[int] = []
    label_order = sorted(by_label)
    cursor = 0
    while len(chosen) < max_samples and label_order:
        label = label_order[cursor % len(label_order)]
        bucket = by_label[label]
        if bucket:
            chosen.append(bucket.pop(0))
        if not bucket:
            label_order.remove(label)
            cursor = 0
        else:
            cursor += 1
    return sorted(chosen)

def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_strings(value: str) -> list[str]:
    return [x.strip().lower() for x in value.split(",") if x.strip()]


def condition_seed(base_seed: int, *parts: object) -> int:
    text = "|".join(str(p) for p in parts)
    acc = int(base_seed) & 0xFFFFFFFF
    for ch in text.encode("utf-8"):
        acc = ((acc * 16777619) ^ ch) & 0xFFFFFFFF
    return acc


def protect_and_decode_bits(source_bits_arr: np.ndarray, code: SystematicLDPC, channel: str, snr: str, seed: int, max_iter: int) -> tuple[np.ndarray, dict]:
    codeword = code.encode(source_bits_arr)
    llr, raw_ber = transmit_bpsk_llr(codeword, channel, snr, seed)
    decoded, success, iterations = code.decode(llr, max_iter=max_iter)
    post_ber = float(np.mean(decoded != source_bits_arr)) if source_bits_arr.size else 0.0
    return decoded, {
        "raw_ber": raw_ber,
        "post_ldpc_ber": post_ber,
        "ldpc_success": bool(success),
        "ldpc_iterations": iterations,
    }



class TorchLDPCBP:
    """Batched min-sum LDPC decoder on torch/CUDA."""

    def __init__(self, code: SystematicLDPC, device: str):
        self.code = code
        self.device = torch.device(device)
        self.k = code.k
        self.n = code.n
        self.m = code.m
        self.edge_var = torch.tensor(code.edge_var, dtype=torch.long, device=self.device)
        check_edges: list[list[int]] = [[] for _ in range(code.m)]
        for edge_idx, check in enumerate(code.edge_check):
            check_edges[int(check)].append(edge_idx)
        self.max_check_degree = max(len(edges) for edges in check_edges)
        check_edge_padded = torch.full((code.m, self.max_check_degree), -1, dtype=torch.long)
        check_mask = torch.zeros((code.m, self.max_check_degree), dtype=torch.bool)
        for check, edges in enumerate(check_edges):
            check_edge_padded[check, : len(edges)] = torch.tensor(edges, dtype=torch.long)
            check_mask[check, : len(edges)] = True
        self.check_edge_padded = check_edge_padded.to(self.device)
        self.check_mask = check_mask.to(self.device)
        var_edges = code.var_edges
        self.max_var_degree = max(len(edges) for edges in var_edges)
        var_edge_padded = torch.full((code.n, self.max_var_degree), -1, dtype=torch.long)
        var_mask = torch.zeros((code.n, self.max_var_degree), dtype=torch.bool)
        for var, edges in enumerate(var_edges):
            edge_ids = [edge_idx for edge_idx, _ in edges]
            var_edge_padded[var, : len(edge_ids)] = torch.tensor(edge_ids, dtype=torch.long)
            var_mask[var, : len(edge_ids)] = True
        self.var_edge_padded = var_edge_padded.to(self.device)
        self.var_mask = var_mask.to(self.device)
        checks = np.stack(code.data_checks, axis=0)
        self.data_checks = torch.tensor(checks, dtype=torch.long, device=self.device)

    def encode(self, data_bits: torch.Tensor) -> torch.Tensor:
        data = data_bits.to(self.device).bool()
        batch = data.shape[0]
        counts = torch.zeros((batch, self.m), dtype=torch.int16, device=self.device)
        index = self.data_checks.unsqueeze(0).expand(batch, -1, -1).reshape(batch, -1)
        src = data.to(torch.int16).unsqueeze(2).expand(-1, -1, self.data_checks.shape[1]).reshape(batch, -1)
        counts.scatter_add_(1, index, src)
        parity = (counts & 1).bool()
        return torch.cat([data, parity], dim=1)

    def syndrome_ok(self, bits: torch.Tensor) -> torch.Tensor:
        edge_bits = bits[:, self.edge_var]
        gathered = edge_bits[:, self.check_edge_padded.clamp_min(0)]
        gathered = gathered & self.check_mask.unsqueeze(0)
        syndrome = (gathered.to(torch.int16).sum(dim=2) & 1).bool()
        return ~syndrome.any(dim=1)

    def transmit_decode(self, data_bits_np: np.ndarray, channel: str, snr: str, seed: int, max_iter: int) -> tuple[np.ndarray, dict]:
        data = torch.as_tensor(data_bits_np, dtype=torch.bool, device=self.device)
        if data.dim() == 1:
            data = data.unsqueeze(0)
        codeword = self.encode(data)
        raw_ber = 0.0 if channel == "none" or snr == "inf" else ber_from_snr(float(snr), channel)
        received = codeword.clone()
        if raw_ber > 0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed) & 0xFFFFFFFF)
            flips = torch.rand(received.shape, device=self.device, generator=generator) < raw_ber
            received = received ^ flips
        p = min(max(float(raw_ber), 1e-9), 1.0 - 1e-9)
        magnitude = math.log((1.0 - p) / p)
        llr = torch.where(received, -magnitude, magnitude).to(torch.float32)
        batch = llr.shape[0]
        q = llr[:, self.edge_var].clone()
        r = torch.zeros_like(q)
        success_iter = torch.full((batch,), max_iter, dtype=torch.int16, device=self.device)
        active = torch.ones((batch,), dtype=torch.bool, device=self.device)
        hard = received.clone()
        for iteration in range(max_iter + 1):
            if iteration > 0:
                check_edges = self.check_edge_padded.clamp_min(0)
                values = q[:, check_edges]
                mask = self.check_mask.unsqueeze(0)
                signs = torch.where(values >= 0, 1.0, -1.0)
                signs = torch.where(mask, signs, 1.0)
                abs_values = torch.where(mask, values.abs(), torch.full_like(values, float("inf")))
                min1, min1_idx = abs_values.min(dim=2, keepdim=True)
                abs_for_min2 = abs_values.scatter(2, min1_idx, float("inf"))
                min2 = abs_for_min2.min(dim=2, keepdim=True).values
                sign_product = signs.prod(dim=2, keepdim=True)
                positions = torch.arange(self.max_check_degree, device=self.device).view(1, 1, -1)
                magnitude_out = torch.where(positions == min1_idx, min2, min1)
                outgoing = sign_product * signs * magnitude_out
                outgoing = torch.where(mask, outgoing, torch.zeros_like(outgoing))
                flat_edges = self.check_edge_padded.reshape(-1)
                flat_mask = flat_edges >= 0
                r[:, flat_edges[flat_mask]] = outgoing.reshape(batch, -1)[:, flat_mask]
                posterior = llr.clone()
                posterior.scatter_add_(1, self.edge_var.unsqueeze(0).expand(batch, -1), r)
                hard = posterior < 0
            ok = self.syndrome_ok(hard)
            newly_ok = active & ok
            success_iter[newly_ok] = iteration
            active = active & ~ok
            if not active.any() or iteration == max_iter:
                break
            posterior_edges = (llr.clone().scatter_add_(1, self.edge_var.unsqueeze(0).expand(batch, -1), r))[:, self.edge_var]
            q = posterior_edges - r
        decoded = hard[:, : self.k]
        post_ber = (decoded != data).to(torch.float32).mean(dim=1)
        return decoded.detach().cpu().numpy().astype(np.uint8), {
            "raw_ber": raw_ber,
            "post_ldpc_ber": post_ber.detach().cpu().numpy(),
            "ldpc_success": self.syndrome_ok(hard).detach().cpu().numpy().astype(bool),
            "ldpc_iterations": success_iter.detach().cpu().numpy().astype(np.int32),
        }


def indices_to_bits(indices: torch.Tensor, k_layers: int, bits_per_index: int = 10) -> np.ndarray:
    arr = indices[..., :k_layers].detach().cpu().numpy().astype(np.uint16).reshape(-1)
    shifts = np.arange(bits_per_index - 1, -1, -1, dtype=np.uint16)
    return ((arr[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def bits_to_indices(bits: np.ndarray, shape: tuple[int, ...], k_layers: int, bits_per_index: int = 10) -> torch.Tensor:
    bits = np.asarray(bits, dtype=np.uint8)[: int(np.prod(shape[:-1])) * k_layers * bits_per_index]
    mat = bits.reshape(-1, bits_per_index)
    weights = (1 << np.arange(bits_per_index - 1, -1, -1, dtype=np.uint16)).astype(np.uint16)
    values = (mat * weights).sum(axis=1).astype(np.int64)
    return torch.from_numpy(values.reshape(*shape[:-1], k_layers))


def classifier_input_from_decoded(decoded_images: list[torch.Tensor], device: str) -> torch.Tensor:
    batch = torch.stack(decoded_images).to(device)
    return classic.neg11_to_classifier_input(batch)


def evaluate_classic_method(method: str, samples: list, classifier, lpips_fn, source_bits: int, total_bits: int, rate_label: str, channels: list[str], snrs: list[str], channel_seeds: list[int], image_size: int, device: str, max_iter: int, ldpc_seed: int) -> list[dict]:
    print(f"[classic encode] method={method} source_bits={source_bits}")
    encoded = []
    for sample in samples:
        if method == "webp":
            payload = classic.encode_webp(sample.image, source_bits)
        elif method == "jpeg2000":
            payload = classic.encode_jpeg2000_pillow(sample.image, source_bits, image_size)
        else:
            raise ValueError(method)
        encoded.append((sample, payload))
    actual_bits = [len(payload) * 8 for _, payload in encoded]
    code = SystematicLDPC(source_bits=source_bits, rate_num=1, rate_den=2, seed=ldpc_seed)
    torch_code = TorchLDPCBP(code, device)
    rows: list[dict] = []
    conditions = [(channel, snr) for channel in channels for snr in snrs]
    for channel, snr in conditions:
        for channel_seed in channel_seeds:
            failures = 0
            decoded_images: list[torch.Tensor] = []
            decoded_labels: list[int] = []
            psnr_values: list[float] = []
            lpips_values: list[float] = []
            packed_items = []
            oversize_failures = 0
            for sample, payload in encoded:
                try:
                    packed_items.append((sample, pack_payload_bits(payload, source_bits)))
                except ValueError:
                    oversize_failures += 1
            if packed_items:
                packed_matrix = np.stack([packed.bits for _, packed in packed_items], axis=0)
                seed = condition_seed(channel_seed, method, source_bits, channel, snr, "batch")
                decoded_matrix, meta = torch_code.transmit_decode(packed_matrix, channel, snr, seed, max_iter)
                post_bers_arr = meta["post_ldpc_ber"]
                success_arr = meta["ldpc_success"]
                iter_arr = meta["ldpc_iterations"]
            else:
                decoded_matrix = np.zeros((0, source_bits), dtype=np.uint8)
                post_bers_arr = np.zeros(0, dtype=np.float32)
                success_arr = np.zeros(0, dtype=bool)
                iter_arr = np.zeros(0, dtype=np.int32)
            failures += oversize_failures
            for row_idx, (sample, packed) in enumerate(packed_items):
                decoded_payload = unpack_payload_bits(decoded_matrix[row_idx], packed.original_num_bytes)
                try:
                    image = Image.open(io.BytesIO(decoded_payload)).convert("RGB")
                    image = classic.ensure_image_size(image, image_size)
                    tensor = classic.pil_to_neg11(image)
                except Exception:
                    failures += 1
                    continue
                decoded_images.append(tensor)
                decoded_labels.append(sample.label)
                ref = sample.tensor_neg11.unsqueeze(0).to(device)
                rec = tensor.unsqueeze(0).to(device)
                mse = torch.mean((ref - rec) ** 2).item()
                psnr_values.append(float(10.0 * math.log10(4.0 / (mse + 1e-12))))
                lpips_values.append(float(lpips_fn(rec, ref).item()))
            post_bers = [1.0] * oversize_failures + [float(x) for x in post_bers_arr.tolist()]
            successes = int(np.asarray(success_arr).sum())
            iter_values = [int(x) for x in np.asarray(iter_arr).tolist()]
            cls_acc_all = 0.0
            cls_acc_decoded = ""
            if decoded_images and classifier is not None:
                correct = 0
                total = 0
                for start in range(0, len(decoded_images), 64):
                    x = classifier_input_from_decoded(decoded_images[start:start + 64], device)
                    y = torch.tensor(decoded_labels[start:start + 64], device=device)
                    pred = classifier(x).argmax(dim=1)
                    correct += int((pred == y).sum().item())
                    total += int(y.numel())
                cls_acc_decoded_float = correct / max(total, 1)
                cls_acc_decoded = f"{cls_acc_decoded_float:.6f}"
                cls_acc_all = correct / max(len(samples), 1)
            rows.append({
                "method": method,
                "family": "classic_ldpc",
                "ldpc_rate": rate_label,
                "source_bits": source_bits,
                "total_bits": total_bits,
                "actual_source_bits_mean": f"{statistics.mean(actual_bits):.3f}",
                "actual_source_bits_std": f"{statistics.pstdev(actual_bits):.3f}",
                "channel": channel,
                "snr": snr,
                "channel_seed": channel_seed,
                "raw_ber": f"{ber_from_snr(float(snr), channel):.8f}",
                "post_ldpc_ber": f"{statistics.mean(post_bers):.8f}" if post_bers else "",
                "ldpc_success_rate": f"{successes / max(len(encoded), 1):.6f}",
                "ldpc_iterations_mean": f"{statistics.mean(iter_values):.3f}" if iter_values else "",
                "decode_failure_rate": f"{failures / max(len(samples), 1):.6f}",
                "cls_acc_all": f"{cls_acc_all:.6f}",
                "cls_acc_decoded": cls_acc_decoded,
                "psnr_valid": f"{statistics.mean(psnr_values):.6f}" if psnr_values else "",
                "lpips_valid": f"{statistics.mean(lpips_values):.6f}" if lpips_values else "",
                "num_samples": len(samples),
            })
            print(f"[classic] {method} bits={total_bits} {channel} {snr} seed={channel_seed} fail={rows[-1]['decode_failure_rate']} cls={rows[-1]['cls_acc_all']}")
    return rows


def evaluate_rstoken(model, model_cfg, train_idx, train_labels, test_idx, test_labels, test_images, spatial, classifier, lpips_fn, source_bits: int, total_bits: int, rate_label: str, channels: list[str], snrs: list[str], channel_seeds: list[int], device: str, max_iter: int, ldpc_seed: int, batch_size: int) -> list[dict]:
    if source_bits % 2560 != 0:
        print(f"[rstoken] skip source_bits={source_bits}: not a multiple of one RVQ layer")
        return []
    k_layers = source_bits // 2560
    if k_layers < 1 or k_layers > model_cfg.rvq_num_quantizers:
        print(f"[rstoken] skip source_bits={source_bits}: k={k_layers} outside model layers")
        return []
    code = SystematicLDPC(source_bits=source_bits, rate_num=1, rate_den=2, seed=ldpc_seed)
    torch_code = TorchLDPCBP(code, device)
    source_bits_arr = indices_to_bits(test_idx, k_layers)
    if source_bits_arr.size != source_bits * test_idx.shape[0]:
        raise RuntimeError("unexpected RS-Token bitstream length")
    scaler = clf = None
    if k_layers == 1:
        scaler, clf = rvq_eval.fit_h0_probe(train_idx[..., :1], train_labels, model_cfg.codebook_size)
    rows: list[dict] = []
    for channel in channels:
        for snr in snrs:
            for channel_seed in channel_seeds:
                bit_matrix = source_bits_arr.reshape(test_idx.shape[0], source_bits)
                seed = condition_seed(channel_seed, "rstoken", source_bits, channel, snr, "batch")
                decoded_matrix, meta = torch_code.transmit_decode(bit_matrix, channel, snr, seed, max_iter)
                post_bers = [float(x) for x in meta["post_ldpc_ber"].tolist()]
                successes = int(np.asarray(meta["ldpc_success"]).sum())
                iter_values = [int(x) for x in np.asarray(meta["ldpc_iterations"]).tolist()]
                decoded = bits_to_indices(decoded_matrix.reshape(-1), tuple(test_idx[..., :k_layers].shape), k_layers).long()
                invalid = int(((decoded < 0) | (decoded >= model_cfg.codebook_size)).sum().item())
                h0_acc = ""
                if k_layers == 1 and scaler is not None and clf is not None:
                    h0_acc = f"{rvq_eval.score_h0_probe(scaler, clf, decoded, test_labels, model_cfg.codebook_size):.6f}"
                psnr, lpips, cls_acc = rvq_eval.reconstruction_metrics(
                    model=model,
                    indices=decoded,
                    x_ref=test_images,
                    labels=test_labels,
                    spatial=spatial,
                    lpips_fn=lpips_fn,
                    classifier=classifier,
                    batch_size=batch_size,
                    device=device,
                )
                rows.append({
                    "method": "rstoken",
                    "family": "rstoken_ldpc",
                    "ldpc_rate": rate_label,
                    "source_bits": source_bits,
                    "total_bits": total_bits,
                    "k": k_layers,
                    "actual_source_bits_mean": f"{source_bits:.3f}",
                    "actual_source_bits_std": "0.000",
                    "channel": channel,
                    "snr": snr,
                    "channel_seed": channel_seed,
                    "raw_ber": f"{ber_from_snr(float(snr), channel):.8f}",
                    "post_ldpc_ber": f"{statistics.mean(post_bers):.8f}",
                    "ldpc_success_rate": f"{successes / max(test_idx.shape[0], 1):.6f}",
                    "ldpc_iterations_mean": f"{statistics.mean(iter_values):.3f}",
                    "invalid_index_rate": f"{invalid / max(test_idx.shape[0] * 256 * k_layers, 1):.8f}",
                    "decode_failure_rate": "0.000000",
                    "h0_acc": h0_acc,
                    "recon_cls_acc": "" if cls_acc is None else f"{cls_acc:.6f}",
                    "psnr": f"{psnr:.6f}",
                    "lpips": f"{lpips:.6f}",
                    "num_samples": int(test_idx.shape[0]),
                })
                print(f"[rstoken] total={total_bits} k={k_layers} {channel} {snr} seed={channel_seed} h0={h0_acc or 'NA'} cls={rows[-1]['recon_cls_acc']}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"no rows to write: {path}")
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_eval_modules()
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", default="rstoken,jpeg2000,webp")
    parser.add_argument("--total_bits", default="5120,10240,20480")
    parser.add_argument("--channels", default="awgn,rayleigh")
    parser.add_argument("--snrs", default="-10,-5,0,5,10")
    parser.add_argument("--channel_seeds", default="0,1,2,3,4")
    parser.add_argument("--ldpc_rate", default="1/2", choices=["1/2"])
    parser.add_argument("--ldpc_seed", type=int, default=2026)
    parser.add_argument("--max_iter", type=int, default=30)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits_dir", default="data/AID_splits")
    parser.add_argument("--classifier_ckpt", default="checkpoints/aid_classifier_resnet34/best.pt")
    parser.add_argument("--rstoken_ckpt", default="checkpoints/rvq_distill/best.pt")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_csv", default="logs/e23_ldpc_protected.csv")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; falling back to CPU")
        args.device = "cpu"
    device = args.device
    methods = parse_strings(args.methods)
    total_bits_values = parse_ints(args.total_bits)
    channels = parse_strings(args.channels)
    snrs = [s.strip() for s in args.snrs.split(",") if s.strip()]
    channel_seeds = parse_ints(args.channel_seeds)
    rate_num, rate_den = (int(x) for x in args.ldpc_rate.split("/"))
    rate_label = args.ldpc_rate

    classifier, classifier_note = classic.load_classifier(project_path(args.classifier_ckpt), device)
    if classifier_note:
        print(f"[classifier] {classifier_note}")
    else:
        print(f"[classifier] loaded {args.classifier_ckpt}")
    lpips_fn = rvq_eval.LPIPSLoss(net="alex").to(device).eval()

    rows: list[dict] = []
    samples = None
    if any(method in methods for method in ["jpeg2000", "webp"]):
        samples = classic.read_split(project_path(args.splits_dir) / "test.csv", args.image_size, args.max_samples)
        print(f"[classic data] samples={len(samples)}")

    rstoken_state = None
    if "rstoken" in methods:
        print(f"[rstoken] loading {args.rstoken_ckpt}")
        model, cfg, model_cfg = rvq_eval.load_rvq_model(project_path(args.rstoken_ckpt), device)
        data_cfg = rvq_eval.AIDConfig(**cfg["data"])
        loaders = rvq_eval.build_loaders(data_cfg)
        tr_loader = rvq_eval.train_eval_loader(data_cfg)
        if args.max_samples is not None:
            train_n = min(args.max_samples, len(tr_loader.dataset))
            test_n = min(args.max_samples, len(loaders["test"].dataset))
            tr_loader = torch.utils.data.DataLoader(
                torch.utils.data.Subset(tr_loader.dataset, stratified_subset_indices(tr_loader.dataset, train_n)),
                batch_size=data_cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
            loaders["test"] = torch.utils.data.DataLoader(
                torch.utils.data.Subset(loaders["test"].dataset, stratified_subset_indices(loaders["test"].dataset, test_n)),
                batch_size=data_cfg.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )
        tr_idx, y_tr, _, spatial = rvq_eval.encode_indices(model, tr_loader, device, keep_images=False)
        te_idx, y_te, x_te, _ = rvq_eval.encode_indices(model, loaders["test"], device, keep_images=True)
        if args.max_samples is not None:
            te_idx = te_idx[: args.max_samples]
            y_te = y_te[: args.max_samples]
            x_te = x_te[: args.max_samples]
        rstoken_state = (model, model_cfg, tr_idx, y_tr, te_idx, y_te, x_te, spatial)
        print(f"[rstoken data] train={tr_idx.shape[0]} test={te_idx.shape[0]}")

    out_path = project_path(args.out_csv)
    t0 = time.time()
    for total_bits in total_bits_values:
        source_bits = total_bits * rate_num // rate_den
        if source_bits * rate_den != total_bits * rate_num:
            raise ValueError(f"total_bits={total_bits} incompatible with rate={args.ldpc_rate}")
        print("\n" + "=" * 80)
        print(f"[budget] total_bits={total_bits} source_bits={source_bits} rate={rate_label}")
        print("=" * 80)
        if "rstoken" in methods and rstoken_state is not None:
            new_rows = evaluate_rstoken(*rstoken_state, classifier, lpips_fn, source_bits, total_bits, rate_label, channels, snrs, channel_seeds, device, args.max_iter, args.ldpc_seed, args.batch_size)
            rows.extend(new_rows)
            write_csv(out_path, rows)
            print(f"[checkpoint] wrote {len(rows)} rows to {out_path}", flush=True)
        for method in [m for m in methods if m in {"jpeg2000", "webp"}]:
            assert samples is not None
            new_rows = evaluate_classic_method(method, samples, classifier, lpips_fn, source_bits, total_bits, rate_label, channels, snrs, channel_seeds, args.image_size, device, args.max_iter, args.ldpc_seed)
            rows.extend(new_rows)
            write_csv(out_path, rows)
            print(f"[checkpoint] wrote {len(rows)} rows to {out_path}", flush=True)

    write_csv(out_path, rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")
    print(f"Elapsed {(time.time() - t0) / 60:.1f} min")



if __name__ == "__main__":
    main()






