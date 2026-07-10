from __future__ import annotations

from dataclasses import asdict, dataclass


BITS_PER_LAYER = 2560


@dataclass(frozen=True)
class PolicyDecision:
    k: int
    quality: str
    reason: str
    source_bits: int
    transmitted_bits: int
    budget_limited: bool
    channel_limited: bool
    next_threshold_db: float | None
    stabilized: bool

    def to_dict(self) -> dict:
        return asdict(self)


# Thresholds are initialized from E17/E36. They are deliberately channel- and
# protection-specific: Rayleigh fading needs a wider safety margin, while the
# rate-1/2 LDPC path can sustain useful enhancement layers at lower SNR.
_THRESHOLDS = {
    ("awgn", "none"): (1.5, 4.0, 6.5),
    ("rayleigh", "none"): (6.5, 9.5, 13.0),
    ("awgn", "ldpc"): (-0.5, 1.5, 3.5),
    ("rayleigh", "ldpc"): (3.5, 7.0, 10.0),
}


def _channel_k(channel: str, protection: str, snr_db: float) -> tuple[int, float | None]:
    if channel == "none":
        return 4, None
    thresholds = _THRESHOLDS[(channel, protection)]
    for idx, threshold in enumerate(thresholds, start=1):
        if snr_db < threshold:
            return idx, threshold
    return 4, None


def recommend_k(
    *,
    channel: str,
    snr_db: float,
    protection: str = "none",
    priority: str = "balanced",
    max_transmitted_bits: int = 20480,
    manual_k: int | None = None,
    previous_k: int | None = None,
    hysteresis_db: float = 0.6,
) -> PolicyDecision:
    channel = channel.lower()
    protection = protection.lower()
    priority = priority.lower()
    if channel not in {"none", "awgn", "rayleigh"}:
        raise ValueError(f"unsupported channel: {channel}")
    if protection not in {"none", "ldpc"}:
        raise ValueError(f"unsupported protection: {protection}")
    if priority not in {"alert", "balanced", "detail"}:
        raise ValueError(f"unsupported priority: {priority}")

    expansion = 2 if protection == "ldpc" else 1
    budget_k = max(1, min(4, int(max_transmitted_bits) // (BITS_PER_LAYER * expansion)))
    channel_k, next_threshold = _channel_k(channel, protection, float(snr_db))

    stabilized = False
    if manual_k is not None:
        selected = max(1, min(4, int(manual_k), budget_k))
        reason = f"手动锁定 k={manual_k}，并按当前每帧预算校验。"
        channel_limited = False
    elif priority == "alert":
        selected = 1
        reason = "当前目标为语义告警；L0 已承载场景语义，不发送额外重建层。"
        channel_limited = channel_k == 1
    else:
        requested = channel_k
        if priority == "detail" and channel != "none":
            requested = min(4, requested + 1)
        if previous_k is not None and priority == "balanced" and channel != "none":
            previous = max(1, min(4, int(previous_k)))
            thresholds = _THRESHOLDS[(channel, protection)]
            adjusted = previous
            if requested > previous:
                while adjusted < requested:
                    boundary = thresholds[adjusted - 1]
                    if snr_db >= boundary + hysteresis_db:
                        adjusted += 1
                    else:
                        break
            elif requested < previous:
                while adjusted > requested:
                    boundary = thresholds[adjusted - 2]
                    if snr_db < boundary - hysteresis_db:
                        adjusted -= 1
                    else:
                        break
            stabilized = adjusted != requested
            requested = adjusted
        selected = min(requested, budget_k)
        channel_limited = selected == channel_k and channel_k < 4
        if channel == "none":
            reason = "无信道误码，发送预算允许的最大 RVQ 前缀。"
        elif selected < requested:
            reason = f"信道可支持 k={requested}，但每帧传输预算将前缀限制为 k={selected}。"
        elif stabilized:
            reason = f"SNR 位于档位切换滞回区，暂时保持 k={selected}，避免链路抖动。"
        elif next_threshold is None:
            reason = f"{channel.upper()} {snr_db:.1f} dB 已进入完整重建工作区，发送全部四层。"
        else:
            reason = (
                f"{channel.upper()} {snr_db:.1f} dB 下选择 k={selected}；"
                f"达到约 {next_threshold:.1f} dB 后再追加下一增强层。"
            )

    if selected == 1:
        quality = "语义保障"
    elif selected == 2:
        quality = "结构预览"
    elif selected == 3:
        quality = "清晰预览"
    else:
        quality = "精细复核"

    source_bits = BITS_PER_LAYER * selected
    transmitted_bits = source_bits * expansion
    return PolicyDecision(
        k=selected,
        quality=quality,
        reason=reason,
        source_bits=source_bits,
        transmitted_bits=transmitted_bits,
        budget_limited=selected == budget_k and budget_k < 4,
        channel_limited=channel_limited,
        next_threshold_db=next_threshold,
        stabilized=stabilized,
    )
