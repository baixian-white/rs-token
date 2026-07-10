# RS-Token v0.5 Review & Experiment Plan

**Manuscript**: `paper_draft/latex/rs_token_v0.5.tex` (compiled `rs_token_v0.5.pdf`, 11 pages)
**Authors**: Baohui Zhang (first), Jianwei Tai (corresponding) — School of Internet, Anhui University
**Reviewer**: Claude Opus 4.7 (in-house review)
**Date**: 2026-06-10
**Target venue**: IEEE GRSL

---

## 0. TL;DR

v0.5 is a tighter manuscript than v0.4. The L0-only distillation thesis is well-supported by the placement counterfactual (Table II), the teacher ablation (Table VII) appropriately deflates "RemoteCLIP exclusivity" to a 1.4–6.0 pp claim, and the strict same-transmitted-bit codec+LDPC stress test (Table IV) is a meaningful boundary check. The manuscript's claim discipline is unusually high.

What the paper still needs, in order of audit priority:

1. **Modern deep-JSCC baseline (NTSCC / ADJSCC / SwinJSCC)** — the only "comparable system" baseline currently in the paper is codec+LDPC. GRSL reviewers will read the absence of any deep-JSCC comparator as a missing baseline, not as a scoping decision.
2. **Continuous SNR sweep** — the paper has only 5 channel points (no-channel, AWGN ±5/+10, Rayleigh +5/+10). A waterfall curve over a denser grid (e.g. AWGN −5..+15 dB) would show *where* the distillation gain is largest and *where* RS-Token's task path actually breaks.
3. **Placement counterfactual (Table II) supplementary seeds** — the "All layers" column rests on a single seed for −1 to −9 pp deltas. The −48 pp L1-only collapse is robust; the moderate "All layers" dilution is not.
4. **WebP rate-floor disclosure** — Table IV row pairs (`WebP, k=1` and `WebP, k=2`) violate the matched-bits premise (actual source bits 14,164 vs. target 2,560 — a 5.5× overshoot). Add a `†` footnote and one sentence in §IV-E. No new experiment needed.
5. **Reconstruction-path zero-shot transfer to NWPU** — Table V only reports h₀ accuracy. PSNR/LPIPS/recon-cls under transfer is half a day's inference and would close the cross-dataset story.

Items 1–3 are new experiments; items 4–5 are inexpensive add-ons. Items 6–9 (below) are framing/Discussion fixes that do not require running anything.

---

## 1. Manuscript review (§-by-§)

### Abstract
- Sentence "RS-Token attains 0% decode-failure while a rate-1/2 LDPC-protected codec pipeline (JPEG2000, WebP) fails on 60–100% of images" is true and the experimental setup in §IV-E is symmetric. It is a feature of representation form (token-level lookup) versus form (entropy-coded container), not a bias of the test. Keep as is **only if** the WebP rate-floor footnote (item 4) lands; otherwise rephrase to "RS-Token indices remain decodable token-by-token by construction".
- Punctuation: full-width Chinese colon "：" appears at "...on the first layer only：L0 carries...". Replace with `:` (ASCII).
- "The receiver transmits a prefix of $k$ layers" — receiver does not transmit. Change to "The transmitter sends a prefix of $k$ layers".

### §I Introduction
- ¶3 currently introduces RVQ index transmission and JSCC in a single sentence. Acceptable for a Letter, but consider splitting so DeepJSCC (already cited in §II) is mentioned once in §I as the closest alternative to compress-then-transmit.

### §II Related Work
- §II-A is missing modern deep-JSCC variants (ADJSCC / NTSCC / SwinJSCC). Cite at least one in addition to the foundational `deepjscc`. This is also where the **modern-JSCC baseline** experiment (Tier-1 Item 1) gets its anchor.
- §II-B "VILA-U explores using the same discrete tokens jointly for understanding and generation" is fine but RS-Token's distinction (semantic-only on L0 vs. unified token across all layers) deserves one explicit sentence.
- §II-D "The LDPC experiment is described separately as a rate-1/2 systematic sparse LDPC code with min-sum belief-propagation decoding" — already conservative. Leave.

### §III Method
- Eq. (8) `L = L_rec + L_vq + λL_distill` does not match §IV-A which lists weights `ℓ1=1, LPIPS=0.1, RVQ=1, distill=0.5`. Either expand Eq. (8) to `L = w_ℓ1 L_ℓ1 + w_LPIPS L_LPIPS + w_vq L_vq + λ L_distill` or add one sentence after Eq. (8): "L_rec is itself a weighted sum of ℓ1 and LPIPS losses; concrete weights are listed in §IV-A."
- Fig. 1 uses `φ` for the distillation head; §III-C and §IV-A use `g_ψ`. Unify (recommend `g_ψ` since it appears in equations).
- Eq. (10) BPSK hard decision assumes coherent reception. Add one half-line after Eq. (9): "We assume perfect channel state information at the receiver for hard-decision detection in (10)."
- Symbol `C_1 = 1024` is introduced in §III-D but not formally tied to the `codebook size 1024` mentioned in §IV-A. One sentence in §III-A defining `C_ℓ ≡ codebook size at layer ℓ` would close this.

### §IV Experiments
- §IV-A — split documentation is solid (data/AID_splits/, n_train=8000, n_val=n_test=1000, per-class 266.7/33.3 mean). NWPU split (60/20/20 in Table V caption) should also be on disk; if `data/NWPU_splits/` exists, mention it in §IV-A or appendix.
- §IV-B Table I — clean. The `Best PSNR / Best LPIPS` rows refer to training-quality references; consider relabeling to `Training-quality reference` to make the "not main evidence" framing explicit.
- §IV-C Table II — the single-seed caveat ("A single seed = 42 is used for all three checkpoints, but the observed deltas dwarf the three-seed standard deviations measured in our main reconstruction experiment") is honest but defends the wrong column. The L1-only collapse is robust at any reasonable σ; the **All layers** column has deltas (−1.1 pp to −9.3 pp on h₀; −0.28 dB to −0.8 dB on PSNR) that *do* fall within 2× of measured per-seed std. This is what Tier-1 Item 3 addresses.
- §IV-D — clean.
- §IV-E Table IV — the `Source bits (mean)` column shows WebP at `target=2,560` actually emits 14,164 bits (5.5× overshoot), `target=5,120` emits 14,301 bits, `target=10,240` emits 15,410 bits. The caption notes "WebP encoder cannot always reach the lowest budget exactly", but 5.5× is not "not exactly". WebP `k=1` and `k=2` rows fail the matched-bits premise. **Required**: footnote `†` on the two affected rows and one sentence in §IV-E text. No re-run needed.
- §IV-F Table V — h₀ only. Add at least PSNR / LPIPS / recon-cls at k=4 for `rvq_distill` (one inference run, no retraining). This is Tier-1 Item 5.
- §IV-G — fine as a stress test, scoped correctly.

### §V Discussion
- The final paragraph lists three limitations honestly (single-task / single-seed ablations / custom LDPC). Consider adding a fourth bullet on `RemoteCLIP vs. any vision-language teacher`: §IV-C Table VII shows the RemoteCLIP-specific contribution is bounded at 1.4–6.0 pp on h₀ over OpenAI CLIP, so the broader claim is "vision-language distillation is necessary; RemoteCLIP gives a domain-specific increment under fading". The Discussion already implies this but does not state it as a limitation.
- Numerical contradiction check needed: Discussion says "no-channel h₀ accuracy increases monotonically to 84.5%" (referring to Table VIII), but the Abstract's main number is 83.33 ± 0.81% (Table I, 3 seeds). These are not the same quantity (Table VIII is a single-seed sweep at λ=1.0; Table I is the main λ=0.5 result over 3 seeds). Add one parenthetical so a reader doesn't think there's a discrepancy.

### §VI Conclusion
- "supports a conservative but useful claim" — this honest framing is good. Keep.

### Bibliography (`rs_token_v05.bib`)
- Confirm all entries are cited at least once. v0.4 had 9 unused refs; verify v0.5 fixed this. (Quick grep: `cite{` over `.tex` against `.bib` keys.)
- Add a deep-JSCC variant ref (NTSCC: Dai et al., 2022; ADJSCC: Xu et al., 2022; SwinJSCC: Yang et al., 2024) when Tier-1 Item 1 lands.

---

## 2. Experiment plan

The plan is organized into **Tier-1 (must-have to be safe at GRSL)**, **Tier-2 (strongly improves the paper)**, and **Tier-3 (text-only Discussion fixes)**. Naming follows the existing `experiments/rs_token_v0.4_p0_p1_experiment_manual.md` convention: experiment IDs continue from E34, log roots under `logs/paper_v05/`, and config roots under `configs/paper_v05/`.

### 2.0 Common setup

```powershell
cd d:\CODE\遥感+通信\遥感+通信\rstoken
conda activate rstoken     # or full python.exe path
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:NUMEXPR_NUM_THREADS='1'
$env:RSTOKEN_LOGREG_MAX_ITER='300'
$env:RSTOKEN_LOGREG_TOL='1e-3'

New-Item -ItemType Directory -Force `
  logs/paper_v05, logs/paper_v05/run_logs, `
  configs/paper_v05, `
  checkpoints/paper_v05
```

**Pre-flight gate** (must all return `True`):
```powershell
Test-Path checkpoints/rvq_distill/best.pt
Test-Path checkpoints/rvq_distill_s41/best.pt
Test-Path checkpoints/rvq_distill_s43/best.pt
Test-Path checkpoints/rvq_baseline/best.pt
Test-Path checkpoints/rvq_baseline_s41/best.pt
Test-Path checkpoints/rvq_baseline_s43/best.pt
Test-Path checkpoints/rvq_distill_all_layers_s42/best.pt
Test-Path checkpoints/rvq_distill_l1_only_s42/best.pt
Test-Path checkpoints/aid_classifier_resnet34/best.pt
Test-Path checkpoints/remoteclip/RemoteCLIP-ViT-B-32.pt
Test-Path data/AID_splits/train.csv
Test-Path data/AID_splits/val.csv
Test-Path data/AID_splits/test.csv
Test-Path data/NWPU_splits/train.csv      # may be False; see E40
```

**Naming conventions**
- Experiment IDs: **E35–E42** (E34 was the last v0.4 ID).
- Log raw CSVs: `logs/paper_v05/eXX_<short_name>_raw.csv`.
- Mean±std CSVs: `logs/paper_v05/eXX_<short_name>_mean_std.csv`.
- Markdown summaries: `logs/paper_v05/final_table_<short_name>.md`.
- Run logs: `logs/paper_v05/run_logs/eXX_<short_name>.log`.

**Audit rule**: every CSV row must include a `source` column citing the script + commit hash + date. No hand-edited numbers.

---

### Tier-1 experiments (must-do)

| ID | Name | Tier | Cost (wall-clock) | Tier-1 reason |
|----|------|------|-------------------|---------------|
| E35 | Deep-JSCC baseline (ADJSCC or SwinJSCC) on AID, matched channel-symbol budget | T1 | 5–10 days | Closes the "no modern JSCC baseline" reviewer attack |
| E36 | Continuous SNR sweep (AWGN −5..+15 dB, Rayleigh −5..+15 dB) for `rvq_baseline` and `rvq_distill` | T1 | 1–2 days | Replaces 5 channel points with a curve; shows distillation-gain envelope |
| E37 | Placement counterfactual `All layers` column at seeds 41, 43 (seed 42 already exists) | T1 | 2 days | Promotes Table II "All layers" deltas from 1-seed to 3-seed |
| E38 | WebP rate-floor footnote + §IV-E sentence | T1 (text-only) | 10 min | Disarms the most obvious 5.5× overshoot attack |
| E39 | Reconstruction-path zero-shot transfer to NWPU (PSNR/LPIPS/recon-cls at k=1..4) | T1 | 1–2 days | Closes Table V to a full-metric story |

#### E35 — Deep-JSCC baseline (ADJSCC or SwinJSCC) on AID

**Goal.** Add a single modern deep-JSCC comparator at matched channel-symbol budget (or matched bits-per-pixel after rate-1/2 LDPC), so that v0.5 has *one* representative against the "continuous JSCC" branch in addition to RVQ index transmission.

**Recommended baseline.** `ADJSCC` (Xu, Wang, Gao, Zhao, "Wireless Image Transmission Using Deep Source Channel Coding With Attention Modules", *IEEE TCSVT* 2022) is the safest pick because:
- Open-source PyTorch reference: `https://github.com/alexxu1988/ADJSCC` (or any well-cited fork).
- It supports a single model trained over a range of SNRs via the `Attention DL` module — matches the multi-SNR evaluation in §IV-D.
- It produces continuous channel symbols; bits-per-pixel comparison is by `2 × n_channels × H_lat × W_lat / (H × W)` with BPSK assumption.

A second-choice baseline is `SwinJSCC` (Yang et al. 2024) which is stronger but harder to integrate. **Run only one of the two.**

**Ground rules.**
- Train ADJSCC from scratch on the **same AID training split** (`data/AID_splits/train.csv`).
- Match the channel symbol budget so that ADJSCC's effective bits-per-image equals RS-Token at k=1, k=2, k=4 (i.e., 2,560, 5,120, 10,240 bits/image under BPSK).
- Use **the same channel models** as §III-D (AWGN with σ for SNR ∈ {5, 10} dB; Rayleigh i.i.d. CN(0,1) per symbol with the same SNR convention).
- Report: PSNR, LPIPS (AlexNet, same backbone as §IV-A), `clean AID ResNet34 reconstructed-image classifier accuracy` (using `checkpoints/aid_classifier_resnet34/best.pt`, identical to RS-Token's recon-cls metric).
- Three model seeds {41, 42, 43}, exactly as RS-Token main config.
- **Scope discipline**: ADJSCC outputs continuous reconstructions. The h₀/L0 task path is undefined for ADJSCC because there are no discrete L0 indices. Therefore E35 is a **reconstruction-path-only** experiment. Do not invent a task-path metric for ADJSCC.

**Steps.**

1. Create `experiments/baselines/adjscc/` directory; clone the reference implementation:
   ```powershell
   New-Item -ItemType Directory -Force experiments/baselines/adjscc
   git clone https://github.com/alexxu1988/ADJSCC experiments/baselines/adjscc/upstream
   ```
   If the link is dead, search for the canonical TCSVT 2022 ADJSCC repo. Do not invent a fork URL.

2. Adapt the channel module to match RS-Token's:
   - Replace ADJSCC's channel layer with RS-Token's `models/channel.py` (same AWGN/Rayleigh implementation that produces §IV-D numbers). This guarantees apples-to-apples channel statistics.
   - Verify by running both modules on the same input tensor for 10 channel seeds and confirming the per-symbol noise variance matches within 1%.

3. Train three seeds at three rates:
   - For each of `bits_per_image ∈ {2560, 5120, 10240}`:
     - Compute `c_per_symbol = bits_per_image / (image_pixels × bits_per_symbol)` for BPSK; configure ADJSCC's channel-symbol layer to that bandwidth.
     - For each `seed ∈ {41, 42, 43}`:
       - Train for the same epoch budget as RS-Token (50 epochs, AdamW, lr 1e-4, batch 16).
       - Save to `checkpoints/paper_v05/adjscc_b{bits}_s{seed}/best.pt`.
   - **Total**: 9 training runs. If GPU time is tight, drop to seed=42 only and clearly mark Table as single-seed.

4. Evaluate at the same channel grid as RS-Token:
   - Channels: no-channel, AWGN +5 dB, AWGN +10 dB, Rayleigh +5 dB, Rayleigh +10 dB.
   - Metrics: PSNR, LPIPS, recon-cls accuracy.
   - Run with 5 channel seeds {0,1,2,3,4} per checkpoint to match the LDPC stress test seed budget.

5. Output:
   ```
   logs/paper_v05/e35_adjscc_raw.csv           # one row per (model_seed, bits, channel, snr, channel_seed)
   logs/paper_v05/e35_adjscc_mean_std.csv      # aggregated over channel_seeds and model_seeds
   logs/paper_v05/final_table_adjscc.md        # paste-ready markdown
   ```

6. Acceptance criteria:
   - All 9 (or 3 if single-seed) checkpoints exist on disk.
   - Each (bits, channel, snr) cell has at least 3 model seeds × 5 channel seeds = 15 raw rows. Single-seed mode: 1 × 5 = 5 raw rows.
   - PSNR / LPIPS / recon-cls all populated; no missing cells.
   - The `source` column reads `E35 ADJSCC; <repo url>; <commit>; <date>`.

7. Paper integration:
   - Add a new sub-table to §IV-D or a new §IV-X; report ADJSCC vs. RS-Token at matched bits/image, at the four channel conditions.
   - Do **not** delete the existing codec+LDPC stress test. ADJSCC is a separate comparator, not a replacement.
   - Add one cite in §II-A (NTSCC and ADJSCC family).

**Risk and fallback.**
- Risk: open-source ADJSCC may not converge cleanly on 256×256 RS imagery. If after 1 day of debugging it still does not match the published Kodak/CIFAR PSNR within 2 dB on a single sanity image, fall back to **`SwinJSCC` reference** (`https://github.com/semcomm/SwinJSCC`).
- Risk: training time exceeds 10 days. Fallback: 1 model seed × 3 bits × 5 channel seeds is acceptable for a Letter; mark explicitly.
- Risk: cannot find a working open-source repo. Fallback: implement a minimal "convolutional autoencoder + AWGN/Rayleigh channel layer + Gaussian likelihood loss" yourself (250–400 lines), reference DeepJSCC paper directly. Mark in the paper as "internal DeepJSCC implementation; not the published author code".

---

#### E36 — Continuous SNR sweep

**Goal.** Show task-path accuracy and reconstruction quality as smooth curves over SNR, not just at 5 grid points. This is reviewer ammunition: "where does distillation help most, and where does the system fail?"

**Models.**
- `rvq_baseline` (seed 42 main + s41 + s43)
- `rvq_distill` (seed 42 main + s41 + s43)

Already on disk; **inference-only experiment**.

**SNR grid.**
- AWGN: `-5, -2, 0, 2, 5, 7, 10, 12, 15` dB (9 points)
- Rayleigh: `-5, 0, 5, 10, 15, 20` dB (6 points; Rayleigh +5 dB is already a stress baseline, so go further both directions)

**Metrics.**
- Task path: `h_0` accuracy at k=1.
- Reconstruction path: PSNR at k=4.

**Steps.**

1. Extend `scripts/05_eval_channel.py` (or call directly via `scripts/09_eval_rvqs_recon_task_split.py`) to accept the dense SNR list. The existing `--task_snrs` and `--recon_snrs` already accept comma-separated lists; verify in the script.

2. Run sweep:
   ```powershell
   foreach ($model in 'rvq_baseline_s41','rvq_baseline','rvq_baseline_s43','rvq_distill_s41','rvq_distill','rvq_distill_s43') {
     $ckpt = "checkpoints/$model/best.pt"
     python scripts/09_eval_rvqs_recon_task_split.py `
       --models "$model=$ckpt" `
       --recon_models "$model" `
       --task_out "logs/paper_v05/e36_task_${model}.csv" `
       --recon_out "logs/paper_v05/e36_recon_${model}.csv" `
       --task_snrs "-5,-2,0,2,5,7,10,12,15" `
       --recon_snrs "-5,0,5,10,15,20" `
       --ks "1,4" `
       --seed 42 `
       --device cuda `
       --batch_size 64 *> "logs/paper_v05/run_logs/e36_${model}.log"
   }
   ```
   (Channel module already supports negative SNR as long as σ stays finite; spot-check on the smoke test before launching the full sweep.)

3. Aggregate over 3 model seeds with the same PowerShell pattern as E24's `mean_std` step (manual M5.7 in v0.4 manual, lines 180–215). Output `logs/paper_v05/e36_continuous_snr_mean_std.csv`.

4. Make figures:
   - `logs/paper_v05/figs/e36_h0_vs_snr_awgn.pdf` — two lines (baseline vs distill), 9 points each, with mean±std error bars.
   - Same for Rayleigh.
   - Same for k=4 PSNR.

5. Acceptance criteria:
   - `e36_continuous_snr_mean_std.csv` has rows for each (model_family, channel, snr) with `n_model_seeds = 3`.
   - Distillation gain curve is monotone or near-monotone in SNR; if it isn't, do not paper over — state in the figure caption.

6. Paper integration:
   - Add Fig. 4 to §IV-B (`L0 task accuracy vs. SNR, AWGN and Rayleigh, distill vs. baseline`).
   - Optionally add a small inset showing PSNR(k=4) vs. SNR.

---

#### E37 — Placement counterfactual seeds 41, 43

**Goal.** Promote Table II's "All layers" column from a 1-seed point estimate to a 3-seed mean±std, so the −1 to −9 pp deltas are not dismissable as seed noise.

**What's missing.** `checkpoints/rvq_distill_all_layers_s42/best.pt` exists. Need:
- `checkpoints/paper_v05/rvq_distill_all_layers_s41/best.pt`
- `checkpoints/paper_v05/rvq_distill_all_layers_s43/best.pt`

(L1-only is robust at 1 seed because the collapse is −34 to −48 pp; do not retrain.)

**Steps.**

1. Locate the existing all-layers config:
   ```powershell
   Get-ChildItem configs -Recurse -Filter "*all_layers*"
   ```
   Should find `configs/paper_p0/rvq_distill_all_layers_s42.yaml`. Copy to:
   ```
   configs/paper_v05/rvq_distill_all_layers_s41.yaml   # set seed: 41, run_name and dirs to s41
   configs/paper_v05/rvq_distill_all_layers_s43.yaml   # set seed: 43, run_name and dirs to s43
   ```
   Only edit `seed`, `run_name`, `logging.ckpt_dir`, `logging.log_dir`. Do NOT change `loss.distill_weight`, model arch, or `distill.target: all_layers`.

2. Train:
   ```powershell
   python scripts/03_train_vqvae.py --config configs/paper_v05/rvq_distill_all_layers_s41.yaml *> logs/paper_v05/run_logs/e37_train_s41.log
   python scripts/03_train_vqvae.py --config configs/paper_v05/rvq_distill_all_layers_s43.yaml *> logs/paper_v05/run_logs/e37_train_s43.log
   ```

3. Evaluate (same protocol as Table II):
   ```powershell
   foreach ($seed in 41,43) {
     python scripts/09_eval_rvqs_recon_task_split.py `
       --models "rvq_distill_all_layers_s$seed=checkpoints/paper_v05/rvq_distill_all_layers_s$seed/best.pt" `
       --recon_models "rvq_distill_all_layers_s$seed" `
       --task_out "logs/paper_v05/e37_task_s$seed.csv" `
       --recon_out "logs/paper_v05/e37_recon_s$seed.csv" `
       --task_snrs "5,10" `
       --recon_snrs "5,10" `
       --ks "1,2,3,4" `
       --seed $seed `
       --device cuda `
       --batch_size 64 *> "logs/paper_v05/run_logs/e37_eval_s$seed.log"

     python scripts/04b_eval_layered_probe.py `
       --ckpt "checkpoints/paper_v05/rvq_distill_all_layers_s$seed/best.pt" `
       --out "logs/paper_v05/e37_layered_probe_s$seed.csv" `
       --device cuda *> "logs/paper_v05/run_logs/e37_probe_s$seed.log"
   }
   ```

4. Aggregate (with seed 42 from `logs/paper_p0/e25_*.csv`):
   - Output `logs/paper_v05/e37_placement_3seed_mean_std.csv` matching Table II's row layout.
   - Compute three-seed mean±std for h₀ at the 5 channel conditions, k=4 PSNR, k=4 recon-cls, and the four cumulative-codeword probes.

5. Acceptance criteria:
   - Both new checkpoints exist.
   - For each cell of Table II's "All layers" column, the 3-seed std is reported.
   - If the |delta| between L0-only and All-layers exceeds 2× the new measured std for ≥ 60% of cells, the L0-only-as-optimum claim survives. If not, the §IV-C narrative needs softening (write "L0-only is comparable to or modestly preferred over All-layers; spreading the loss across all layers does not help and modestly hurts in fading").

6. Paper integration:
   - Update Table II's "All layers" column to mean±std.
   - Replace §IV-C's single-seed defense paragraph with the new 3-seed comparison.
   - Keep L1-only as 1-seed (note in caption); the collapse is so large it is not seed-bound.

---

#### E38 — WebP rate-floor footnote (text-only)

**Goal.** Disclose that WebP cannot reach the ≤ 14k bits/image floor at 256×256, so Table IV's `WebP, k=1` and `WebP, k=2` rows do not satisfy the matched-bits premise.

**No experiment.** Edit `paper_draft/latex/rs_token_v0.5.tex`:

1. In Table IV, mark the rows `WebP / Rayleigh / k=1`, `WebP / Rayleigh / k=2`, `WebP / AWGN / k=1`, `WebP / AWGN / k=2` with a `$^\dagger$` (LaTeX: `\textsuperscript{$\dagger$}`).

2. Add a footnote to the table:
   ```
   $^\dagger$ WebP encoder cannot reach the source-bit budget at 256$\times$256;
   actual encoded bits (Table IV "Source bits (mean)" column) are 14{,}164--14{,}301
   for these rows, exceeding target by 2.8$\times$--5.5$\times$. The matched-bits
   premise is therefore not satisfied and these cells are reported for completeness;
   the headline RS-Token vs. WebP+LDPC comparison is the $k=4$ row at 20{,}480 total bits
   (15{,}410 actual source bits, $1.5\times$ overshoot, the smallest gap WebP can reach).
   ```

3. In §IV-E "Result.", add one sentence:
   ```
   We note that WebP's encoder cannot meet the smaller two source-bit
   budgets at $256\times256$ (overshoot $2.8\times$--$5.5\times$),
   so the $k=1$ and $k=2$ WebP rows in Table~\ref{tab:strict_codec_ldpc}
   are reported with a $\dagger$ and the headline comparison is
   the $k=4$ matched row.
   ```

4. Acceptance: `pdflatex` cleanly recompiles; the dagger and footnote render.

---

#### E39 — Reconstruction-path zero-shot transfer to NWPU

**Goal.** Table V currently reports h₀ accuracy on NWPU. Add the reconstruction half so the cross-dataset claim is symmetric with the AID main result.

**Models.** `rvq_distill` and `rvq_baseline` (frozen tokenizers, no retraining).

**Steps.**

1. Confirm `data/NWPU_splits/{train,val,test}.csv` exist (E30 in the v0.4 manual). If not, run `python scripts/14_prepare_nwpu_splits.py` first.

2. The `clean AID ResNet34 reconstructed-image classifier` cannot be reused because NWPU has 45 classes ≠ AID's 30. Two options:

   **Option A (recommended)**: Train a **clean NWPU ResNet34 classifier** (matching the AID classifier's recipe in `scripts/08_train_aid_classifier.py`). Cost: 1 day, identical recipe.

   **Option B**: Skip recon-cls; report only PSNR and LPIPS. Faster but weaker.

   Pick A unless GPU budget is tight.

3. Run reconstruction-path inference on NWPU test split:
   ```powershell
   foreach ($model in 'rvq_baseline','rvq_distill') {
     python scripts/09_eval_rvqs_recon_task_split.py `
       --models "$model=checkpoints/$model/best.pt" `
       --recon_models "$model" `
       --task_out "logs/paper_v05/e39_task_nwpu_$model.csv" `
       --recon_out "logs/paper_v05/e39_recon_nwpu_$model.csv" `
       --task_snrs "5,10" `
       --recon_snrs "5,10" `
       --ks "1,2,3,4" `
       --seed 42 `
       --device cuda `
       --batch_size 64 `
       --eval_dataset_csv data/NWPU_splits/test.csv `
       --num_classes 45 `
       --classifier_ckpt checkpoints/paper_v05/nwpu_classifier_resnet34/best.pt `
       *> "logs/paper_v05/run_logs/e39_nwpu_$model.log"
   }
   ```
   If `scripts/09_eval_rvqs_recon_task_split.py` does not yet accept `--eval_dataset_csv` and `--num_classes`, **the minimum code change is two added flags** that flow into the existing `AIDDataset` constructor and the recon-cls classifier loader. Confirm in `scripts/09_eval_rvqs_recon_task_split.py` before launching.

4. Single seed (42) is acceptable for Table V symmetry; mark in caption.

5. Output:
   ```
   logs/paper_v05/e39_recon_nwpu_raw.csv
   logs/paper_v05/e39_recon_nwpu_summary.md
   ```

6. Acceptance: PSNR / LPIPS / recon-cls populated for both models, all 5 channel conditions, k=1..4. The recon-cls accuracy on NWPU is **not** required to match AID — degradation is expected and is itself a result.

7. Paper integration:
   - Extend Table V to add columns `k=4 PSNR`, `k=4 LPIPS`, `k=4 recon-cls (45-class)`.
   - One sentence in §IV-F: "Reconstruction-path transfer is consistent with the task-path transfer: clean-channel PSNR drops by X dB and recon-cls drops by Y pp, while the relative ordering of `rvq_distill` over `rvq_baseline` is preserved."

---

### Tier-2 experiments (strongly improves the paper)

| ID | Name | Tier | Cost | Reason |
|----|------|------|------|--------|
| E40 | Init-seed control: same seed, distill vs. baseline | T2 | 2 days | Quantifies "any extra L0 supervision" floor |
| E41 | RVQ direct-quantize-RemoteCLIP control | T2 | 3–5 days | Tests whether RVQ encoder is needed at all |
| E42 | K (number of RVQ layers) ablation: K∈{2,3,4,6} | T2 | 4–5 days | Defends K=4 choice |

#### E40 — Init-seed control

**Question.** The 25-pp h₀ gap between `rvq_distill` and `rvq_baseline` could be partly attributable to network init noise. The current 3-seed protocol uses different seeds {41, 42, 43} for both models, so init noise contributes to *the std* but is not isolated. E40 trains baseline and distill **with bit-identical init** at three seeds and measures the *paired* gap.

**Steps.**
1. Modify `scripts/03_train_vqvae.py` to accept `--load_init_from <ckpt>` that copies the encoder + RVQ codebooks from a reference state-dict at step 0, so distill and baseline start from the same point.
2. For each seed in {41, 42, 43}:
   - Generate a `step_0` checkpoint from a fresh init at that seed.
   - Train `rvq_baseline_paired_s{seed}` and `rvq_distill_paired_s{seed}` from that step_0 init.
3. Evaluate task path (same as Table I).
4. Report the paired gap mean±std.

**Acceptance.** If the paired gap is within 1 pp of the unpaired 3-seed gap, the v0.5 conclusion is unchanged and is *strengthened*. If the paired gap shrinks materially, §IV-B needs to add "init noise contributes ~X pp; the residual gap is the actual distillation contribution".

**Note.** Mark as Tier-2 because the v0.5 manuscript already controls for the same shared seed (s42 baseline and s42 distill differ only in `λ_distill`, per §IV-A), and the 3-seed std is small. E40 is a paranoid extra control, not a fix to a known flaw.

---

#### E41 — RVQ direct-quantize-RemoteCLIP control

**Question.** Why is the RVQ encoder needed at all? Could you simply RVQ-quantize the RemoteCLIP image embedding, bypass the encoder, and obtain comparable task accuracy?

**Steps.**
1. Build a tokenizer that:
   - Skips the encoder.
   - Takes RemoteCLIP image embeddings (512-d) directly.
   - Applies a 4-layer RVQ on the embedding (codebook size 1024 each, same total bits/image as RS-Token via reshaping + chunking).
2. Train end-to-end on AID with reconstruction loss to a synthesis decoder (or skip recon and report task-only).
3. Compare h₀ accuracy and PSNR to RS-Token.

**Expected result.** Direct RemoteCLIP quantization will likely give **higher** h₀ accuracy (RemoteCLIP is closer to the task) but **much worse** PSNR (RemoteCLIP feature is not invertible). This *supports* RS-Token's hierarchical design.

**Paper integration.** One paragraph in §V Discussion: "We considered a direct-quantize-RemoteCLIP variant which trades reconstruction for task fidelity..." Cite the result table in an appendix.

---

#### E42 — K (RVQ layers) ablation

**Question.** The paper fixes K=4 RVQ layers without justification.

**Steps.**
1. Train `rvq_distill_K{2,3,6}_s42`. K=4 already exists.
2. Compare:
   - Total bits at full K: 5,120 / 7,680 / 10,240 / 15,360 bits/image.
   - h₀ at k=1.
   - PSNR at k=K (full).
3. K=2: h₀ may rise (less dilution) but max PSNR drops; K=6: max PSNR rises but per-prefix bit cost grows.

**Acceptance.** K=4 should be a sweet spot; if K=3 dominates K=4 on both axes, the paper needs to either re-run main results at K=3 or justify K=4 (e.g. "matches deployment bit grid").

---

### Tier-3 (text-only, no experiment)

| ID | Item | Location |
|----|------|----------|
| T3.1 | Fix half-width "：" in Abstract | §Abstract |
| T3.2 | "receiver transmits" → "transmitter sends" | §Abstract |
| T3.3 | Unify φ vs. g_ψ | Fig.1 caption + §III-C |
| T3.4 | Eq. (8) loss expansion or reference to §IV-A weights | §III-C |
| T3.5 | Eq. (10) "perfect CSI at receiver" | §III-D |
| T3.6 | Define `C_ℓ` formally in §III-A | §III-A |
| T3.7 | Add 1 deep-JSCC variant cite (NTSCC/ADJSCC/SwinJSCC) | §II-A and bib |
| T3.8 | Discussion bullet on "vision-language is necessary, RemoteCLIP increment is bounded" | §V |
| T3.9 | Discussion bullet on `5G NR LDPC out of scope; results invariant under code class up to a constant gap` | §V |
| T3.10 | Limitation on "scene classification only; object detection / segmentation needs L0 grid > 16×16" | §V |
| T3.11 | Verify all bib entries cited; v0.4 had 9 unused refs | bib |

---

## 3. Execution order

Recommended order to maximize "paper improves earliest":

1. **Day 0**: T3.1–T3.6 (typography and equations) + E38 footnote — 30 min.
2. **Day 0–1**: E36 SNR sweep (inference-only, ~1 day) — gives a striking new figure.
3. **Day 1–3**: E37 placement counterfactual seeds 41, 43 — strengthens the keystone Table II.
4. **Day 1–3 (parallel)**: E39 NWPU recon transfer — closes Table V.
5. **Day 2–10**: E35 ADJSCC baseline — the single most important reviewer-defense item.
6. **Day 6–9 (parallel if GPU permits)**: E40, E42.
7. **Day 8–12**: T3.7–T3.11 Discussion edits as the new results land.
8. **Day 12+**: E41 if time allows.

Tier-1 alone covers the most likely audit attacks. Tier-2 lifts the paper from "GRSL-safe" to "TGRS-borderline". Tier-3 is non-negotiable polish.

---

## 4. Acceptance gate before next submission

Do not re-submit until:
- [ ] E35 ADJSCC table appears in the manuscript (or 1 paragraph in §V explaining why deep-JSCC was not run, with one citation).
- [ ] E36 continuous-SNR figure replaces or supplements the discrete 5-point grid.
- [ ] E37 Table II "All layers" column shows mean±std over 3 seeds.
- [ ] E38 WebP dagger footnote present.
- [ ] E39 Table V has PSNR / LPIPS / recon-cls columns.
- [ ] All Tier-3 items applied.
- [ ] Bib hygiene check passed: every `\cite{key}` resolves and no `.bib` entry is unused.
- [ ] Total page count still ≤ 5 (+ refs) for IEEE GRSL Letters template.

---

## 5. Files this plan will create

```
configs/paper_v05/
  rvq_distill_all_layers_s41.yaml
  rvq_distill_all_layers_s43.yaml
  adjscc_b{2560,5120,10240}_s{41,42,43}.yaml         # E35 baselines
  rvq_distill_K{2,3,6}_s42.yaml                       # E42

checkpoints/paper_v05/
  rvq_distill_all_layers_s41/best.pt
  rvq_distill_all_layers_s43/best.pt
  adjscc_b{...}_s{...}/best.pt
  nwpu_classifier_resnet34/best.pt                    # E39 option A
  rvq_distill_K{...}_s42/best.pt

logs/paper_v05/
  e35_adjscc_raw.csv, e35_adjscc_mean_std.csv, final_table_adjscc.md
  e36_continuous_snr_raw.csv, e36_continuous_snr_mean_std.csv
  e36_h0_vs_snr_{awgn,rayleigh}.pdf
  e37_placement_3seed_mean_std.csv, final_table_placement_3seed.md
  e39_recon_nwpu_raw.csv, e39_recon_nwpu_summary.md
  e40_init_seed_control_paired.csv                    # T2
  e41_direct_quantize_rclip.csv                       # T2
  e42_k_ablation.csv                                  # T2
  run_logs/e35_*.log, e36_*.log, e37_*.log, e39_*.log, e40_*.log, e41_*.log, e42_*.log

paper_draft/latex/rs_token_v0.5.tex   # patches for Tier-3 + E38 footnote + new tables/figs
```

End of plan.
