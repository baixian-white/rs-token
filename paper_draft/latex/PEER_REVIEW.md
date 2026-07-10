# PEER REVIEW — rs_token_v0.4.tex

**Manuscript**: RS-Token: Hierarchical RemoteCLIP-Distilled Tokens for Channel-Robust Remote Sensing Communication
**Author**: Baohui Zhang
**Reviewer skill**: peer-review (K-Dense Inc., CONSORT/STROBE-style structured review adapted to ML/communications letter)
**Date**: 2026-06-08

---

## Summary Statement

RS-Token proposes applying RemoteCLIP distillation to **only the first layer** of a four-layer RVQ tokenizer for remote-sensing image communication, so that L0 indices alone can support scene-level decisions while L1–L3 progressively refine reconstruction. The technical idea is clean and the evaluation design is unusually well-disciplined: task path ($h_0$/L0 bag-of-words), layered probe (cumulative codeword embeddings), and reconstruction path (PSNR/LPIPS/clean-classifier) are kept strictly separate, and each claim is bound to one metric family. Across three model seeds on AID, RemoteCLIP distillation improves no-channel L0 task accuracy from 58.23% to 83.33%, with similar gains preserved under AWGN +5 dB and Rayleigh +10 dB. Reconstruction quality grows monotonically with k=1..4. Supplementary LDPC and unprotected WebP/JPEG2000 experiments are clearly scoped as boundary tests, not as primary claims.

The paper is most weakened by **literature/citation hygiene** rather than by methodology: 9 of 16 entries in `rs_token.bib` are never cited in the body, including directly competing work (MOC-RVQ, DeepJSCC, SemCLIP). A single asymmetric statistical-reporting issue (3-seed task path vs. single-seed reconstruction path) and a title that overstates "channel robustness" given Rayleigh +5 dB performance round out the major concerns. None of the major issues are foundational; all are addressable in a major revision.

**Overall recommendation**: **Major Revision** (mandatory before any submission). With P0 fixes (citations, recon-path seeding, title), the paper is suitable for **IEEE GRSL** or a similar letter venue.

**Key strengths**
- Strict task-path / probe / reconstruction-path separation is rare and reviewer-friendly.
- `rvq_baseline` is a tightly controlled internal comparator (only λ_distill changes).
- Self-disciplined claim scoping ("not a 5G NR LDPC", "not a primary v0.4 claim", etc.) — anticipates and disarms many likely reviewer attacks.

**Key weaknesses**
- 9 unused bib entries; missing engagement with directly competing work (MOC-RVQ).
- Reconstruction-path Table 3 has no std despite the same seed infrastructure being used elsewhere.
- Title says "Channel-Robust" but Rayleigh +5 dB k=1 recon-cls = 47% even *with* LDPC.

---

## Major Comments

### MC1. Nine unused bibliography entries; missing engagement with the most directly competing work (MOC-RVQ)

**Location**: `rs_token.bib`, §2.1, §2.2.

`rs_token.bib` defines 16 entries. The body of `rs_token_v0.4.tex` cites only 7 of them (`gao2022task`, `vqvae`, `lee2022residual`, `soundstream`, `encodec`, `liu2024remoteclip`, `zhang2018lpips`, `xia2017aid`). The 9 entries that are defined but never cited include `moc-rvq`, `deepjscc`, `revqom`, `semclip`, `vilau`, `beitv2`, `zhang2024speechtokenizer`, `proakis`, and one additional. IEEE editorial screening typically flags unused references; this can cause desk-rejection independent of technical merit.

Of those 9 unused entries, **MOC-RVQ (Zhou et al., arXiv:2401.01272) is the most directly competing concurrent work** and must be engaged with: it also combines residual VQ with multi-codebook structure for semantic communication. The current Related Work does not contrast RS-Token's L0-only distillation against MOC-RVQ's multilevel codebook design. **DeepJSCC** (Bourtsoulatze, Burth Kurka, Gündüz, 2019) is the foundational JSCC anchor that almost every semantic-communication paper cites, and it is missing from §1 ¶2 where compress-then-transmit is contrasted with semantic communication. **SemCLIP** (Hu et al., 2025) is a CLIP-family semantic-communication paper that should be acknowledged.

**Why this is problematic**: even setting aside editorial-screening risk, a related-work section that omits the most directly competing concurrent work weakens contribution claims. A reviewer cannot judge novelty without seeing the contrast against MOC-RVQ.

**Required revisions**
1. Either cite or delete every entry in `rs_token.bib`. Do not submit with unused refs.
2. Add a paragraph in §2.1 (or new §2.5) that explicitly contrasts RS-Token vs. MOC-RVQ on at least: which layer carries semantic supervision, which teacher, which task, and which channel model.
3. Cite `deepjscc` in §1 ¶2 where compress-then-transmit is contrasted with semantic communication.
4. Cite `semclip` and `revqom` in §2.1 / §2.2 as contemporaneous foundation-model-based or RVQ-based communication work.

**Essential for publication**: yes.

---

### MC2. Reconstruction-path Table 3 reports point estimates while task-path Table 1 reports mean ± std

**Location**: §4.4, Table tab:recon_path.

Table 1 (`tab:task_seed`) reports task-path metrics as mean ± std over 3 model seeds {41, 42, 43}. Table 3 (`tab:recon_path`) reports reconstruction-path metrics as point estimates from a single main seed. The paper uses the same seed infrastructure for both, so this asymmetry is a reporting choice, not a data limitation.

**Why this is problematic**: the asymmetric reporting weakens reconstruction-path claims relative to task-path claims, and makes it harder for a reader to assess whether the "k=1 → k=4 improves PSNR by 2.97 dB" gain is reliably outside seed noise. Reviewers will read this asymmetry as the paper hiding noisy reconstruction numbers.

**Required revisions**
1. Re-run reconstruction-path for the no-channel and AWGN +5 dB sweeps on seeds 42 and 43. At minimum, k=1 and k=4.
2. Report Table 3 with std, matching Table 1's reporting style.
3. If full sweep is infeasible in the revision window, run only k=1 and k=4, and add a paragraph stating that intermediate k=2, k=3 are reported on the main seed only.

**Essential for publication**: yes.

---

### MC3. Title "Channel-Robust" overstates the demonstrated robustness

**Location**: title; abstract.

The title claims RS-Token is "Channel-Robust." However, Table tab:recon_path reports Rayleigh +5 dB k=4 reconstructed-image classifier accuracy of **21.0%** unprotected, and Table tab:ldpc_rstoken reports Rayleigh +5 dB k=1 reconstructed-image classifier accuracy of **47.2 ± 0.6%** even *with* a rate-1/2 LDPC code. The paper's own Discussion §5 ¶4 acknowledges this: "Rayleigh +5 dB remains difficult ... channel coding improves robustness but does not fully remove the effect of strong fading."

**Why this is problematic**: the headline word in the title is contradicted by the paper's own results table. Reviewers will treat this as overclaiming.

**Required revisions**
- Rename the title to one of: "Channel-Aware Remote Sensing Communication," "Channel-Adaptive Remote Sensing Communication," or "Hierarchical Tokens for Variable-Rate Remote Sensing Communication." These match what the paper actually demonstrates.
- Update the abstract's framing accordingly.

**Essential for publication**: strongly recommended.

---

### MC4. The layered-probe conclusion is partially circular

**Location**: §4.3 Conclusion paragraph.

The layered probe (Table tab:layer_probe) is interpreted as evidence that L1–L3 do not naturally encode task semantics. However, the distilled model is **engineered** to apply distillation only on L0; L1–L3 receive no semantic supervision by construction. Of course they will not gain task accuracy. The probe demonstrates that the engineered specialization was achieved, not that L1–L3 are intrinsically non-semantic. The current phrasing — "supports the intended specialization" — invites a reviewer to point this out.

**Why this is problematic**: a reviewer who notices the circularity will interpret the §4.3 result as confirming the construction rather than discovering a property. This weakens the inferential weight of the experiment.

**Required revisions**
- Rephrase the §4.3 Conclusion as: "The probe confirms that the engineered specialization was achieved at training time — L0 carries scene semantics while L1–L3 add little additional task accuracy, consistent with their reconstruction-residual role."
- Optionally, add a counterfactual probe: train a second `rvq_distill` variant with distillation applied to L1 instead of L0, and show that L0 there does not gain task accuracy. This would convert §4.3 from an engineering confirmation into a genuine causal test of "distillation localizes semantics to its target layer."

**Essential for publication**: rephrasing is essential; the counterfactual probe is a strong but optional improvement.

---

### MC5. No cross-dataset evidence — generalization claims rest on AID alone

**Location**: §4 throughout.

All experiments use AID. Standard practice in remote-sensing scene classification is to evaluate on at least one additional benchmark (NWPU-RESISC45, UC Merced, RESISC-45). RS-Token's claim that L0 carries "remote-sensing scene semantics" is not currently testable outside AID's 30 classes.

**Why this is problematic**: the paper's framing is general (RS-Token is "for remote-sensing communication"), but the evidence is dataset-specific. A reviewer could reasonably ask whether L0's RemoteCLIP-aligned representation transfers to RESISC45 categories not seen during distillation.

**Required revisions**
- Run zero-shot tokenizer transfer on NWPU-RESISC45: encode RESISC45 images with the AID-trained `rvq_distill` model, build $h_0$ on RESISC45, train a linear probe. Report top-1.
- Even one cross-dataset table substantially raises generalization confidence.

**Essential for publication**: required for journal venues (TGRS, TCOM); recommended but not strictly required for letter venues (GRSL, WCL).

---

## Minor Comments

### mc1. Abstract length

The abstract is approximately 370 words. IEEE GRSL guidance is ≤200 words. Cut to 200 words by keeping: one-sentence framing, one-sentence method, three core results (one number each), one supplementary-experiment sentence.

### mc2. AID train/val/test split is undeclared

§4.1 says "a fixed train/validation/test split on AID" but does not state the ratio or per-split sample counts. Add one sentence specifying both.

### mc3. System cost is missing

For a paper framed around UAV / emergency communication, encoder parameters, decoder parameters, inference FLOPs/image, and the fact that RemoteCLIP teacher is *training-only* should be reported. Add a short subsection or paragraph.

### mc4. Repeated metric-separation declaration

The "task path / reconstruction path / layered probe" separation is stated in §1 ¶5, again at start of §3.4, and again at start of §4. Consolidate to one location (start of §4) and reference back from §1 and §3.

### mc5. Table 4 lacks an unprotected baseline column

The most informative LDPC datapoint — "Rayleigh +5 dB k=1 recon-cls jumps from 21% (Table 3) to 47% (Table 4)" — is currently split across two tables. Either add an "unprotected" column to Table 4, or add a footnote pointing to the Table 3 row.

### mc6. Inconsistent std reporting

Table 4 reports mean ± std (over 5 channel seeds). Table 5 reports mean only. Pick one convention and apply consistently, or explicitly note that channel-seed variance for WebP/JPEG2000 was not computed.

### mc7. Notation `K=4` vs. `k`

§3.1 uses `K=4` for the total number of layers; the rest of the paper uses lowercase `k` for the prefix length. The single uppercase `K` in §3.1 is fine, but worth scanning that no later equation accidentally mixes them.

### mc8. Figure paths are local

`\includegraphics{../../rstoken/figs/...}` is robust for local build but may break in submission packaging. Move figures into a `figures/` subdirectory under `paper_draft/latex/` and use relative paths within the latex root before submission.

### mc9. Contribution #2 wording

Contribution #2 says "We formulate prefix-style RVQ index transmission." Prefix decoding is a defining property of RVQ since SoundStream/EnCodec; the contribution is *applying* prefix transmission to RS communication. Rephrase as "We adapt prefix-style RVQ index transmission to remote-sensing communication and tie its operating points to scene-level task accuracy."

### mc10. §5 ¶1 sentence length

Several sentences in §5 ¶1 exceed 80–100 words. Letter venues prefer shorter sentences. One pass for sentence-level brevity.

---

## Questions for the Author

1. Is the reconstruction-path Table 3 actually a single-seed run, or did you run all three seeds and choose to report only the main seed? If the data exists, why was std not reported?
2. Was the LDPC column (Table 4) `--` for $h_0$ at k=4 a deliberate methodological choice, or was the experiment simply not run? A footnote stating "$h_0$ is k=1 only by definition" would clarify.
3. RemoteCLIP-ViT-B/32 is the teacher; what is the distillation-feature dimensionality at the projection head $g_\psi(q_1)$ ? Is there a teacher-feature normalization step beyond cosine?
4. Is `rvq_baseline` initialized from the same random seed as `rvq_distill`, or independently? If independently, please confirm that the 25-pp gap is not partially attributable to initialization variance.
5. The teacher is described as "frozen RemoteCLIP-ViT-B/32." During distillation, is the teacher applied to the original image $x$ or to the reconstruction $\hat{x}_1$? §3.3 implies the original image — please confirm.
6. The bag-of-words feature $h_0$ is described as 1024-dimensional (one bin per L0 codeword index). Are positional weights applied, or is it pure histogram? §3.4 and §4.1 should align on this.

---

## Final Checklist

- [x] Summary statement clearly conveys overall assessment
- [x] Major concerns are clearly identified and justified
- [x] Suggested revisions are specific and actionable
- [x] Minor issues are noted but properly categorized
- [x] Statistical methods evaluated (asymmetric seeding flagged in MC2)
- [x] Reproducibility evaluated (split ratio in mc2; system cost in mc3)
- [x] Reporting standards: IEEE letter format guidance applied
- [x] Figures and tables evaluated
- [x] Writing quality assessed
- [x] Tone is constructive and professional throughout
- [x] Recommendation consistent with identified issues: **Major Revision**

---

*This review was generated using the peer-review skill (K-Dense Inc., MIT). Companion analytical scoring is in `SCHOLAR_EVALUATION.md` (ScholarEval framework, 8 dimensions).*
