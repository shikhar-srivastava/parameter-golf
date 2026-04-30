# LayerRoPE: Complex-Rotation Depth Gates with Shared Gamma + RoPE-Global

Modification of the **PR #1851 record stack** (`val_bpb = 1.06145` 3-seed mean, see [README_latest_record.md](README_latest_record.md)) that replaces the per-block `attn_scale` / `mlp_scale` per-channel residual gains and the fixed `ln_scale_factor = 1/√(ℓ+1)` pre-norm gain with a single **execution-step-conditioned** depth-gate construction applied to four shared gamma vectors per gate position.

**Status**: Implemented and parsing-validated against the new record codebase. Not yet trained.

> **Implementation file:** [`train_gpt_layerrope.py`](train_gpt_layerrope.py) (164,554 bytes; +12 KB / +234 lines vs. record source).

---

## What this is, in one paragraph

The record's residual stream is gated at three points per Block: a fixed `1/√(ℓ+1)` pre-norm scalar and per-channel `attn_scale` / `mlp_scale` learnable scales. We collapse all three into a single learned, depth-conditioned, complex-rotation operation acting on shared `γ` vectors. The depth schedule is a power-law in `log(k+1)` (with `k` the unrolled step index), and the rotation angle is expanded RoPE-style across consecutive channel pairs. Net effect: a **per-step, per-channel rotation-and-scale** of γ, applied at four positions (input pre-norm and post-block residual, for both attn and MLP branches).

The integration is layered cleanly *on top of* every existing technique in the new record (CaseOps SP8192, SmearGate + BOS fix, LQER asymmetric, Phased TTT with LoRA, Polar Express Muon, fused softcapped CE, MIN_LR=0.1, parallel residuals from layer 8, depth recurrence with looping at frac=0.35).

---

## Mathematical specification

For each visit step `k ∈ {0..max_steps-1}` (with `max_steps = 17` once looping is active in the default config) and each gate position `p ∈ {input_attn, input_mlp, residual_attn, residual_mlp}`:

`u = log(k+1)`

```
mag_p(k)   = α_p + β_p · u                              [FP32 scalar]
Θ_p(k)     = exp(α_rot,p + β_rot,p · u)                  [FP32 scalar]
θ_{p,j}(k) = Θ_p(k) · 100^(−2j/H), j ∈ {0..H/2-1}        [FP32 (H/2,)]
```

The effective γ at step `k` is obtained by treating consecutive pairs `(γ_p[2j], γ_p[2j+1])` as Re+i·Im and multiplying by `exp(mag_p(k) + i·θ_{p,j}(k))`:

```
γ_eff,p[2j]   = exp(mag_p(k)) · ( γ_p[2j]·cos(θ_{p,j}) − γ_p[2j+1]·sin(θ_{p,j}) )
γ_eff,p[2j+1] = exp(mag_p(k)) · ( γ_p[2j]·sin(θ_{p,j}) + γ_p[2j+1]·cos(θ_{p,j}) )
```

**Application** (per branch ∈ {attn, mlp}):
- Pre-norm input gate: `pre_norm_out = γ_eff,input_branch(k) ⊙ RMSNorm(x_in)` — replaces `ln_scale_factor`.
- Residual gate: `block_contribution = γ_eff,residual_branch(k) ⊙ block_output` — replaces `attn_scale` / `mlp_scale`.

Standard residual addition follows. In parallel-residual visits (decoder layers ≥ `parallel_start_layer`), the residual gate fires *before* the per-layer 2×2 mixing matrix `parallel_post_lambdas` distributes attn_out / mlp_out across the two lanes.

All gate arithmetic runs in FP32, matching the FP32 invariants of the surrounding stack (Muon updates, GPTQ Hessians, EMA).

### Indexing modes

| Mode | `step_idx` for visit at position `k` | Effect |
|---|---|---|
| **`exec_step`** (default) | `k` (0-indexed position in unrolled forward pass) | Looped layers receive *progressively more damping* per visit. Aligns with how `BatchedTTTLoRA` already keys per-visit adapters. |
| `physical` | physical layer index of visit `k` | Looped layers reuse the same gate every visit. Matches the published shared-gamma convention. |

Toggle via `DEPTH_GATE_INDEXING={exec_step,physical}`.

---

## Modifications to the record code (`train_gpt.py` → `train_gpt_layerrope.py`)

| # | Where | What |
|---|---|---|
| 1 | `Hyperparameters` | New env vars `GATE_ALPHA_INIT`, `GATE_BETA_INIT`, `GATE_ALPHA_ROT_INIT`, `GATE_BETA_ROT_INIT`, `GATE_ROPE_BASE_FREQ`, `DEPTH_GATE_INDEXING` |
| 2 | New class `DepthGates` (after `RMSNorm`) | 4 shared γ vectors of `(H,)` + 16 schedule scalars + 2 FP32 buffers (`log_depths` of size `max_steps`, `rope_freqs` of size `H/2`); both buffers `persistent=False` |
| 3 | `Block.__init__` | Removed: `attn_scale`, `mlp_scale`, `ln_scale_factor`. Kept: `attn_norm`, `mlp_norm`, `resid_mix` and all attention-internal flags |
| 4 | `Block.forward` | Signature gains `gates`; 3 references to removed fields replaced by gate ops |
| 5 | `GPT.__init__` | Computes `max_steps = max(num_layers, len(encoder_indices)+len(decoder_indices))` after looping config is known; instantiates `self.depth_gates = DepthGates(...)` |
| 6 | `_parallel_block` | Signature gains `gates`; same 3 substitutions |
| 7 | `_block_with_lora` (TTT path) | Signature gains `gates`; same 3 substitutions |
| 8 | `_parallel_block_with_lora` (TTT path) | Signature gains `gates`; same 3 substitutions |
| 9 | `_forward_hidden` | Precomputes `gates_per_step` (length depends on `looping_active`); passes `gates_per_step[step]` to all 4 block call sites |
| 10 | `forward_ttt` | Same `gates_per_step` precompute; passes to `_block_with_lora` / `_parallel_block_with_lora` (slot index doubles as exec_step index — already what `BatchedTTTLoRA` uses) |
| 11 | `CONTROL_TENSOR_NAME_PATTERNS` | Adds `gate_alpha,gate_beta,gamma_input,gamma_residual` |
| 12 | `Optimizers.__init__` | Appends `base_model.depth_gates.parameters()` to `scalar_params` (depth-gate scalars + γ vectors → AdamW with `scalar_lr`, `adam_wd`) |
| 13 | `restore_fp32_params` | Re-casts `DepthGates.log_depths` and `DepthGates.rope_freqs` to FP32 after the global `.bfloat16()` |

---

## Parameter and artifact-size delta vs the record

| | Δ |
|---|---|
| Removed: per-block `attn_scale` + `mlp_scale` (11 × 2 × 512) | −11,264 |
| Added: 4 shared γ vectors (4 × 512) | +2,048 |
| Added: 16 schedule scalars (4 per gate position × 4 positions) | +16 |
| **Net Δ params** | **−9,200** |

`log_depths` (17 floats) and `rope_freqs` (256 floats) are buffers with `persistent=False` — excluded from `state_dict`, so they do not contribute to the saved artifact. The 16 schedule scalars + 2,048 γ entries route through `passthrough (float16)` in `gptq_mixed_quantize` (numel < 65,536, no `attn_gate_w` match) — same as `attn_scale`/`mlp_scale` were before. Net artifact size shrinks slightly (fewer FP16-passthrough bytes).

---

## Compatibility with each new-record technique

| Technique | Interaction with LayerRoPE |
|---|---|
| **CaseOps SP8192** | None (tokenizer-side; data flows through unchanged) |
| **SmearGate + BOS fix** | Operates on token embeddings before residual stream forms; LayerRoPE applies inside residual stream → orthogonal |
| **LQER asymmetric** | Operates on quantized `qo_bank/kv_bank/mlp_*_bank` weights; LayerRoPE params (1D) take the `passthrough (float16)` path → orthogonal |
| **Polar Express Muon** | Updates the 4 weight banks; LayerRoPE params route to scalar AdamW → orthogonal |
| **Fused softcapped CE** | Logit-side; orthogonal |
| **Phased TTT with LoRA** | `BatchedTTTLoRA` already keys per-visit by exec-step slot index; LayerRoPE gates use the *same* exec-step index in `forward_ttt` → train and TTT see identical gates per position. Note: TTT does not adapt our gate parameters (only LoRA adapters train) — gates contribute via *training-time* effect, evaluated at pre-quant / quantized-sliding metrics |
| **`MIN_LR=0.1`** | Floors the warmdown; gate scalars sit on the scalar_lr schedule and benefit from non-zero floor for late-training convergence |
| **Parallel residuals from layer 8** | LayerRoPE gates fire before the `parallel_post_lambdas` 2×2 mix; sequential decoder layers (3–7) and parallel decoder layers (8–10) get identical gate treatment |
| **Looping (depth recurrence)** | `max_steps = 17` covers both pre-loop (11 visits) and post-loop (17 visits) regimes. With `exec_step` indexing, layer 4's three visits get gates `(3,7,10)` rather than `(3,3,3)` — mechanism's intent (residual-magnitude damping) tracks accumulated residual count, not physical-layer ID |

---

## Reproduction (step-by-step)

Working directory: `/gpfs/fs2/scratch/ssrivas9/parameter-golf`. All commands assume you are on a **GPU node** (e.g. `bhg0074`) with the `pgolf` conda env active. The login node's libstdc++ is too old to load PyTorch.

### Step 0 — Sanity-check the environment

```bash
hostname                                # should NOT be the login node
conda activate pgolf
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# Expect: 2.9.1+cu128, 4 (or 8 on SXM)
```

### Step 1 — Data prep (one-time, ~16 GB)

The new record's README says `python3 prepare_caseops_data.py` "downloads from romeerp/parameter-golf-caseops-v1", but the script does *not* download. Its docstring documents `--docs/--out/--sp` for tokenizing a local `docs_selected.jsonl`, but **the script's defaults also don't reproduce the record's shards**: the HF manifest shows the record was generated with `--val-docs 50000` (script default 10000) and `SHARD_TOKENS=100_000_000` (hardcoded in the script as 10,000,000 on line 66). Following the docstring literally would yield ~800 train shards of 10M tokens each — incompatible with what `train_gpt.py` expects.

The HF repo already ships the canonical pre-tokenized shards (with the record's actual generation parameters baked in), so the safe and only source-controlled path is to download them directly:

```bash
pip install brotli python-minifier hf_transfer

HF_HUB_ENABLE_HF_TRANSFER=1 python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='romeerp/parameter-golf-caseops-v1',
    repo_type='dataset',
    local_dir='data/datasets/fineweb10B_sp8192_caseops',
    max_workers=8,
)
"
```

The HF repo's internal layout already matches what `train_gpt.py` expects under `CASEOPS_ENABLED=1`, so no symlinks or post-processing.

**Verify the layout (should see 80 train shards + 1 val + 1 val_bytes + tokenizer):**

```bash
DATASET_DIR=data/datasets/fineweb10B_sp8192_caseops/datasets/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved
TOK_DIR=data/datasets/fineweb10B_sp8192_caseops/datasets/tokenizers

ls $DATASET_DIR/fineweb_train_*.bin     | wc -l    # expect 80
ls $DATASET_DIR/fineweb_val_000000.bin              # expect present
ls $DATASET_DIR/fineweb_val_bytes_000000.bin        # expect present
ls $TOK_DIR/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model    # expect present
du -sh data/datasets/fineweb10B_sp8192_caseops      # expect ~16 GB
```

**Verify byte-identical to the record's data** (5 representative SHA-256s — these are the canonical hashes from the HF repo's git LFS pointers):

```bash
sha256sum \
  $DATASET_DIR/fineweb_train_000000.bin \
  $DATASET_DIR/fineweb_train_000079.bin \
  $DATASET_DIR/fineweb_val_000000.bin \
  $DATASET_DIR/fineweb_val_bytes_000000.bin \
  $TOK_DIR/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

Expected (line-for-line):
```
d6cc7eaeb55d935f3d43927801763fdfc4c36ddf51f5a728a9ef2b83197546b4  fineweb_train_000000.bin
ea41f9ae8e4a728694d40cf04757f43d222cf5359288dc2735820bc3ea59edd4  fineweb_train_000079.bin
a9be3eceb56f49358a199a97bd42ccaf69a10f0e103a117514ca5debad7083ab  fineweb_val_000000.bin
9b7e97ce6d7b119ba9ef8f5df6d3c733924eda9a973fa9f1452d9411e14c22d5  fineweb_val_bytes_000000.bin
97754754561f20def5edd99765ca0668f880bd94e67316d54aa10ab25f45cc4d  fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
```

The tokenizer hash also matches the file shipped under `records/track_10min_16mb/2026-04-29_SmearGateBOSFix_3Seed_1.06141/tokenizers/`. Token-count fingerprint additionally confirms identity: 80 train shards × 100M tokens + val raw 47,853,344 → `val_tokens: 47851520` after train_gpt's truncation, matching the record's `train_seed42_rerun_gptq8s.log`.

(If you ever need to *re-tokenize from scratch* — e.g. to vary the SP model — you'd need the source `docs_selected.jsonl` (sha256 `84386dfa…` per HF manifest) plus two non-default settings: edit `SHARD_TOKENS = 100_000_000` on line 66 of `prepare_caseops_data.py`, and pass `--val-docs 50000`. The script is otherwise byte-deterministic.)

### Step 2 — Move prior model outputs out of the way

The script writes fixed paths `final_model.pt` and `final_model.int6.ptz`. Move any prior outputs first so the new run doesn't clobber them:

```bash
mv -n final_model.pt        final_model.layerrope_prev.pt        2>/dev/null || true
mv -n final_model.int6.ptz  final_model.layerrope_prev.int6.ptz  2>/dev/null || true
```

### Step 3 — Smoke test (~3 minutes, no TTT)

Verify the model builds, gate wiring is correct, and step-1 training loss is finite. Stops at 180s wallclock — won't produce a meaningful val_bpb.

`GATE_*_INIT` env vars are stated explicitly here to make the depth-gate inits visible in the run command (they match the file's defaults, so they're redundant — but transparent).

```bash
GATE_ALPHA_INIT=0.0 GATE_BETA_INIT=-0.5 \
GATE_ALPHA_ROT_INIT=0.0 GATE_BETA_ROT_INIT=-0.5 \
GATE_ROPE_BASE_FREQ=100.0 \
DEPTH_GATE_INDEXING=exec_step \
SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
TTT_ENABLED=0 \
ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=180 VAL_LOSS_EVERY=0 \
RUN_ID=layerrope_smoke_seed42 \
torchrun --standalone --nproc_per_node=4 train_gpt_layerrope.py
```

**Post-smoke verification** — inspect `logs/layerrope_smoke_seed42.txt`:

```bash
LOG=logs/layerrope_smoke_seed42.txt

# 1. Hyperparameters block contains gate config
grep -E "^  (gate_|depth_gate_|model_dim:|num_layers:)" $LOG

# 2. Model param count is ~9,200 lower than a stock-record run with the same flags
grep "model_params:" $LOG

# 3. Gates appear in the passthrough(fp16) line of GPTQ output
grep "passthrough" $LOG | tr ',' '\n' | grep -E "depth_gates|gate_alpha|gate_beta|gamma_input|gamma_residual"

# 4. Training did not crash (warmup + at least step 5)
grep -E "warmup_step:|^[0-9]+/[0-9]+ train_loss" $LOG | head -20
```

Expected: ~25 entries beginning with `depth_gates.` in the passthrough line (4 γ vectors + 16 schedule scalars).

### Step 4 — Full single-seed comparison run on 4×H100 NVL (~90 minutes with TTT)

If smoke passed, kick off the real run. Compares against the new record's seed-42 of `quantized_ttt val_bpb = 1.06083288`.

```bash
GATE_ALPHA_INIT=0.0 GATE_BETA_INIT=-0.5 \
GATE_ALPHA_ROT_INIT=0.0 GATE_BETA_ROT_INIT=-0.5 \
GATE_ROPE_BASE_FREQ=100.0 \
DEPTH_GATE_INDEXING=exec_step \
SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
ITERATIONS=4550 MAX_WALLCLOCK_SECONDS=0 \
RUN_ID=layerrope_seed42 \
torchrun --standalone --nproc_per_node=4 train_gpt_layerrope.py
```

**Post-run comparison:**

```bash
LOG=logs/layerrope_seed42.txt
grep -E "pre-quantization|quantized val_loss:|quantized_sliding_window|quantized_ttt" $LOG
echo "Compare against record seed-42 (re-run with GPTQ_RESERVE_SECONDS=8.0):"
echo "  pre-quantization post-ema val_bpb : 1.0608* (close)"
echo "  quantized                  val_bpb : 1.0699* (close)"
echo "  quantized_sliding_window   val_bpb : 1.0635* (close)"
echo "  quantized_ttt              val_bpb : 1.06083288 (TARGET)"
```

### Step 5 — 8×SXM submission attempt

When you have access to 8×H100 SXM, drop the iteration / wallclock overrides (script defaults to `ITERATIONS=20000` and `MAX_WALLCLOCK_SECONDS=600`) and set `--nproc_per_node=8`:

```bash
GATE_ALPHA_INIT=0.0 GATE_BETA_INIT=-0.5 \
GATE_ALPHA_ROT_INIT=0.0 GATE_BETA_ROT_INIT=-0.5 \
GATE_ROPE_BASE_FREQ=100.0 \
DEPTH_GATE_INDEXING=exec_step \
SEED=42 \
CASEOPS_ENABLED=1 \
EMBED_BITS=7 \
SMEAR_GATE_ENABLED=1 \
SPARSE_ATTN_GATE_ENABLED=1 \
MIN_LR=0.1 \
EMBED_CLIP_SIGMAS=15.0 \
MLP_CLIP_SIGMAS=12.0 \
GPTQ_RESERVE_SECONDS=8.0 \
PHASED_TTT_NUM_PHASES=3 \
RUN_ID=layerrope_seed42_8gpu \
torchrun --standalone --nproc_per_node=8 train_gpt_layerrope.py
```

For a leaderboard-eligible artifact, the file would still need `python-minifier` minification (matching the new record's wrapped form). Repeat for `SEED=314` and `SEED=1234` to produce a 3-seed mean.

### Optional ablations (no code edits)

```bash
# Compare exec_step (default) vs physical layer indexing:
DEPTH_GATE_INDEXING=physical RUN_ID=layerrope_physical_seed42 ... torchrun ... train_gpt_layerrope.py

# Disable depth dependence at init (gates start at exp(mag)=1, Θ=1; only RoPE per-pair freq decay remains):
GATE_ALPHA_INIT=0 GATE_BETA_INIT=0 GATE_ALPHA_ROT_INIT=0 GATE_BETA_ROT_INIT=0 \
RUN_ID=layerrope_zero_init_seed42 ... torchrun ... train_gpt_layerrope.py
```

---

## Tuning knobs (without code edits)

| Env var | Default | Purpose |
|---|---|---|
| `GATE_ALPHA_INIT` | `0.0` | Initial α for magnitude schedule (per-position, applied to all 4 positions) |
| `GATE_BETA_INIT` | `-0.5` | Initial β (matches baseline `1/√(ℓ+1)` magnitude curve at exec_step indexing) |
| `GATE_ALPHA_ROT_INIT` | `0.0` | Initial α_rot for rotation schedule |
| `GATE_BETA_ROT_INIT` | `-0.5` | Initial β_rot |
| `GATE_ROPE_BASE_FREQ` | `100.0` | Per-pair RoPE-style base for θ_j(k) = Θ(k) · base^(−2j/H) |
| `DEPTH_GATE_INDEXING` | `exec_step` | `exec_step` (per-visit) or `physical` (per-layer) — see Indexing modes table |
| `SCALAR_LR` | `0.02` | Learning rate for all scalar params, including gates |

Setting `GATE_BETA_INIT=0`, `GATE_BETA_ROT_INIT=0`, `GATE_ALPHA_INIT=0`, `GATE_ALPHA_ROT_INIT=0` gives a near-baseline-equivalent init (`exp(mag)=1`, `Θ=1`) where the only depth dependence at init comes from the RoPE-style per-pair frequency decay — useful as an ablation.

---

## Files

| File | Purpose |
|---|---|
| [`train_gpt_layerrope.py`](train_gpt_layerrope.py) | LayerRoPE on the new record stack |
| [`train_gpt.py`](train_gpt.py) | Unmodified new record source (val_bpb 1.06145) |
| [`README_latest_record.md`](README_latest_record.md) | New record's documentation (PR #1851 reproduction) |
| `README_ours.md` | This file |

Earlier (now-superseded) variants against the **previous** record (val_bpb 1.0810):
- [`train_gpt_complex_rot.py`](train_gpt_complex_rot.py) — linear schedule, **physical** indexing
- [`train_gpt_complex_rot_quadratic.py`](train_gpt_complex_rot_quadratic.py) — adds quadratic Taylor terms

The new `train_gpt_layerrope.py` supersedes both for the current record.

---

## Diagnostic checklist (post-smoke-test)

After the smoke run, verify in `logs/<RUN_ID>.txt`:

- [ ] `Hyperparameters:` block includes `gate_alpha_init`, `gate_beta_init`, `depth_gate_indexing`, etc.
- [ ] `model_params:` line is **9,200 lower** than a record-source run with the same flags (Δ from removing `attn_scale`/`mlp_scale` net of adding γ vectors and 16 scalars).
- [ ] `Quantized weights:` line shows `passthrough (float16)` containing entries beginning with `depth_gates.gamma_input_attn`, `depth_gates.gamma_residual_*`, `depth_gates.gate_alpha_*`, etc.
- [ ] `Total submission size quantized+brotli:` is similar to (slightly smaller than) the record's ~15.95 MB.

---

## Open questions / next steps

- [ ] Smoke-test linear `exec_step` (default) on 4×NVL.
- [ ] Single-seed comparison `quantized_ttt val_bpb` vs new record's seed-42 `1.06083`.
- [ ] Ablate `DEPTH_GATE_INDEXING=physical` for one seed to measure exec_step's contribution specifically.
- [ ] Consider adding LoRA-style adapters to gate scalars so they can adapt during Phased TTT (currently they're frozen during TTT).
- [ ] Optional: add the quadratic `½γ·u²` Taylor term as a togglable variant against this stack (was implemented for the previous record in `train_gpt_complex_rot_quadratic.py`; same approach generalizes).

---

## Credits / lineage

- **PR #1851 record stack** (val_bpb 1.06145) — @aquariouseworkman, @nprime06, @romeerp, @dexhunter, @cocohearts, @abaybektursun, @clarkkev, @Christopher-Lee-McClendon (see [README_latest_record.md](README_latest_record.md))
- **Complex-rotation depth gates** lineage — research from `/scratch/ssrivas9/large-activations/` (in-progress; not yet published)
