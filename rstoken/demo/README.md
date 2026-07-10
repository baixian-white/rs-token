# RS-Token Adaptive Communication Demo

This demo runs the trained RS-Token checkpoint as an interactive UAV remote-
sensing communication link. It performs real single-image RVQ encoding,
bit-level AWGN/Rayleigh corruption, optional rate-1/2 sparse LDPC decoding,
automatic prefix selection, L0 task inference, and progressive reconstruction.

## Run

From PowerShell:

```powershell
cd rstoken
.\demo\run_demo.ps1
```

The launcher chooses the first free port starting at `7860`, opens the browser,
waits for the model health check, and keeps the model resident on the GPU in a
hidden process. Use `-NoBrowser` for a server-only run or `-Device cpu` when
CUDA is unavailable. Stop the background service with:

```powershell
.\demo\stop_demo.ps1
```

The committed frontend is already built. On a fresh machine, `run_demo.ps1`
automatically invokes `setup.ps1` when the frozen L0 probe is absent. Probe
export uses the paper's clean AID train/test protocol and is only performed
once.

## Scientific Scope

- The unprotected channel uses the same closed-form BPSK BER model as E17/E36.
- The LDPC option is the project's custom rate-1/2 systematic sparse code with
  min-sum BP decoding. It is not 5G NR LDPC.
- `k=1` task predictions use the frozen L0 BoW linear probe. Predictions from
  reconstructed pixels use the independently trained clean-AID ResNet34.
- RemoteCLIP is a training-only teacher and is not loaded by the demo.
- The current downstream task is scene classification, not detection or
  segmentation. Uploaded images outside the AID domain may be out of
  distribution.

## Layout

```text
demo/
  artifacts/       frozen L0 probe
  backend/         FastAPI service, inference engine, adaptive policy
  frontend/        React/TypeScript source
  static/          production frontend served by FastAPI
  setup.ps1        one-time dependency and artifact preparation
  run_demo.ps1     single-command launcher
```
