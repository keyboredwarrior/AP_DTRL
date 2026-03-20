# AP_DTRL (LibTorch-ready)

This repository contains a physics + character-control framework for terrain locomotion, now refactored to:

- remove Caffe usage,
- support a LibTorch policy backend,
- allow runtime model-mode switching between:
  - **classic RL** (`policy_model: classic_rl`),
  - **deep RL** (`policy_model: deep_rl`),
  - **deep RL + attention** (`policy_model: deep_rl_attention`).

## 1) Prerequisites (Visual Studio 2026 / MSVC)

Install:

- Visual Studio 2026 with **Desktop development with C++**.
- CMake 3.24+
- vcpkg (recommended) for:
  - `bullet3`
  - `glfw3`
  - `jsoncpp`
- LibTorch for Windows (C++ distribution, matching your MSVC runtime).

## 2) Local Eigen setup (standalone)

Eigen is expected as a **local vendored dependency**:

- `external/eigen/Eigen/...`

The code now includes local headers (for example from `util/MathUtil.h` and `anim/IKSolver.h`) instead of `#include "Eigen/..."` from system paths.

## 3) Configure with CMake

From `x64 Native Tools Command Prompt for VS 2026`:

```bat
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 ^
  -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ^
  -DAP_DTRL_ENABLE_TORCH=ON ^
  -DAP_DTRL_BUILD_APP=OFF ^
  -DTorch_DIR=C:/libtorch/share/cmake/Torch
```

Then build:

```bat
cmake --build build --config Release
```

## 4) Policy config format

`cNeuralNet::LoadNet(...)` now supports a JSON config that can switch algorithm mode without code changes:

```json
{
  "policy_model": "deep_rl",
  "input_size": 128,
  "output_size": 32,
  "model_file": "models/policy.pt",
  "scale_file": "models/policy_scale.json"
}
```

### Supported `policy_model`

- `classic_rl`
  - Uses local Q-table entries in the same net JSON (`q_table` array).
  - No LibTorch module required.
- `deep_rl`
  - Uses LibTorch scripted model (`model_file`).
- `deep_rl_attention`
  - Uses LibTorch model, with a lightweight attention-style pre-weighting on state features before forward pass.

## 5) Notes about current framework reuse

The existing controller hierarchy is preserved and reused:

- `cNNController` remains the common policy entry point.
- `cBaseControllerQ` remains available for classic value-based policy flow.
- `cBaseControllerCacla` and `cBaseControllerMACE` continue to use neural-policy pathways.
- `cMACETrainer` utility methods are now provided locally in `learning/MACETrainer.h`.

## 6) Running

The current repository snapshot does not include the original `scenarios/` source tree referenced by `Main.cpp`. Build the core library (`ap_dtrl_core`) first. If you restore/add scenario files, set:

- `-DAP_DTRL_BUILD_APP=ON`

and rebuild.

