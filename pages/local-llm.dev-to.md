---
title: I Turned My M1 MacBook Into an Offline AI Coding Agent - $0 API Cost, Zero Cloud
published: false
tags: #ai, #llm, #privacy, #devex
cover_image: https://media2.dev.to/dynamic/image/width=1000,height=420,fit=cover,gravity=auto,format=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fpwoeloxoze5o3ncnudkp.png
---

The cloud is great — until you hit a rate limit mid-refactor. Or you're on a flight. Or you're working on code that should never leave your machine.

I spent three weeks obsessing over one question: **how close can you actually get to a GPT-4-level agentic coding experience running 100% locally, with zero internet?**

The answer surprised me. My M1 MacBook Pro — no discrete GPU, no cloud subscription, no API key — now runs a 26-billion parameter model that reads my codebase, writes code, applies diffs, and proposes Git changes. Autonomously. Offline.

This post is the exact, reproducible blueprint. Every command is copy-pasteable. Every decision is explained.

> **TL;DR** — I compiled `llama.cpp` with Metal GPU acceleration on an M1 Mac, loaded Google's Gemma-4 26B via Unsloth's quantization, and wired it to OpenCode for a fully agentic, offline coding workflow. Total API cost: **$0**. Data sent to the cloud: **0 bytes**.

---

## Why This Matters Right Now

Most conversations about "local AI" treat it as a hobbyist curiosity — small models, toy tasks, nothing you'd trust on real work. That was true 18 months ago. It isn't anymore.

Three things converged to make this actually viable:

| What Changed | Why It Matters |
|:---|:---|
| **Apple's Unified Memory** | GPU and CPU share the same RAM pool. A 32GB M1 can feed a 26B parameter model to the GPU like a dedicated VRAM machine. |
| **`llama.cpp` + Metal** | CPU/GPU inference optimized specifically for Apple Silicon. Not a port — built for it. |
| **Unsloth quantizations** | Aggressive, quality-preserving quantization that fits Gemma-4 26B into ~16GB without meaningful quality loss. |

Put those three together and a standard developer laptop becomes a credible inference machine.

---

## The Hardware and the Brain

I'm running this on an **M1 MacBook Pro with 32GB of unified memory**.

For the model, I chose [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) after reviewing the [hardware requirements for Gemma-4](https://unsloth.ai/docs/models/gemma-4#hardware-requirements). Here's why each component of that model name matters:

| Component | Why It Matters |
|:---|:---|
| **Unsloth** | The leading framework for efficient LLM quantization, with recent bugfixes not yet in the [ggml-org](https://huggingface.co/ggml-org/gemma-4-26B-A4B-it-GGUF) or [Google](https://huggingface.co/google/gemma-4-26B-A4B-it) releases. |
| **Gemma-4 26B** | A massive, highly capable architecture from Google DeepMind. |
| **Instruction-Tuned (it)** | Crucial for agentic workflows — the model follows complex commands, not just predicts text. |
| **GGUF** | The optimized file format required for local CPU/Metal execution via `llama.cpp`. |

At a Q4 quantization, the 26B model requires roughly 15–16GB of memory. On 32GB unified memory, that leaves more than enough overhead for macOS, your IDE, and OpenCode running simultaneously.

---

## Prerequisites

Everything below is available through Homebrew or pip. No manual compilation required except `llama.cpp` itself.

```bash
# Xcode Command Line Tools (required for cmake, git, and Metal framework headers)
xcode-select --install

# Core build dependencies
brew install cmake libomp

# Hugging Face CLI for model downloads
pip install huggingface_hub hf_transfer

# Parallel download engine (optional but strongly recommended for large models)
brew install aria2

# OpenCode — the agentic coding orchestrator
brew install anomalyco/tap/opencode
```

With these in place, every step below works on a clean macOS install.

---

## Step 1: Compile `llama.cpp` from Scratch with Metal

You *could* download a pre-built binary. But if you want every drop of performance from the M1's Metal GPU framework, build from source.

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

The key flag is `-DGGML_METAL=ON` — this compiles with Apple's Metal GPU framework. The `-j$(sysctl -n hw.ncpu)` parallelizes the build across all CPU cores.

When you compile directly on the M1, inference speeds jump dramatically. You aren't just running code — you're running code hyper-optimized for your specific silicon.

After the build, create symlinks to keep commands clean:

```bash
ln -s ./llama.cpp/build/bin/llama-cli llama-cli
ln -s ./llama.cpp/build/bin/llama-server llama-server
```

`llama-cli` handles interactive terminal prompts. `llama-server` is the HTTP inference server that exposes an OpenAI-compatible API — the piece that connects to OpenCode. Both are built from the same source tree.

### Validate the Build Before the Big Download

Before downloading the massive 18GB Gemma-4 model, validate the entire pipeline with a smaller model: [NVIDIA Nemotron-3-Nano-4B](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF) at Q8 quantization, just 3.9GB.

**This step is not optional.** You don't want to wait hours for an 18GB download only to discover your build is broken.

![Downloading NVIDIA Nemotron-3-Nano-4B as a pipeline validation step](https://sectumpsempra.github.io/assets/images/blog_nemotron_nano_download.png)

Booting the server with the smaller model confirms what you need to see: Metal framework fully initialized, unified memory detected, all GPU families registered.

![Metal GPU framework initialization on the M1 — unified memory confirmed, bfloat support active, and recommendedMaxWorkingSetSize showing access to the full 32GB memory pool](https://sectumpsempra.github.io/assets/images/blog_nemotron_llama_server.png)

The critical line in that output: `has unified memory = true`. The `recommendedMaxWorkingSetSize` of roughly 26,800 MB tells you exactly how much VRAM the Metal backend can access — and on the M1, it draws directly from system RAM.

Pipeline is solid. Now bring in the real model.

---

## Step 2: Download Gemma-4 26B Weights

Because we're building an offline environment, the model file needs to be local.

```bash
hf download unsloth/gemma-4-26B-A4B-it-GGUF \
  --local-dir unsloth/gemma-4-26B-A4B-it-GGUF \
  --include "*mmproj-BF16*" \
  --include "*UD-Q4_K_XL*"
```

The `--include` filters are important:
- `*mmproj-BF16*` — the multimodal vision projector, giving the model the ability to understand images alongside code
- `*UD-Q4_K_XL*` — the sweet spot quantization for quality vs. memory on 32GB (~15.9GB on disk)

### Fair Warning: 18.3GB Downloads Are Fragile

![An 18.3GB download failing mid-transfer via the default hf CLI — hours of progress, gone](https://sectumpsempra.github.io/assets/images/blog_failed_download.png)

My download crawled to 519KB/s before failing entirely. The default `hf download` CLI supports resuming in theory, but in practice it's fragile on large files over unstable connections.

Switch to [`hfd.sh`](https://gist.github.com/yeahjack/31f542ee6cab3c3e2c30594b7693cb22#file-hfd-sh) with `aria2c` as the engine. Unlike the default CLI, `aria2c` tracks per-segment progress in `.aria2` control files — a dropped connection picks up exactly where it left off instead of restarting the entire file.

```bash
./hfd.sh unsloth/gemma-4-26B-A4B-it-GGUF \
  --local-dir unsloth/gemma-4-26B-A4B-it-GGUF \
  --include "*mmproj-BF16*" \
  --include "*UD-Q4_K_XL*" \
  --tool aria2c -x 16 -n 8
```

`-x 16` opens 16 connections per server. `-n 8` splits each file into 8 parallel segments. On a decent connection, this is dramatically faster and more resilient than the default downloader.

---

## Step 3: Wire the Brain to OpenCode

Having a powerful local LLM is interesting. Having it autonomously write, edit, and debug your code is a different thing entirely. That's where [OpenCode](https://opencode.ai/docs/) comes in.

OpenCode bridges the local LLM and your codebase. The key insight: `llama-server` exposes an OpenAI-compatible `/v1/chat/completions` endpoint out of the box. OpenCode's `@ai-sdk/openai-compatible` adapter speaks that protocol natively. No custom prompt templates, no manual token wrangling — the chat template baked into the GGUF handles everything at the server level.

Create `opencode.json` at your project root:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8001"
      },
      "models": {
        "gemma-4:26b-a4b-it": {
          "name": "Gemma-4-26B-A4B-it (local)",
          "limit": {
            "context": 32000,
            "output": 65536
          }
        },
        "nvidia-nemotron-3-nano:4b": {
          "name": "NVIDIA-Nemotron-3-Nano-4B (local)",
          "limit": {
            "context": 32000,
            "output": 65536
          }
        }
      }
    }
  }
}
```

A few things worth noting:

- The `baseURL` points to `127.0.0.1:8001` where `llama-server` will listen
- Context is set to 32K tokens — the model supports up to 262K, but 32K is a practical ceiling for stable agentic sessions on 32GB RAM
- The second model (Nemotron-3 Nano 4B) is configured as a lightweight alternative for fast, low-overhead tasks

Verify the model is running correctly by checking the llama.cpp web interface at `http://127.0.0.1:8001`:

![Gemma-4 26B loaded and serving — 15.9GB model, 25.23B parameters, 262K context window, Vision modality confirmed](https://sectumpsempra.github.io/assets/images/blog_model_details.png)

25.23 billion parameters. A 262,144-token context window. Vision capability. Running from a file on local disk, served over localhost. No cloud, no API key, no rate limit.

---

## Step 4: Full Offline Agentic Coding

With the model downloaded, `llama.cpp` compiled, and `opencode.json` locked in, I turned off Wi-Fi.

**Zero internet. Zero API calls. Zero data leaving the machine.**

Open two terminals:

```bash
# Terminal 1: Start the llama.cpp inference server with Gemma-4
./llama-server \
  --model unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --temp 0.6 \
  --top-p 0.95 \
  --alias "gemma-4-26B" \
  --port 8001 \
  -ngl 99 -t 8 -b 512 --mmap
```

Still validating with Nemotron before the full download? Same flags work:

```bash
# Alternative: lighter Nemotron model for testing
./llama-server \
  --model unsloth/NVIDIA-Nemotron-3-Nano-4B-GGUF/NVIDIA-Nemotron-3-Nano-4B-Q8_0.gguf \
  --temp 0.6 \
  --top-p 0.95 \
  --alias "nvidia/nemotron-3-nano-4B-GGUF" \
  --port 8001 \
  --reasoning on \
  -ngl 99 -t 8 -b 512 --mmap
```

Note the `--reasoning on` flag on Nemotron — it activates a built-in chain-of-thought mode that improves output quality on complex tasks. Useful for validating multi-step reasoning before scaling up to Gemma-4.

```bash
# Terminal 2: Launch OpenCode
opencode
```

Here's what each server flag does:

| Flag | Purpose |
|:---|:---|
| `-ngl 99` | Offloads all model layers to the Metal GPU |
| `-t 8` | Sets 8 CPU threads for operations that fall back to CPU |
| `-b 512` | Controls batch size for prompt processing |
| `--mmap` | Memory-maps the model file — macOS manages paging without loading all 15.9GB upfront |
| `--temp 0.6` | Slightly below default for more deterministic code generation |
| `--top-p 0.95` | Nucleus sampling — keeps output focused while allowing creativity |

![OpenCode analyzing a project's architecture, generating documentation, and proposing Git changes across 5 files — powered locally by gemma-4-26B in 45 seconds](https://sectumpsempra.github.io/assets/images/blog_opencode_web.png)

The result: Gemma-4 26B analyzed my codebase, understood the architecture of local files, and began writing, diffing, and applying code. In the screenshot above, it analyzed an `architect.py` file, broke down Pydantic data models, explained the `run_architect` function flow, and proposed 5 Git changes across the project.

The M1 pushed out tokens fast enough for real-time development. The footer confirms it: `gemma-4-26B`, 45 seconds for a full architectural analysis and code generation pass.

---

## What This Actually Means

We are crossing a threshold.

For two years, the industry assumed truly capable AI agents require data centers. This experiment proves otherwise.

**For engineering teams working on sensitive codebases** — defense, healthcare, fintech — this means AI coding assistants without a single byte crossing a network boundary. No SOC 2 reviews for another SaaS vendor. No data processing agreements. No trust boundaries to negotiate.

**For engineering leaders**, the math is compelling:
- Zero marginal API cost per developer
- Zero vendor lock-in
- Works identically on an airplane, in a SCIF, or behind an air-gapped network

**For individual developers**, the practical reality: you can now run a frontier-class coding agent on hardware you already own, using models that are openly licensed, with no subscription, no quota, no latency spikes on someone else's overloaded GPU cluster.

Absolute privacy and top-tier AI capability are no longer mutually exclusive.

---

## What I'm Exploring Next

This setup is a foundation, not a ceiling.

**Larger context windows.** The 32K context in `opencode.json` is conservative. With careful memory management and `llama.cpp`'s Flash Attention support, 128K+ is feasible on 32GB for longer agentic sessions.

**Multi-model routing.** Running Nemotron for fast, lightweight tasks and Gemma-4 for heavy reasoning — switching between models based on task complexity, all locally. Think of it as a cheap/smart tier system without the cloud bill.

**Fine-tuning on proprietary code.** Unsloth supports LoRA and QLoRA fine-tuning. Training a domain-specific adapter on your team's codebase and merging it into the GGUF gives you a model that *thinks* in your architecture and naming conventions.

**Team-wide access.** Embed `llama-server` in a container behind your internal network so the entire team gets local AI without each developer maintaining their own build.

---

## The Full Stack, in One Place

For anyone who wants to reproduce this exactly:

| Component | What It Is | Link |
|:---|:---|:---|
| `llama.cpp` | Metal-accelerated inference engine | [github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Gemma-4 26B (Unsloth) | The model, Q4_K_XL quantization | [huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF) |
| OpenCode | Agentic coding orchestrator | [opencode.ai/docs](https://opencode.ai/docs/) |
| `hfd.sh` | Reliable large-file downloader | [gist.github.com/yeahjack](https://gist.github.com/yeahjack/31f542ee6cab3c3e2c30594b7693cb22#file-hfd-sh) |

The tools are here. The models are capable enough. The only question is what you build with them.

---

If you found this useful, the full write-up with additional context lives on [my site](https://sectumpsempra.github.io). Questions, improvements, or your own local AI stack? Drop them in the comments — I'd genuinely like to hear what you're running.

*Written by [Amit Bhatt](https://linkedin.com/in/amit-bhatt) — [GitHub](https://github.com/sectumpsempra)*
