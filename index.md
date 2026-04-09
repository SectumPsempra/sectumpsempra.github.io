---
layout: default
title: "AI Engineering"
description: "Building real systems with AI - from Apple Silicon inference to production multi-agent pipelines."
---

A hands-on engineering blog. Every post covers a complete, working system - the decisions behind it, the code that runs it, and the failure modes that shaped it.

---

## ⚡ Severing the API Cord

[![OpenCode analyzing a codebase, powered by Gemma-4 26B running fully offline on an M1 Mac](/assets/images/offline-coding-agent/blog_opencode_web.png)](/offline-coding-agent/)

`llama.cpp` &nbsp;·&nbsp; `Gemma-4 26B` &nbsp;·&nbsp; `OpenCode` &nbsp;·&nbsp; `Apple Metal` &nbsp;·&nbsp; *~10 min read*

I compiled `llama.cpp` with Metal GPU acceleration on an M1 Mac, loaded Google's Gemma-4 26B via Unsloth's quantization, and wired it to OpenCode for a fully agentic, offline coding workflow. Total API cost: **$0**. Data sent to the cloud: **0 bytes**.

[**Read post →**](/offline-coding-agent/)

---

## 🤖 InfraSquad - Multi-Agent AWS Infrastructure

[![InfraSquad - four AI agents collaborating on cloud infrastructure design and security auditing](/assets/images/infrasquad/infrasquad_hero.png)](/infrasquad/)

`LangGraph` &nbsp;·&nbsp; `Terraform` &nbsp;·&nbsp; `AWS` &nbsp;·&nbsp; `Security Automation` &nbsp;·&nbsp; *~15 min read*

Four specialized AI agents collaborate in a cyclic LangGraph state machine to architect, write Terraform, audit security with remediation loops, and render architecture diagrams - from a single plain-English description.

[**Read post →**](/infrasquad/)
