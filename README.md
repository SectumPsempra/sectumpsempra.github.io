---
layout: default
title: "AI Engineering"
description: "Deep dives on building real systems with AI - from Apple Silicon inference to production multi-agent pipelines."
---

<p class="blog-intro">Engineering deep dives on building real systems with AI. No fluff, no toy demos - just the exact decisions, code, and trade-offs behind production-grade AI tooling.</p>
<div class="post-grid">

  <article class="post-card">
    <a href="/offline-coding-agent/" class="card-image-wrap" aria-label="Read: Severing the API Cord">
      <img
        src="/assets/images/offline-coding-agent/blog_opencode_web.png"
        alt="OpenCode analyzing a codebase - powered by Gemma-4 26B running fully offline on an M1 Mac"
        class="card-image">
    </a>
    <div class="card-body">
      <div class="card-tags">
        <span class="tag tag-local">Local LLM</span>
        <span class="tag tag-privacy">Privacy-First</span>
        <span class="tag tag-metal">Apple Metal</span>
      </div>
      <h2 class="card-title"><a href="/offline-coding-agent/">Severing the API Cord</a></h2>
      <p class="card-meta">⏱ 10 min read &nbsp;·&nbsp; llama.cpp &middot; Gemma-4 26B &middot; OpenCode</p>
      <p class="card-excerpt">I compiled <code>llama.cpp</code> with Metal GPU acceleration on an M1 Mac, loaded Google's Gemma-4 26B via Unsloth's quantization, and wired it to OpenCode for a fully agentic, offline coding workflow. Total API cost: <strong>$0</strong>. Data sent to the cloud: <strong>0 bytes</strong>.</p>
      <a href="/offline-coding-agent/" class="read-btn">Read Post &rarr;</a>
    </div>
  </article>

  <article class="post-card">
    <a href="/infrasquad/" class="card-image-wrap" aria-label="Read: InfraSquad">
      <img
        src="/assets/images/infrasquad/infrasquad_hero.png"
        alt="InfraSquad - four AI agents collaborating on AWS infrastructure design and security auditing"
        class="card-image">
    </a>
    <div class="card-body">
      <div class="card-tags">
        <span class="tag tag-agents">Multi-Agent</span>
        <span class="tag tag-iac">Terraform</span>
        <span class="tag tag-devops">LangGraph</span>
      </div>
      <h2 class="card-title"><a href="/infrasquad/">InfraSquad</a></h2>
      <p class="card-meta">⏱ 15 min read &nbsp;·&nbsp; LangGraph &middot; AWS &middot; Security Automation</p>
      <p class="card-excerpt">Four specialized AI agents collaborate in a LangGraph state machine to architect, write Terraform, audit security findings, and render architecture diagrams - all from a single plain-English description, with automatic remediation loops.</p>
      <a href="/infrasquad/" class="read-btn">Read Post &rarr;</a>
    </div>
  </article>

</div>
