---
title: "I Let 4 AI Agents Design AWS Infrastructure, Write the Terraform, and Audit Their Own Security. Here's What Broke."
published: false
tags: ai, devops, terraform, langgraph
cover_image: https://sectumpsempra.github.io/assets/images/infrasquad/infrasquad_hero.png
---

Designing cloud infrastructure usually takes three meetings.

One with the architect to decide which services to use. One with the DevOps engineer to actually write the Terraform. One with the security team to explain, again, why `0.0.0.0/0` is not an acceptable production CIDR.

By the time all three conversations happen, the architecture diagram is already out of date.

So we asked a different question: what if all four roles ran as AI agents in a single automated pipeline?

You type your requirements in plain English. You get back deployable Terraform HCL, a security audit with specific remediation guidance, and a rendered architecture diagram. In one shot, without the meetings.

That's InfraSquad. This post is about what we learned building it, what broke badly, and what we would tell ourselves at the start.

> **TL;DR:** InfraSquad is a multi-agent system built on LangGraph. Four agents collaborate in a cyclic state machine. Security findings loop back to the DevOps agent for fixes, capped at three cycles. Without that cap, the loop runs forever. We learned this during testing. The code is open source at [Andela-AI-Engineering-Bootcamp/infrasquad](https://github.com/Andela-AI-Engineering-Bootcamp/infrasquad).

---

## Meet the Squad

Four agents. One shared pipeline. Here is what each one actually does:

| Agent | Responsibility | Output |
|:---|:---|:---|
| **Product Architect** | Reads your requirements, considers scale, compliance, cost | A numbered AWS architecture plan |
| **DevOps Engineer** | Translates the plan into code; fixes security findings when sent back | Valid Terraform HCL |
| **Security Auditor** | Runs tfsec or checkov via MCP; classifies every finding by severity | A structured JSON security report |
| **Visualizer** | Reads the final plan and code after security passes | A Mermaid architecture diagram rendered to PNG |

The critical word in that table is "sent back." The Security Auditor does not just generate a report and hand it off. It can send the DevOps Engineer back to fix its own code. That feedback loop is the most interesting design decision in the system. It is also how we nearly created an infinite loop on the second day of integration testing.

![InfraSquad -- four AI agents collaborating on cloud infrastructure design](https://sectumpsempra.github.io/assets/images/infrasquad/infrasquad_hero.png)

---

## The Pipeline (and the Two Places It Can Loop)

Here is the full state machine:

![InfraSquad pipeline diagram showing the LangGraph state machine -- validate_input, architect, devops, validate_output, security, visualizer, with two loop-back arrows for HCL errors and security findings](https://sectumpsempra.github.io/assets/images/infrasquad/infrasquad_pipeline_diagram.png)

The happy path is straightforward:

1. **validate_input** runs three checks before anything expensive happens
2. **architect** produces a numbered AWS architecture plan
3. **devops** writes Terraform HCL from that plan
4. **validate_output** checks the HCL deterministically for forbidden patterns and structural validity
5. **security** scans with tfsec or checkov via MCP
6. **visualizer** renders the architecture as a Mermaid diagram

Two of those six nodes can send the pipeline backwards. That is intentional. It is also dangerous if you do not cap the cycle count, which we did not do initially.

All six agents share a single typed state object:

```python
class AgentState(TypedDict, total=False):
    user_request: str
    architecture_plan: str
    terraform_code: str
    security_report: dict[str, Any]
    security_passed: bool
    remediation_count: int
    hcl_remediation_count: int
    hcl_validation_errors: list[str]
    current_phase: str
```

The `total=False` matters. Without it, every agent would need to set every field, even fields it knows nothing about. With it, agents only write what they own. Silent downstream failures from unexpected `None` values were the most frustrating class of bug we hit early on.

---

## We Almost Created an Infinite Loop on Day Two

During integration testing, we ran a request for an internet-facing Application Load Balancer.

The Security Auditor flagged it: `AVD-AWS-0107, HIGH -- security group allows unrestricted ingress from 0.0.0.0/0`. The DevOps agent tried to fix it. The Security Auditor re-scanned. Same finding. The DevOps agent tried again. Same finding.

The problem: a public ALB is supposed to have unrestricted public ingress. That is what "internet-facing" means. The security finding was technically correct and permanently unfixable given the design intent. The LLM had no way to distinguish "security issue to remediate" from "accepted design constraint."

Without an exit condition, this loop runs forever.

Here is what the routing logic looks like with the cap in place:

```python
def route_after_security(state: AgentState) -> Literal["visualizer", "devops"]:
    if state.get("security_passed", False):
        return "visualizer"
    if state.get("remediation_count", 0) >= settings.max_remediation_cycles:
        return "visualizer"  # move on regardless
    return "devops"
```

After three cycles, the pipeline proceeds with whatever state it has. Unresolved findings appear as advisory warnings in the Security tab, not hard failures. The same cap exists on the HCL validation loop.

**Add your cycle caps before your first integration test.** Not after. You will hit this case.

---

## The One Bad Habit We Couldn't Engineer Away

Every model we tested had the same behavior: for any internet-facing resource, it generated `0.0.0.0/0` as the security group ingress CIDR. Even with explicit instructions in the system prompt. Even with examples. Even with counter-examples.

We tried prompt engineering for weeks. The model would acknowledge the constraint, then generate `0.0.0.0/0` anyway on the next call.

So we stopped fighting it and added a deterministic sanitizer that runs on the DevOps agent's output before validation even starts:

```python
_CIDR_SANITISATIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r'"0\.0\.0\.0/0"'), '"10.0.0.0/8"'),
    (re.compile(r'"::/0"'),         '"fc00::/7"'),
]

def _sanitize_hcl(hcl: str) -> str:
    for pattern, replacement in _CIDR_SANITISATIONS:
        hcl = pattern.sub(replacement, hcl)
    return hcl

clean_hcl = _sanitize_hcl(output.terraform_hcl)
```

`10.0.0.0/8` is a broad internal placeholder. Operators narrow it before deploying to production.

This single function broke the HCL validation loop for the most common case. First-pass generations stopped triggering the guardrail on CIDR issues almost entirely. When the guardrail fires now, it catches a genuine structural problem.

**When a model reliably produces the same wrong output, fix it deterministically. Do not prompt your way out of a consistency problem.**

---

## Three Questions Before the LLM Sees Anything

The most expensive mistake in an agentic pipeline is burning tokens on requests that should never reach the agents. InfraSquad catches these at three layers before anything expensive runs.

![Three layers of input validation -- chitchat detection, keyword matching, and LLM classification as a narrowing funnel](https://sectumpsempra.github.io/assets/images/infrasquad/infrasquad_validation_funnel.png)

**Layer 1: Chitchat detection (zero cost)**

A frozenset of 40+ conversational tokens returns immediately. "Thanks", "ok cool", "sounds good", a thumbs-up emoji -- none of these should reach the architect.

```python
_CHITCHAT_TOKENS: frozenset[str] = frozenset({
    "cool", "okay", "ok", "sure", "thanks", "thank you",
    "hi", "hello", "great", "awesome", "got it", "yep", ...
})

def _is_chitchat(text: str) -> bool:
    normalized = text.strip().lower().rstrip("!.,?")
    tokens = [t.strip("!.,?") for t in normalized.split()]
    return bool(tokens) and len(tokens) <= 4 and all(t in _CHITCHAT_TOKENS for t in tokens)
```

No LLM call. No latency. Instant return.

**Layer 2: Keyword matching (zero cost)**

A compiled regex matches 45 AWS infrastructure keywords. Two or more matches skip the LLM check entirely. "VPC with RDS Postgres and ALB" is obviously a valid request. Spending 2 seconds and tokens to confirm this is wasteful.

```python
elif keyword_match_count(user_request) >= 2:
    # High confidence -- skip the LLM round-trip (~2s saved per clear request)
    is_valid = True
```

About 70% of valid requests take this fast path.

**Layer 3: LLM plausibility (borderline cases only)**

Single-keyword matches are genuinely ambiguous. "Server" could be valid. "AWS tomato server" should not be. For these, a lightweight LLM call returns one of three outcomes:

```python
class _FirstMessageClassification(BaseModel):
    intent: Literal["proceed", "clarify", "reject"]
```

`clarify` triggers a helpful guidance message. `reject` returns a polite explanation. Both avoid running the full pipeline on nonsense input.

There is a catch. In active conversations, keyword matching stops working correctly. "Explain the Terraform code" contains the word "Terraform" but is clearly a follow-up question, not a new generation request. So in active sessions, we switch to a full intent classifier that distinguishes `new_generation`, `follow_up`, and `off_topic`.

![InfraSquad handling an off-topic query -- the guardrail returns a helpful explanation without triggering the pipeline](https://sectumpsempra.github.io/assets/images/infrasquad/handling_off_topic_query.png)

---

## The Security Check That Does Not Trust the LLM

Two patterns are blocked by hardcoded regex, independent of everything else:

```python
_ADMIN_ACCESS_PATTERN = re.compile(r"AdministratorAccess", re.IGNORECASE)
_STAR_POLICY_PATTERN  = re.compile(r'"Action"\s*:\s*"\*"')
```

`AdministratorAccess` policies and wildcard IAM actions are blocked regardless of what the model thought it generated. Not by the HCL guardrail. Not by the Security Auditor. By a function that runs on every output, unconditionally.

The reason for running this separately from the guardrail: the HCL guardrail checks for `AdministratorAccess` in a string pattern that could miss an IAM policy embedded inside a heredoc JSON block. The standalone regex catches it regardless of context.

Two independent checks. Neither relying on the other being correct.

---

## The HCL Guardrail: Before Security Even Runs

Before the Terraform reaches the Security Auditor, it passes through a deterministic validator. This runs on every generation -- first pass and every remediation:

```python
_FORBIDDEN_PATTERNS = [
    (r"AdministratorAccess", "Uses AdministratorAccess IAM policy"),
    (r"0\.0\.0\.0/0",        "Contains 0.0.0.0/0 CIDR -- opens resource to the internet"),
    (r"public\s*=\s*true",   "Sets public access to true"),
]
```

It also checks structural validity: `provider` and `resource` blocks must be present, resource signatures must be well-formed, and braces must balance. Any failure sends the code back to the DevOps agent with the specific error list attached to the next prompt.

The CIDR sanitizer runs before this check. That is intentional. Remove `0.0.0.0/0` before validation, so the guardrail only fires on real structural problems.

---

## What the Pipeline Actually Produces

Here is a real run. Request: "VPC with an RDS Postgres instance, an Application Load Balancer, and a Redis caching layer."

The DevOps agent follows a security baseline baked into its system prompt. S3 buckets get KMS encryption, versioning, and public access blocks by default. RDS gets `storage_encrypted = true`, `deletion_protection = true`, and 7-day backup retention. ElastiCache gets encryption at rest and in transit. VPCs get flow logs.

![Terraform HCL generated by the DevOps agent in the InfraSquad UI, showing encryption and security defaults applied](https://sectumpsempra.github.io/assets/images/infrasquad/response_with_code.png)

After the security check passes, the Visualizer reads the finalized plan and code and generates a Mermaid diagram:

![InfraSquad UI showing the generated Mermaid architecture diagram as source and rendered PNG](https://sectumpsempra.github.io/assets/images/infrasquad/response_with_mermaid_diagram.png)

If `mmdc` is installed, the Mermaid source renders directly to PNG:

![Rendered architecture diagram -- VPC, ALB, EC2 Auto Scaling Group, RDS Multi-AZ, ElastiCache, and S3 as a flowchart](https://sectumpsempra.github.io/assets/images/infrasquad/architecture_diagram.png)

If `mmdc` is not installed, the Mermaid source is saved as-is. It is still fully useful -- paste it into any Mermaid viewer and you get the diagram.

---

## The Security Audit Loop: How It Actually Works

When the Security Auditor finds issues, it does not just list them. It produces a structured prompt that becomes the DevOps agent's next input:

```
MANDATORY SECURITY REMEDIATION -- 3 finding(s)
Fix EVERY numbered item below. Do NOT skip any.

Finding 1. [HIGH] AVD-AWS-0107 - aws_security_group.app_sg
   Issue: Security group allows unrestricted ingress on port 443
   Fix:   Restrict ingress to specific CIDR ranges or security group references.

Finding 2. [HIGH] AVD-AWS-0132 - aws_s3_bucket.assets
   Issue: S3 bucket does not use KMS encryption with a customer-managed key
   Fix:   Add aws_kms_key + aws_s3_bucket_server_side_encryption_configuration
```

The DevOps agent sees `MANDATORY SECURITY REMEDIATION` in its next prompt and treats every numbered item as a required fix.

`security_passed = True` only when there are zero CRITICAL and zero HIGH findings. MEDIUM and LOW findings get reported but do not block the pipeline. The visualization still renders.

---

## Why LangGraph Over CrewAI or AutoGen

This came down to one question: does the framework support cycles with explicit state management and typed contracts?

| Framework | Cyclic workflows | Typed shared state | Explicit retry caps |
|:---|:---|:---|:---|
| **LangGraph** | Native conditional edges | TypedDict, full control | Direct in routing logic |
| **CrewAI** | Workarounds required | Role-based model | Not built-in |
| **AutoGen** | Conversation-driven | Implicit | Not built-in |

The security remediation loop is cyclic by design. The Security Auditor sends the DevOps Engineer back; the DevOps Engineer generates new code; the new code gets re-scanned. Both CrewAI and AutoGen require workarounds for this pattern. LangGraph's conditional edges handle it natively.

The typed state was also non-negotiable. Without a clear contract on what each agent receives and produces, integration failures are silent. An agent gets `None` where it expected a string and fails three nodes downstream with a cryptic error. `TypedDict` with `total=False` gives every agent a contract it cannot accidentally break.

---

## External Tools Through MCP

tfsec and `mmdc` (Mermaid rendering) run as MCP tools, not direct imports. The agent calls a tool through a protocol.

This looks like over-engineering for a project at this scale. The argument for it: tfsec and `mmdc` are external processes that can timeout, crash, or produce unexpected output. Wrapping them in an MCP tool forces explicit failure handling at every call site.

```python
result = _try_tfsec(tmpdir) or _try_checkov(tmpdir)
```

tfsec unavailable? Try checkov. checkov unavailable? LLM security review with the full security system prompt. `mmdc` unavailable? Save Mermaid source. Every external dependency ended up with a fallback path, which would not have happened if they were direct imports.

The MCP server also runs independently. It can be swapped or extended without touching any agent code.

---

## Five Things We'd Tell Ourselves at the Start

**1. Hard-cap your cycles before the first integration test.**
You will hit the infinite loop case. Probably on a public-facing resource where the security finding is technically correct and architecturally intentional. Add the counter before you need it.

**2. Regex beats prompting for deterministic security invariants.**
If a property can be expressed as a pattern, enforce it with code. LLM compliance on security constraints is probabilistic. Code compliance is guaranteed. The CIDR sanitizer took 10 lines to write and immediately reduced first-pass HCL failures by the majority.

**3. Typed state is not optional in multi-agent systems.**
Silent failures are the worst kind. A `TypedDict` with `total=False` is a contract every agent signs. Without it, you are debugging `None` errors three nodes downstream and trying to reconstruct which agent set which field when.

**4. Pydantic schema retry saves more than you expect.**
Without `invoke_with_schema_retry`, the pipeline fails silently on every malformed JSON response. With it, about 80% of schema failures resolve on the first retry with an error correction prompt. Make this load-bearing from day one.

**5. Input validation saves more tokens than it looks like it will.**
Chitchat and off-topic requests are common in demo environments. Every one that reaches the architect burns tokens before returning an unhelpful or confusing response. The three-layer guardrail means only genuine infrastructure requests reach the expensive part of the pipeline.

---

## Run It Yourself

```bash
git clone https://github.com/Andela-AI-Engineering-Bootcamp/infrasquad.git
cd infrasquad
uv venv && uv sync
```

Add your OpenRouter API key to `.env`:

```bash
OPENROUTER_API_KEY=sk-or-...
```

Start the UI:

```bash
python app.py              # localhost:7860
python app.py --share      # public Gradio URL
python app.py --port 8080  # custom port
```

The default model is `openai/gpt-4o-mini` via OpenRouter. Swap to any model OpenRouter supports by changing two env vars:

```bash
LLM_MODEL=anthropic/claude-3-5-sonnet
LLM_BASE_URL=https://openrouter.ai/api/v1
```

Or point it at a local Ollama instance:

```bash
LLM_MODEL=qwen2.5:72b
LLM_BASE_URL=http://localhost:11434/v1
```

Optional tools for real scanner output and rendered diagrams:

```bash
brew install tfsec
pip install checkov
npm install -g @mermaid-js/mermaid-cli
```

If none of these are installed, the pipeline still completes. Security falls back to LLM review and diagrams save as Mermaid source.

---

![Built with: LangGraph, Python 3.12+, OpenRouter, FastMCP, Gradio, pydantic-settings, tfsec/checkov, Mermaid.js, uv](https://sectumpsempra.github.io/assets/images/infrasquad/infrasquad_stack_card.png)

---

Full source: [infrasquad - github](https://github.com/Andela-AI-Engineering-Bootcamp/infrasquad)

Built at [Andela AI Engineering Bootcamp](https://help.andela.com/hc/en-us/articles/48808339012115-Welcome-to-the-AI-Engineering-Bootcamp) by [Amit](https://linkedin.com/in/amit-bhatt), Ayesha, Elijah, Joel, Stella, and Adetayo.

If you are building anything with [LangGraph](https://github.com/langchain-ai/langgraph), multi-agent pipelines, or IaC automation, drop a comment. Especially curious whether anyone else hit the public ALB infinite loop case -- and how you handled it.
