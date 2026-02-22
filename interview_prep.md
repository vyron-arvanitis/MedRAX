# MedRAX Interview Preparation (MCML HiWi, 1-hour conversational interview)

This document is tailored to a research-style interview where you may be asked anything about the **paper and codebase**.

---

## Section A — Paper-level understanding

### 1) What problem does MedRAX solve?
MedRAX addresses the limitation that many CXR AI systems are narrow, single-task systems (e.g., only classification, only report generation). The project proposes a unified agent that can orchestrate multiple tools for richer clinical reasoning.

### 2) What is the core contribution?
- A tool-augmented CXR reasoning agent architecture.
- Integration of heterogeneous tools (classification, segmentation, VQA, grounding, reporting, generation, utility).
- Evaluation through ChestAgentBench, a benchmark with 2,500 complex queries over 7 reasoning categories.

### 3) What is novel vs. engineering integration?
- Likely novelty: practical agentic orchestration for medical image reasoning + benchmarked multi-step behavior.
- Strong engineering: robust multi-tool integration with selective initialization, model management, and deployment pathways.

### 4) What assumptions should you mention in interview?
- Tool outputs are reliable enough to chain together.
- The selected benchmark approximates clinically meaningful reasoning.
- The backbone LLM can effectively interpret tool outputs and produce coherent decisions.

### 5) Main limitations you should proactively state
- Multi-tool pipelines can propagate upstream errors.
- Benchmark performance does not necessarily equal real-world clinical safety.
- Tool and LLM availability/hardware constraints may impact reproducibility.

---

## Section B — Method and architecture deep dive

### 1) End-to-end flow (how to explain on whiteboard)
1. User submits a medical query + image context.
2. Agent (LLM backbone) decides which tool(s) to call.
3. Selected tools run (e.g., segmentation/classification/VQA/grounding/report).
4. Intermediate outputs are fed back to the agent.
5. Agent synthesizes and returns answer.

### 2) Why agentic over single-model?
- Better specialization: each tool is optimized for a subtask.
- Better extensibility: new tools can be added without retraining one giant model.
- Better interpretability: intermediate tool outputs can be audited.

### 3) How tool modularity appears in code
`main.py` defines `all_tools` as a dictionary of lazy constructors and allows `tools_to_use` selective initialization.

### 4) Expected architecture interview follow-ups and crisp answers
- **Q:** How do you add a new tool?  
  **A:** Implement tool class in `medrax/tools`, add constructor entry in `all_tools`, include in `selected_tools` as needed.
- **Q:** How do you run lightweight experiments?  
  **A:** Start with a reduced `selected_tools` subset and use CPU or quantized models where available.

---

## Section C — Code-grounded details they can probe

### 1) Agent initialization details
- `initialize_agent(...)` sets prompt, tool registry, checkpointer, and `ChatOpenAI` model.
- Memory checkpointing is configured with `MemorySaver`.
- OpenAI connection parameters can be overridden through `openai_kwargs`.

### 2) Tool configuration patterns
- Some tools use model caches in `model_dir`.
- Grounding and LLaVA-Med expose quantization flags (`load_in_8bit`).
- Image generation requires manual RoentGen setup.

### 3) Benchmark script behavior (`quickstart.py`)
- Supports URL or local-image loading.
- Encodes images to base64 and sends multimodal messages.
- Retries requests (`tenacity`) and logs each case as JSON.
- Supports `--max-cases` for quick smoke runs.

### 4) Practical code-level talking points
- You can swap OpenAI-compatible endpoints via `OPENAI_BASE_URL`.
- Script includes graceful shutdown handling (SIGINT/SIGTERM) to preserve progress.
- Cases with missing images are skipped and logged as such.

---

## Section D — Evaluation and scientific rigor

### 1) What to say about benchmarking
ChestAgentBench tests 7 reasoning categories (detection/classification/localization/comparison/relationship/diagnosis/characterization), which is better than single-task reporting but still may not capture all real-clinic edge cases.

### 2) Ablations to propose
1. Remove one tool at a time and measure category-specific drops.
2. Compare tool subsets (minimal vs full stack).
3. Quantization impact on accuracy/latency.
4. Prompt-template sensitivity.
5. Error taxonomy: perception error vs reasoning error vs retrieval/tool-selection error.

### 3) Strong researcher-style critique lines
- Distinguish aggregate score gains from clinically meaningful gains.
- Inspect whether improvements are uniform across categories or concentrated in a few.
- Report confidence calibration, not just final correctness.

---

## Section E — Whiteboard coding/design prompts

### Prompt 1: Tool router
Design a router that maps query intents to candidate tools, then uses confidence-weighted fusion for final answer synthesis.

### Prompt 2: Conflict resolution across tools
If classifier and VQA disagree, use:
1. tool reliability priors,
2. confidence estimates,
3. optional adjudication prompt that includes both outputs explicitly.

### Prompt 3: Safety-first output layer
Add a final safety checker that can:
- block overconfident claims,
- request missing context,
- present uncertainty bands.

### Prompt 4: Fast prototyping idea
Add a lightweight execution planner that minimizes expensive tool calls if a high-confidence answer is already achieved.

---

## Section F — Research-fit and collaboration questions

### 1) Questions you may get
- Why do you want to join this group?
- What is your first 3-month plan?
- Which extension could become publishable?
- How do you work with uncertain results and negative findings?

### 2) Suggested 3-month plan
- Month 1: reproduce baseline + logging + failure taxonomy.
- Month 2: run tool-ablation and routing studies.
- Month 3: implement one reliability extension (e.g., confidence-aware fusion) and evaluate.

### 3) Questions to ask them
- Mentorship cadence and expectation for HiWi ownership.
- Whether success is publication-oriented or engineering-impact oriented.
- Preferred experimentation and review standards in the group.

---

## 20 likely interview questions — with compact answers

1. **What gap does the work close?**  
   It connects isolated CXR capabilities into one orchestrated agent for multi-step reasoning.

2. **Why not one monolithic model?**  
   Specialized tools are easier to improve independently and can outperform one-size-fits-all systems on niche subtasks.

3. **What is the main technical risk?**  
   Error propagation across tools and brittle routing decisions.

4. **How is modularity implemented?**  
   A tool registry and selective initialization pattern in `initialize_agent`.

5. **How would you add a new finding-localization tool?**  
   Implement interface-compatible tool class, register it, add to selected tool list, and benchmark category-wise impact.

6. **How do you ensure reproducibility?**  
   Fixed config, logged prompts/inputs/outputs, deterministic evaluation settings where possible, and clear model/version tracking.

7. **What ablation is mandatory?**  
   Leave-one-tool-out across all benchmark categories.

8. **How do you evaluate reasoning vs perception errors?**  
   Manual or semi-automated error coding on sampled failures, separating visual misread from inference mistake.

9. **What if benchmark scores improve but clinicians disagree?**  
   Prioritize external/clinician-grounded evaluation and investigate mismatch causes before claiming practical gains.

10. **How can latency be reduced?**  
    Early-exit routing, cheaper first-pass tools, quantization, and caching.

11. **What role does prompt engineering play?**  
    It strongly affects tool-use behavior and final synthesis quality; should be systematically stress-tested.

12. **How do you handle missing images/corrupt inputs?**  
    Skip with explicit logging and avoid silent failures.

13. **How can local LLM endpoints be used?**  
    Set `OPENAI_BASE_URL` and compatible API key.

14. **What is one likely confound in reported gains?**  
    Gains may stem from benchmark/tool overlap or category imbalance, not generalized clinical reasoning.

15. **How to handle disagreement between tools?**  
    Confidence-aware fusion + explicit adjudication prompt.

16. **What extension would you implement first?**  
    Confidence-calibrated routing/fusion with a reject option.

17. **How do you reduce hallucination risk?**  
    Force evidence citation from tool outputs and abstain when evidence is insufficient.

18. **How to make this publishable as a HiWi project?**  
    Add a novel reliability mechanism and show statistically robust gains on reasoning and safety metrics.

19. **What would falsify the core claim?**  
    If a simple single-model baseline matches or beats full MedRAX across categories under fair settings.

20. **What is your strongest criticism of the approach?**  
    Benchmark success alone cannot establish clinical readiness; calibration and prospective validation remain crucial.

---

## Review (OpenReview comments summary + uncovered issues)

### Source requested
Paper thread: `https://openreview.net/forum?id=JiFfij5iv0`

### Access status
- 🔴 **Could not fetch OpenReview reviewer comments from this environment due network/proxy access restrictions (HTTPS proxy tunnel returned 403).**
- 🔴 **Therefore, I cannot truthfully summarize reviewer-specific comments or answer reviewer-specific questions from that thread here.**

### What to do before interview
1. Open the OpenReview page manually.
2. Copy reviewer comments into this section.
3. For each reviewer point, prepare:
   - your agreement/disagreement,
   - one concrete experiment/analysis to address it,
   - expected impact on claims.

### Fill-in template (use once you have comments)
- **Reviewer concern:** ...  
  **Your answer:** ...  
  **Extra experiment:** ...  
  **If unresolved:** 🔴 bring as open discussion point in interview.

