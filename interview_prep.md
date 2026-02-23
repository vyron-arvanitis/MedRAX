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

#### First clarify current MedRAX behavior vs. proposed extension

You are correct: in the current MedRAX setup, the LLM agent (ChatGPT-style model) reads the tool descriptions and decides which tool(s) to call; tool outputs are then fed back into the LLM, and the LLM synthesizes the final answer. There is not a strict explicit weighted-sum fusion module hard-coded in the main flow.

So in an interview, frame it like this:
- **Current system:** LLM-driven tool orchestration and synthesis.
- **Proposed improvement (if interviewer asks design ideas):** add an explicit confidence-aware router/fusion layer before final synthesis.

#### How to implement this extension (interview-ready answer)

Use a 4-stage pattern:

1. **Intent detection** (what task is being asked?)
   - Parse the user query into one or more intents: `{classification, localization, report, comparison, generation, dicom}`.
   - Fast option: rule + keyword mapping.
   - Better option: small LLM/router prompt that outputs structured JSON.

2. **Candidate tool selection**
   - Map each intent to tools with a static lookup table.
   - Example:
     - `classification -> [ChestXRayClassifierTool, XRayVQATool]`
     - `localization -> [XRayPhraseGroundingTool, ChestXRaySegmentationTool]`
     - `report -> [ChestXRayReportGeneratorTool, XRayVQATool]`

3. **Tool execution + normalized confidence**
   - Execute top-k tools.
   - Convert each tool output to a common schema:
     - `answer`
     - `confidence` in `[0,1]`
     - `evidence` (e.g., logits/probabilities/bounding boxes/text span)
   - If tool has no native confidence, estimate proxy confidence (e.g., softmax margin, answer consistency across prompt variants).

4. **Confidence-weighted fusion + abstention**
   - Score each tool output with:

   ```text
   fused_score_i = w_tool_i * confidence_i * evidence_quality_i
   ```

   - `w_tool_i` is a reliability prior learned from validation data (per task category).
   - Pick answer with highest aggregated score.
   - If top score < threshold, abstain and ask for more context or run one extra adjudication step.

#### Minimal pseudocode (whiteboard)

```python
def route_and_fuse(query, image):
    intents = detect_intents(query)  # e.g. ["classification", "localization"]
    tools = select_tools(intents)    # e.g. [Classifier, Grounder, VQA]

    outputs = []
    for tool in tools:
        raw = tool.run(query=query, image=image)
        norm = normalize_output(raw, tool_name=tool.name)  # answer/conf/evidence
        norm["weight"] = reliability_prior(tool.name, intents)
        norm["fused_score"] = norm["weight"] * norm["confidence"] * norm["evidence_quality"]
        outputs.append(norm)

    best = aggregate(outputs)
    if best["fused_score"] < ABSTAIN_THRESHOLD:
        return {"decision": "abstain", "reason": "low confidence", "next_step": "request more context"}

    return {"decision": best["answer"], "evidence": best["evidence"], "used_tools": [o["tool"] for o in outputs]}
```

#### What is a reliability prior? (plain explanation)
A **reliability prior** is a pre-estimated trust score for each tool, usually conditioned on task type.

- Think of it as: “Historically, how often is this tool correct on this kind of question?”
- Example: if `ChestXRayClassifierTool` is very strong on classification but weaker on localization, then its prior is high for classification and lower for localization.
- These priors are computed from validation data (accuracy + calibration), then used to scale each tool’s influence in fusion.
- In a pure LLM-driven setup (current MedRAX), you can still use reliability priors by exposing them in the system prompt or by post-processing tool outputs before the final LLM synthesis.

**Important implementation note:** in this repository, tools do **not** expose a built-in field like `ChestXRayClassifierTool.reliability_prior`.
- `reliability_prior` is a design-time/runtime value you compute externally from validation experiments.
- In practice, store priors in a config/dictionary (for example: `priors[task][tool_name]`) and apply them in your router/fusion logic.
- Optionally, you can extend tool wrappers to attach metadata, but that is an added feature, not current default behavior.

Example storage pattern:

```python
priors = {
    "classification": {
        "ChestXRayClassifierTool": 0.88,
        "XRayVQATool": 0.72,
    },
    "localization": {
        "XRayPhraseGroundingTool": 0.81,
        "ChestXRaySegmentationTool": 0.76,
    },
}
```

At runtime you read `priors[task][tool_name]` and multiply it with confidence/evidence scores in your fusion logic.

#### What to say if interviewer asks "how do you train weights?"
First clarify there are **two different kinds of weights**:

1. **Model weights** (inside each tool’s backbone model)
   - Example: `ChestXRayClassifierTool` uses a pretrained DenseNet checkpoint.
   - These are learned during model training by the original model authors.

2. **Router/fusion weights** (reliability priors, e.g., `w_tool_i`)
   - These are **not** neural network parameters inside the tool.
   - These are scalar trust coefficients you estimate from your own validation runs.

How to estimate reliability priors in practice:
- Build a validation split grouped by task category (classification/localization/report/etc.).
- Run each candidate tool on that split.
- Compute a reliability metric per tool per task (e.g., calibrated accuracy, ECE-adjusted score).
- Map metrics to `[0,1]` and store as priors: `priors[task][tool_name] = score`.
- Use these priors at inference time to scale each tool’s contribution.
- Re-estimate periodically when models, prompts, or data distribution change.

### Prompt 2: Conflict resolution across tools
If classifier and VQA disagree, use:
1. tool reliability priors,
2. confidence estimates,
3. optional adjudication prompt that includes both outputs explicitly.

#### How disagreement works in current LLM-agent flow
- Yes, the model can detect that tool outputs conflict because it sees tool outputs as structured/plain text content in the conversation state.
- Practically, this is handled at the text/structured-message level after tool execution (the tool result content is passed back to the model), even though internally the model reasons in embedding space.
- So for interview language: **inputs are explicit tool messages; decision is made by the LLM over those messages**.

#### Practical decision policy you can propose
When two tools disagree (example: 0.9 vs 0.7 confidence), use a deterministic policy before final synthesis:

1. **Agreement check**
   - Are predictions semantically equivalent? (e.g., "cardiomegaly" vs "enlarged cardiac silhouette")
2. **Confidence margin check**
   - If top confidence exceeds second by margin `m` (e.g., `m >= 0.15`) and tool prior is reasonable, choose top.
3. **Reliability-aware override**
   - If lower-confidence tool has much higher reliability prior for this task, allow override.
4. **Adjudication step**
   - If conflict remains near-tied, run a final adjudication prompt that explicitly compares evidence from both tools.
5. **Abstain/fallback**
   - If still uncertain, abstain or request more context.

#### SOTA-style patterns for multi-tool conflict handling
These are common modern patterns you can cite in interviews:

- **Mixture-of-Experts style gating**: a learned router predicts which expert/tool to trust per query.
- **Calibrated confidence fusion**: combine outputs only after calibration (temperature scaling / isotonic methods).
- **Verifier-critic pipelines**: one model/tool proposes, another verifies consistency with evidence.
- **Self-consistency / debate**: multiple reasoning traces or tool calls vote, then aggregate.
- **Uncertainty-aware abstention**: reject option when confidence is low or disagreement is high.
- **RAG-style evidence grounding**: final answer must be justified by explicit tool evidence snippets.

In this repo, you can position these as **extensions on top of the current LLM-driven orchestration**.

#### Code draft (interview sketch)

```python
def resolve_conflict(task, outputs, priors, margin=0.15, abstain_th=0.45):
    """
    outputs: list of dicts, each = {
      "tool": str,
      "prediction": str,
      "confidence": float,      # normalized to [0,1]
      "evidence": str,
    }
    priors: dict like priors[task][tool_name] -> float in [0,1]
    """

    # 1) normalize into reliability-aware scores
    scored = []
    for o in outputs:
        prior = priors.get(task, {}).get(o["tool"], 0.5)
        score = prior * o["confidence"]
        scored.append({**o, "prior": prior, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top = scored[0]
    second = scored[1] if len(scored) > 1 else None

    # 2) if effectively same answer, merge evidence and return
    if second and semantic_match(top["prediction"], second["prediction"]):
        return {
            "decision": top["prediction"],
            "mode": "agreement",
            "evidence": [top["evidence"], second["evidence"]],
            "score": top["score"],
        }

    # 3) margin-based decision
    if second is None or (top["score"] - second["score"] >= margin):
        if top["score"] >= abstain_th:
            return {"decision": top["prediction"], "mode": "direct", "evidence": [top["evidence"]], "score": top["score"]}

    # 4) adjudication when close conflict
    adjudicated = llm_adjudicate(
        task=task,
        candidates=[
            {"tool": top["tool"], "prediction": top["prediction"], "confidence": top["confidence"], "prior": top["prior"], "evidence": top["evidence"]},
            {"tool": second["tool"], "prediction": second["prediction"], "confidence": second["confidence"], "prior": second["prior"], "evidence": second["evidence"]},
        ],
        instruction=(
            "Pick the best-supported answer. If evidence is insufficient or contradictory, return ABSTAIN."
        ),
    )

    if adjudicated.get("decision") == "ABSTAIN":
        return {"decision": "ABSTAIN", "mode": "uncertain", "score": max(top["score"], second["score"]) if second else top["score"]}

    return {"decision": adjudicated["decision"], "mode": "adjudicated", "score": adjudicated.get("score", top["score"])}
```

#### Adjudication prompt template (quick)
"""
Task: {task}
Candidate A: tool={tool_a}, pred={pred_a}, conf={conf_a}, prior={prior_a}, evidence={ev_a}
Candidate B: tool={tool_b}, pred={pred_b}, conf={conf_b}, prior={prior_b}, evidence={ev_b}
Instruction: Choose the best-supported prediction. If evidence is insufficient/contradictory, output ABSTAIN.
Return JSON: {"decision": "...", "rationale": "...", "score": 0-1}
"""

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
