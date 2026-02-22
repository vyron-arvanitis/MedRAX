# Four Prompt Full Workflows (All Selected Tools)

This file documents the exact runtime path for this `selected_tools` set in `main.py`:

```python
selected_tools = [
    "ImageVisualizerTool",
    "DicomProcessorTool",
    "ChestXRayClassifierTool",
    "ChestXRaySegmentationTool",
    "ChestXRayReportGeneratorTool",
    "XRayVQATool",
    "LlavaMedTool",
    "XRayPhraseGroundingTool",
    "ChestXRayGeneratorTool",
]
```

It focuses on:
- exact `messages` payload creation in `interface.py`
- LangGraph node flow (`process` / `execute`)
- tool call dispatch (`args_schema` validation -> `_run`)
- where images become tensors/tokens
- four concrete prompts that cover all tools.

## 1) End-to-End Runtime Path (Exact)

### A. UI submit chain (`interface.py`)
Upload-time behavior:
- If uploaded suffix is `.dcm`, `ChatInterface.handle_upload(...)` immediately calls
  `DicomProcessorTool._run(...)` once to create a displayable PNG for the right panel.
- It still keeps `original_file_path` as the original `.dcm` so the agent/tool chain can
  reason over the source file path.

1. `txt.submit(interface.add_message, ...)` updates UI chat history.
2. `.then(interface.process_message, ...)` starts agent streaming.
3. `process_message(...)` builds `messages`:
   - initializes `messages = []`
   - if image exists: appends text hint `"image_path: ..."` and a multimodal `image_url` item
   - if user text exists: appends a `{"type": "text", "text": ...}` content item.
4. It calls:
   ```python
   self.agent.workflow.stream(
       {"messages": messages},
       {"configurable": {"thread_id": self.current_thread_id}},
   )
   ```

### B. Graph entry and loop (`medrax/agent/agent.py`)
Graph topology:
- entry node: `process`
- conditional:
  - if tool calls exist -> `execute`
  - else -> `END`
- edge: `execute -> process` (loop)

Execution sequence per turn:
1. `process_request(state)`:
   - optional `SystemMessage` prepended
   - `self.model.invoke(messages)` returns an `AIMessage`.
2. `has_tool_calls(state)`:
   - checks `state["messages"][-1].tool_calls`.
3. If tool calls exist: `execute_tools(state)`:
   - iterates each `call`
   - resolves tool by `call["name"]` in `self.tools`
   - runs `self.tools[call["name"]].invoke(call["args"])`
   - wraps result into `ToolMessage(content=str(result), ...)`
4. Tool messages are appended into graph state.
5. Loop back to `process_request(...)` so model can use tool outputs.

### C. Tool dispatch internals (LangChain)
For each tool call:
1. `BaseTool.invoke(input=args_dict)` -> `run(...)`
2. `_parse_input(...)` validates against the tool `args_schema` (Pydantic model).
3. `_to_args_and_kwargs(...)` converts validated input to kwargs.
4. tool-specific `_run(...)` is called.

## 2) Tool Mapping: Class Key -> Actual Tool Name -> Args -> Function

The model calls the `BaseTool.name` values (not class names).

| `selected_tools` key in `main.py` | `BaseTool.name` used in tool call | `args_schema` fields | Function that executes |
|---|---|---|---|
| `ImageVisualizerTool` | `image_visualizer` | `image_path: str`, `title: Optional[str]`, `description: Optional[str]`, `figsize: Optional[tuple]=(10,10)`, `cmap: Optional[str]="rgb"` | `ImageVisualizerTool._run(...)` |
| `DicomProcessorTool` | `dicom_processor` | `dicom_path: str`, `window_center: Optional[float]`, `window_width: Optional[float]` | `DicomProcessorTool._run(...)` |
| `ChestXRayClassifierTool` | `chest_xray_classifier` | `image_path: str` | `ChestXRayClassifierTool._run(...)` |
| `ChestXRaySegmentationTool` | `chest_xray_segmentation` | `image_path: str`, `organs: Optional[List[str]]` | `ChestXRaySegmentationTool._run(...)` |
| `ChestXRayReportGeneratorTool` | `chest_xray_report_generator` | `image_path: str` | `ChestXRayReportGeneratorTool._run(...)` |
| `XRayVQATool` | `chest_xray_expert` | `image_paths: List[str]`, `prompt: str`, `max_new_tokens: int=512` | `XRayVQATool._run(...)` |
| `LlavaMedTool` | `llava_med_qa` | `question: str`, `image_path: Optional[str]` | `LlavaMedTool._run(...)` |
| `XRayPhraseGroundingTool` | `xray_phrase_grounding` | `image_path: str`, `phrase: str`, `max_new_tokens: int=300` | `XRayPhraseGroundingTool._run(...)` |
| `ChestXRayGeneratorTool` | `chest_xray_generator` | `prompt: str`, `height: int=512`, `width: int=512`, `num_inference_steps: int=75`, `guidance_scale: float=4.0` | `ChestXRayGeneratorTool._run(...)` |

## 3) Exact `messages` Payload Shapes from `interface.py`

### Case A: image uploaded + user text
```python
messages = [
    {"role": "user", "content": "image_path: temp/upload_123.png"},
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,<...>"},
            }
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "<your prompt text>"}],
    },
]
```

### Case B: text only (no upload)
```python
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "<your prompt text>"}],
    }
]
```

## 4) Four Prompts That Cover All Tools

Note: To make tool routing reliable, each prompt explicitly names tool names and expected order.

---

### Prompt 1: DICOM Full Analysis Pipeline

Use this after uploading a `.dcm`:

Natural-language user prompt example:

```text
In this image, what type of disease do we have, and where is it located? Use all available tools for diagnosis and localization.
```

Tool-routing version (recommended for deterministic tool order):

```text
Run tools in this order:
1) dicom_processor on the uploaded DICOM path
2) chest_xray_classifier on the converted PNG image_path
3) chest_xray_segmentation on the converted PNG for organs ["Left Lung","Right Lung","Heart"]
4) chest_xray_report_generator on the converted PNG
5) image_visualizer on segmentation_image_path with title "Segmentation Overlay"
Then summarize all outputs.
```

#### Expected graph/tool sequence
1. `process` -> model emits tool call for `dicom_processor`.
2. `execute`:
   ```python
   {"name": "dicom_processor", "args": {"dicom_path": "temp/upload_<ts>.dcm"}}
   ```
   -> validates `DicomProcessorInput`
   -> calls `DicomProcessorTool._run(...)`
   -> returns `({"image_path": "temp/processed_dicom_xxxx.png"}, metadata)`
3. `process` -> model reads `ToolMessage` and emits `chest_xray_classifier`.
4. `execute`:
   ```python
   {"name": "chest_xray_classifier", "args": {"image_path": "temp/processed_dicom_xxxx.png"}}
   ```
   -> `ChestXRayClassifierTool._run(...)`.
5. `process` -> emits `chest_xray_segmentation`.
6. `execute`:
   ```python
   {
     "name": "chest_xray_segmentation",
     "args": {
       "image_path": "temp/processed_dicom_xxxx.png",
       "organs": ["Left Lung", "Right Lung", "Heart"]
     }
   }
   ```
   -> `ChestXRaySegmentationTool._run(...)`
   -> output includes `segmentation_image_path`.
7. `process` -> emits `chest_xray_report_generator`.
8. `execute`:
   ```python
   {"name": "chest_xray_report_generator", "args": {"image_path": "temp/processed_dicom_xxxx.png"}}
   ```
   -> `ChestXRayReportGeneratorTool._run(...)`.
9. `process` -> emits `image_visualizer`.
10. `execute`:
   ```python
   {
     "name": "image_visualizer",
     "args": {
       "image_path": "temp/segmentation_xxxx.png",
       "title": "Segmentation Overlay"
     }
   }
   ```
   -> `ImageVisualizerTool._run(...)`.
11. `process` -> final natural-language summary (no tool calls) -> END.

#### Image/tensor path in this workflow
- `dicom_processor`: DICOM pixel array -> optional windowing -> PNG file.
- `chest_xray_classifier`: `skimage.imread` -> normalize -> crop -> tensor -> DenseNet scores.
- `chest_xray_segmentation`: image -> normalize/resize -> PSPNet -> sigmoid masks -> metrics.
- `chest_xray_report_generator`: PIL image -> `ViTImageProcessor` `pixel_values` -> ViT-BERT generate -> decode text.

---

### Prompt 2: Dual VQA Cross-Check (CheXagent + LLaVA-Med)

Use this after uploading a PNG/JPG:

```text
Run chest_xray_expert first with:
image_paths=[<uploaded image path>]
prompt="List the top abnormalities, then explain supporting visual signs."
Then run llava_med_qa on the same image with:
question="Do you agree with that interpretation? Point out any disagreement."
Finally compare both answers.
```

#### Expected graph/tool sequence
1. `process` -> emits `chest_xray_expert`.
2. `execute`:
   ```python
   {
     "name": "chest_xray_expert",
     "args": {
       "image_paths": ["temp/upload_<ts>.png"],
       "prompt": "List the top abnormalities, then explain supporting visual signs."
     }
   }
   ```
   -> validates `XRayVQAToolInput`
   -> calls `XRayVQATool._run(...)`.
3. `process` -> emits `llava_med_qa`.
4. `execute`:
   ```python
   {
     "name": "llava_med_qa",
     "args": {
       "question": "Do you agree with that interpretation? Point out any disagreement.",
       "image_path": "temp/upload_<ts>.png"
     }
   }
   ```
   -> validates `LlavaMedInput`
   -> calls `LlavaMedTool._run(...)`.
5. `process` -> final comparison text -> END.

#### Important: no majority-vote logic here
- The current agent does **not** implement majority voting between CheXagent and LLaVA-Med.
- What happens instead:
  1) model calls `chest_xray_expert`
  2) model calls `llava_med_qa`
  3) model reads both tool outputs and writes one final response in a later `process` step.
- So the final answer is synthesis by the orchestrator LLM, not a hard vote rule.

#### Image/token path in this workflow
- `chest_xray_expert`:
  - `tokenizer.from_list_format([{"image": ...}, {"text": ...}])`
  - `apply_chat_template(...)` -> `input_ids`
  - `model.generate(...)` -> decode response.
- `llava_med_qa`:
  - text prompt embeds image token marker
  - `tokenizer_image_token(...)` -> text `input_ids`
  - `process_images(...)` -> vision tensor
  - `model.generate(input_ids, images=image_tensor, ...)` -> decode.

---

### Prompt 3: Phrase Grounding + Visualization

Use this after uploading a PNG/JPG:

```text
Run xray_phrase_grounding with:
phrase="left pleural effusion"
max_new_tokens=300
Then run image_visualizer on the returned visualization_path with title "Grounded Finding".
Explain the box coordinates in plain language.
```

#### Expected graph/tool sequence
1. `process` -> emits `xray_phrase_grounding`.
2. `execute`:
   ```python
   {
     "name": "xray_phrase_grounding",
     "args": {
       "image_path": "temp/upload_<ts>.png",
       "phrase": "left pleural effusion",
       "max_new_tokens": 300
     }
   }
   ```
   -> validates `XRayPhraseGroundingInput`
   -> calls `XRayPhraseGroundingTool._run(...)`.
3. `process` -> emits `image_visualizer` on `visualization_path`.
4. `execute` -> `ImageVisualizerTool._run(...)`.
5. `process` -> explanation -> END.

#### Image/token path in this workflow
- `xray_phrase_grounding`:
  - `processor.format_and_preprocess_phrase_grounding_input(...)`
    produces `input_ids` and `pixel_values`
  - generation result decoded by `processor.decode(...)`
  - structured predictions parsed by
    `convert_output_to_plaintext_or_grounded_sequence(...)`
  - boxes converted to image coordinates and overlay PNG saved.

---

### Prompt 4: Generate Synthetic X-ray Then Analyze

Use without uploading any image:

```text
Run chest_xray_generator with:
prompt="PA chest X-ray showing moderate cardiomegaly and mild bilateral pleural effusions"
height=512
width=512
num_inference_steps=75
guidance_scale=4.0
Then run image_visualizer on the generated image.
Then run chest_xray_classifier, chest_xray_segmentation, and chest_xray_report_generator on that same generated image.
Finish with a concise synthesis.
```

#### Expected graph/tool sequence
1. `process` -> emits `chest_xray_generator`.
2. `execute`:
   ```python
   {
     "name": "chest_xray_generator",
     "args": {
       "prompt": "PA chest X-ray showing moderate cardiomegaly and mild bilateral pleural effusions",
       "height": 512,
       "width": 512,
       "num_inference_steps": 75,
       "guidance_scale": 4.0
     }
   }
   ```
   -> validates `ChestXRayGeneratorInput`
   -> calls `ChestXRayGeneratorTool._run(...)`
   -> output `{"image_path": "temp/generated_xray_xxxx.png"}`.
3. `process` -> emits `image_visualizer`.
4. `execute` -> `ImageVisualizerTool._run(...)`.
5. `process` -> emits classifier/segmentation/report tool calls against generated image path.
6. `execute` -> runs each tool `_run(...)`.
7. `process` -> final synthesis -> END.

#### Image/token path in this workflow
- `chest_xray_generator`: Stable Diffusion pipeline encodes prompt and denoises latent image -> PNG output.
- downstream tools then consume that PNG as described in previous prompts.

## 5) Tokenization and Image Encoding (Model-Accurate)

### 5.0 Your JSON tool-call example: what is and is not tokenized

Example:
```json
{
  "name": "chest_xray_generator",
  "args": {
    "prompt": "PA chest X-ray showing moderate cardiomegaly and mild bilateral pleural effusions",
    "height": 512,
    "width": 512,
    "num_inference_steps": 75,
    "guidance_scale": 4.0
  }
}
```

Important split:
1. **This JSON object itself is not tokenized by your Python tool code.**
   - It is already a structured tool-call dict when `Agent.execute_tools` receives it.
   - LangChain validates `args` against `ChestXRayGeneratorInput` and calls `_run(...)`.
2. **Inside `ChestXRayGeneratorTool._run`, the `prompt` string *is* tokenized by Diffusers internals.**
   - `StableDiffusionPipeline.__call__` calls `encode_prompt(...)`.
   - In `encode_prompt(...)`, `self.tokenizer(prompt, ...)` tokenizes text.
   - Then `self.text_encoder(...)` converts token IDs to text embeddings.
   - Then diffusion denoising uses those embeddings to generate image latents -> final PNG.

### 5.1 Orchestrator model (`ChatOpenAI` in `Agent.process_request`)
- Input source: `interface.py` builds `messages` with:
  - plain text like `image_path: temp/...`
  - multimodal content item with `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`
  - text item `{"type": "text", "text": "..."}`
- These are sent to the OpenAI API via `self.model.invoke(messages)`.
- Text tokenization and image feature extraction happen server-side in the OpenAI model stack (not in this repo code).

### 5.2 CheXagent (`chest_xray_expert`)
Code path: `medrax/tools/xray_vqa.py::_generate_response`.

1. Build multimodal query:
   ```python
   query = tokenizer.from_list_format([{"image": path1}, ..., {"text": prompt}])
   ```
2. Wrap chat turns:
   ```python
   conv = [{"from": "system", ...}, {"from": "human", "value": query}]
   ```
3. Tokenize chat into IDs:
   ```python
   input_ids = tokenizer.apply_chat_template(..., return_tensors="pt")
   ```
   - tensor shape is `[1, T_prompt]`.
4. Generate continuation:
   ```python
   output = model.generate(input_ids, max_new_tokens=...)
   ```
   - output shape is `[1, T_prompt + T_new]`.
5. Decode only generated part:
   ```python
   tokenizer.decode(output[0][input_ids.size(1): -1])
   ```

Note:
- `from_list_format(...)` / `apply_chat_template(...)` use `trust_remote_code=True`; exact image-token internals are model-specific to CheXagent.
- Operationally, this code path is where text + image references become model-ready token IDs.

### 5.3 LLaVA-Med (`llava_med_qa`)
Code path: `medrax/tools/llava_med.py::_process_input` and `medrax/llava/mm_utils.py`.

Text side:
1. Prefix question with image markers:
   - either `<im_start><image><im_end>\n...`
   - or `<image>\n...`
2. Build Vicuna-style conversation prompt string.
3. Call:
   ```python
   tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX=-200, return_tensors="pt")
   ```
   - splits on literal `<image>`
   - tokenizes text chunks
   - inserts sentinel ID `-200` between chunks.

Image side:
1. `Image.open(image_path)` -> PIL image.
2. `process_images([image], image_processor, model.config)`:
   - optional pad-to-square
   - `image_processor.preprocess(...)["pixel_values"]`
   - returns vision tensor.

Fusion inside model:
- In `medrax/llava/model/llava_arch.py::prepare_inputs_labels_for_multimodal`, each `IMAGE_TOKEN_INDEX` placeholder in text stream is replaced with projected vision features (`mm_projector` output), expanding sequence length from text-only `T` to multimodal `T_mm`.

### 5.4 Phrase grounding (`xray_phrase_grounding`)
Code path: `medrax/tools/grounding.py::_run`.

1. `processor.format_and_preprocess_phrase_grounding_input(...)` returns at least:
   - `input_ids` (tokenized prompt)
   - `pixel_values` (image tensor for vision encoder)
2. Generation emits text sequence containing grounded output format.
3. `processor.decode(...)` decodes generated IDs.
4. `processor.convert_output_to_plaintext_or_grounded_sequence(...)` parses decoded text into structured phrase + bounding boxes.

### 5.5 Report generation (`chest_xray_report_generator`)
Code path: `medrax/tools/report_generation.py`.

1. `ViTImageProcessor(..., return_tensors="pt").pixel_values` creates image tensor for ViT encoder.
2. If needed, image tensor is resized to model encoder expected size.
3. Decoder generates token IDs with `model.generate(...)`.
4. `BertTokenizer.batch_decode(...)` converts IDs back to findings/impression strings.

### 5.6 Classification and segmentation
- `chest_xray_classifier` and `chest_xray_segmentation` are not text-token models.
- They perform image preprocessing (`normalize`, crop/resize) and operate directly on tensors.

### 5.7 Synthetic generator (`chest_xray_generator`)
- Uses Diffusers `StableDiffusionPipeline`.
- Prompt text tokenization and latent diffusion steps are handled inside pipeline internals.
- Result is saved image file, then consumed by other tools.

Detail:
- In `diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py`:
  - `__call__` invokes `encode_prompt(...)`
  - `encode_prompt(...)` does:
    - `self.tokenizer(prompt, ...)` -> token IDs
    - `self.text_encoder(text_input_ids, ...)` -> prompt embeddings
- So for generator, text tokenization is real, but hidden inside Diffusers, not in `generation.py`.

### 5.8 Complete per-tool tokenization matrix

| Tool (`BaseTool.name`) | Text tokenization path | Image encoding path |
|---|---|---|
| `image_visualizer` | None | `skimage.io.imread(...)` for display/reference only (no tokenization) |
| `dicom_processor` | None | `pydicom.dcmread(...).pixel_array` + windowing/rescale to PNG |
| `chest_xray_classifier` | None | `skimage.imread` -> normalize -> crop -> tensor for DenseNet |
| `chest_xray_segmentation` | None | `skimage.imread` -> normalize -> resize(512) -> tensor for PSPNet |
| `chest_xray_report_generator` | No user text prompt tokenization in this tool input | `ViTImageProcessor(...).pixel_values` for encoder; decoder token IDs are generated then `BertTokenizer.batch_decode(...)` |
| `chest_xray_expert` (CheXagent) | `tokenizer.from_list_format(...)` + `apply_chat_template(...)` -> `input_ids` | Handled via model-specific remote tokenizer/processor path (through `from_list_format` representation) |
| `llava_med_qa` | `tokenizer_image_token(...)` inserts `IMAGE_TOKEN_INDEX` into `input_ids` | `process_images(...)` -> vision tensor; model replaces image-token placeholders with projected image features |
| `xray_phrase_grounding` | `AutoProcessor.format_and_preprocess_phrase_grounding_input(...)` builds `input_ids` | Same processor call builds `pixel_values` for image encoder |
| `chest_xray_generator` | Diffusers `encode_prompt`: `self.tokenizer(prompt, ...)` -> IDs, then `self.text_encoder(...)` -> embeddings | No input image; output image is synthesized from latent diffusion |

Takeaway:
- Some tools are pure tensor/image models (no text tokenization).
- Some are multimodal LLMs that build text token IDs + image tensors.
- The orchestrator tool-call dict is structured metadata; actual tokenization happens either in the orchestrator model service or inside each model tool implementation.

## 6) How Tool Outputs Re-enter the UI

Inside `interface.py`, for `event["execute"]["messages"]`:
1. each item is a `ToolMessage`
2. content is stringified tuple from `Agent.execute_tools`:
   - effectively `str((output, metadata))`
3. UI does:
   ```python
   tool_result = eval(message.content)[0]
   ```
   so it takes only tuple element `[0]` (the `output` object).
4. if `message.name == "image_visualizer"`, UI updates `display_file_path` from `tool_result["image_path"]`.

## 7) Important Current Caveats (Code-Accurate)

1. `xray_phrase_grounding` currently contains:
   ```python
   output = self.model.generate(...)
   ```
   in `medrax/tools/grounding.py`, which is a literal ellipsis placeholder.
   As written, this is likely to fail at runtime and return an error payload.

2. `llava_med_qa` has `image_path` optional in schema, but `_run` unconditionally does:
   `image_tensor = image_tensor.to(...)`.
   If no image is provided, this can raise `NoneType` errors.

3. For DICOM uploads, `process_message` currently base64-encodes `original_file_path`.
   If that path is `.dcm`, it is still wrapped as `data:image/jpeg;base64,...`.
   Tool-based path flow still works, but model-side multimodal decoding may be unreliable for raw DICOM bytes.

4. `eval(message.content)` is powerful but unsafe for untrusted content.
   Safer parsing would use structured serialization (`json.dumps`/`json.loads`) or `ast.literal_eval`.

## 8) Quick Debug Checklist While Running These Prompts

1. Confirm tool names in model tool calls match `BaseTool.name` values above.
2. Confirm each tool call `args` keys match that tool's `args_schema`.
3. Check `logs/tool_calls_<timestamp>.json` for exact execution history.
4. If UI shows tool errors, inspect `message.content` before `eval(...)`.
5. If graph appears to skip context, verify stable `thread_id` is retained (or click "New Thread" intentionally).
