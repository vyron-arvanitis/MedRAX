from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from medrax.llava.model import LlavaMistralForCausalLM
from medrax.llava.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)


def load_pretrained_model(
    model_path,
    model_base,
    model_name,
    load_in_8bit=False,
    load_in_4bit=True,
    device="cuda",
    cache_dir: str = "/model-weights",
    low_cpu_mem_usage=True,
    torch_dtype=torch.bfloat16,
):
    """Load tokenizer + model weights for either multimodal LLaVA or text-only causal LM usage.

    What this function does:
    - Builds load kwargs (device mapping + optional 8-bit / 4-bit quantization config).
    - Chooses a model-loading branch from `model_name` / `model_base`.
    - For LLaVA models, also initializes the vision tower path used to turn images into
      embeddings that can be fused with text tokens.
    - Computes the usable context length from model config.

    Loading branches:
    1) LLaVA branch (`"llava"` in `model_name`):
       - Loads `AutoTokenizer` from `model_path`.
       - Loads `LlavaMistralForCausalLM` from `model_path`.
       - Adds multimodal special tokens (image patch / optional image start-end).
       - Loads and moves vision tower + projector to the target device/dtype.
       - Exposes `vision_tower.image_processor` for image preprocessing.

    2) Text-only branch (`"llava"` not in `model_name`):
       - If `model_base` is provided: loads base model, applies PEFT/LoRA from `model_path`,
         then merges adapters into the base weights.
       - Else: loads a plain `AutoModelForCausalLM` from `model_path`
         (with MPT-specific `trust_remote_code=True`).

    Returns:
    - tokenizer (`PreTrainedTokenizer`):
      Tokenizer used to convert text <-> token ids for prompts and decoding.
    - model (`PreTrainedModel`):
      The full inference model used at generation time:
      1) it takes current prompt/context tokens,
      2) computes logits (scores) for the next token,
      3) repeats this step autoregressively to build the output sequence.
      In LLaVA mode, this same object also includes the image path:
      `vision_tower` encodes pixels -> image features, and `mm_projector` maps those
      features into the language-model embedding space before text generation.
    - image_processor (`ImageProcessor` or `None`):
      Image preprocessor from the vision tower for LLaVA models; `None` for text-only models.
    - context_len (`int`):
      Maximum supported input context length (from `model.config.max_sequence_length`
      when available, otherwise fallback `2048`).
    """

    kwargs = {}

    if device != "cuda":
        kwargs["device_map"] = {"": device}
    # else:
    #     kwargs["device_map"] = "auto"

    if load_in_8bit:
        kwargs["load_in_8bit"] = True
    elif load_in_4bit:
        # kwargs["load_in_4bit"] = True
        
        # Load the LLM with 4-bit stored weights to save VRAM; 
        # do the math in torch_dtype (e.g., bf16) for stability/speed; 
        # use NF4 for better 4-bit accuracy; double-quantize scales to save a bit more memory.
        kwargs["quantization_config"] = BitsAndBytesConfig(  
            load_in_4bit=True, # store weight sin 4bit
            bnb_4bit_compute_dtype=torch_dtype, # what type is to be used in the layers for computation
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    # else:
    # kwargs["torch_dtype"] = torch_dtype

    if "llava" in model_name.lower():
        # LLaVA multimodal model path
        if "mistral" in model_name.lower():
            # convert text to tokens, prevent re-downloading on every run with the cache_dir
            tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir) 
            model = LlavaMistralForCausalLM.from_pretrained( # intantiate  `LlavaMistralForCausalLM`
                model_path,
                low_cpu_mem_usage=low_cpu_mem_usage, # load model in memory-efficient way
                use_flash_attention_2=False, # disable FlashAttention v2 kernel
                cache_dir=cache_dir,
                torch_dtype=torch_dtype, # the type to load weights into
                **kwargs,
            )

    else:
        # Text-only language model path
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(
                model_base, use_fast=False, cache_dir=cache_dir
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_base,
                low_cpu_mem_usage=True,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                **kwargs,
            )
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print("Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch_dtype)
        else:
            # Plain AutoModelForCausalLM (special-case MPT)
            use_fast = False
            if "mpt" in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=True, cache_dir=cache_dir
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path, use_fast=False, cache_dir=cache_dir
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    cache_dir=cache_dir,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )

    image_processor = None

    if "llava" in model_name.lower():  # or 'mistral' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()

        vision_tower.to(device=device, dtype=torch_dtype)
        model.model.mm_projector.to(device=device, dtype=torch_dtype)

        if not (load_in_4bit or load_in_8bit):
            model.to(device=device, dtype=torch_dtype)

        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
