from PIL import Image
from io import BytesIO
import base64
import random
import torch
from transformers import StoppingCriteria
from medrax.llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        # sample a random between 0 and (width - height) // 2
        y_start = random.randint((width - height) // 2, (width - height) // 2 + 1)
        result.paste(pil_img, (0, y_start))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        # sample a random between 0 and (height - width) // 2
        x_start = random.randint((height - width) // 2, (height - width) // 2 + 1)
        result.paste(pil_img, (x_start, 0))
        return result


def process_images(images, image_processor, model_cfg):
    """
    Preprocess a list of PIL images for the vision tower.

    - Optionally pad to square if model_cfg.image_aspect_ratio == "pad".
    - Use the CLIP image_processor to resize/normalize into pixel_values.
    - Return a stacked tensor if all images end up the same shape; otherwise return a list of tensors.
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    for image in images:
        if image_aspect_ratio == "pad":
            if image.mode == "L":
                background_color = int(
                    255 * sum(image_processor.image_mean) / len(image_processor.image_mean)
                )
            else:
                background_color = tuple(int(x * 255) for x in image_processor.image_mean)
            image = expand2square(image, background_color)
        image = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        new_images.append(image)
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None) -> torch.Tensor | list[int]:
    """
    Build text input_ids with <image> placeholders for multimodal LLaVA.

    Where it is used:
    - Called when preparing prompts that will later go through
      `prepare_inputs_labels_for_multimodal` in `medrax/llava/model/llava_arch.py`.
      That function replaces each IMAGE_TOKEN_INDEX with projected image features.

    What it does (analytic view):
    1) Split the prompt string on the literal "<image>" marker.
    2) Tokenize each text chunk independently.
    3) Interleave the chunks with the special image token id (IMAGE_TOKEN_INDEX).
    4) If the tokenizer adds a BOS token, keep it once at the very front.

    Shapes / lengths:
    - Let K = number of text chunks = (# of "<image>" markers + 1).
    - Let T_i = token count of chunk i after tokenizer (including BOS on first chunk if present).
    - `prompt_chunks`: length-K ragged list with per-chunk lengths [T_0, T_1, ..., T_{K-1}].
    - `input_ids`: flat length T, where
      T = (sum_i T_i) - (BOS_removed ? 1 : 0) + (# of "<image>" markers).

    Returns:
    - If `return_tensors is None`: Python list[int] of length T.
    - If `return_tensors == "pt"`: torch.LongTensor with shape [T].

    Note:
    - The text length T referenced later in `llava_arch.py` is `len(cur_input_ids)`
      after padding is removed by the attention mask.
    """
    # Split prompt by "<image>" so we know where the image placeholder is
    # Each chunk is tokenized separately → list of token-id lists
    # prompt = "Describe this <image> and compare to this <image>."
    # prompt.split("<image>") -> [ "Describe this", " and comapre to this " , "."]
    # tokenizer(chunk).input_ids-> [ [ 1, 5, 232 ,2546] , [395, 3, 6], [2] ] 
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")] # list[list[int]]

    # Interleave: [chunk0, IMG, chunk1, IMG, chunk2, ...]
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    # If tokenizer adds BOS token to the first chunk, keep it once
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id ## begining of sequence token indicates start of prompt
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    # Interleave chunks with the special image token id, then flatten
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    # Return tensor if requested
    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids



def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith("checkpoint-"):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def call_for_batch(
        self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0] :] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
