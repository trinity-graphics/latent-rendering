import gc

import mitsuba as mi
import numpy as np
import torch
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)


def get_blip2_caption(model, image, num_tokens = 32, dtype: torch.dtype = torch.float32, device=torch.device("cpu")):
    with torch.no_grad():
        # Convert image to UInt8
        print(image.shape)
        image = mi.Bitmap(np.asarray(image.cpu().detach().squeeze(0).permute(1, 2, 0))).convert(
            pixel_format=mi.Bitmap.PixelFormat.RGB,
            component_format=mi.Struct.Type.UInt8,
            srgb_gamma=False,
        )
        image = np.array(image)
        
        processor = AutoProcessor.from_pretrained(model, use_fast=True)
        model = Blip2ForConditionalGeneration.from_pretrained(
            model, dtype=dtype, use_safetensors=True
        ).to(device)

        # Generate image caption
        inputs = processor(image, return_tensors="pt").to(device, dtype)
        generated_ids = model.generate(**inputs, max_new_tokens=num_tokens)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        del processor, model, inputs, generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return generated_text

def get_clip_embeddings(model, prompt: str, dtype: torch.dtype = torch.float32, device=torch.device("cpu")):
    with torch.no_grad():
        tokenizer = CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", dtype=dtype, use_safetensors=True)
        text_encoder = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", dtype=dtype, use_safetensors=True).to(device)
    
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        embeddings = text_encoder(tokens)["last_hidden_state"].to(device)

        del tokenizer, text_encoder, tokens
        gc.collect()
        torch.cuda.empty_cache()

        return embeddings

def get_sd3_embeddings(
    model_id: str, 
    prompt: str, 
    dtype: torch.dtype = torch.float16, 
    device=torch.device("cuda")
):
    with torch.no_grad():
        # 1. Load all three tokenizers and encoders
        # In production, you'd want to load these once in the __init__ of your SDS class
        tokenizers = []
        tokenizers.append(CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer"))
        tokenizers.append(CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2"))
        tokenizers.append(T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_3"))

        text_encoders = []
        text_encoders.append(CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=dtype
        ).to(device))
        text_encoders.append(CLIPTextModelWithProjection.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=dtype
        ).to(device))
        text_encoders.append(T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_3", torch_dtype=dtype
        ).to(device))

        def encode_single(text):
            # 1. CLIP-L and CLIP-G
            clip_embeds_list = []
            clip_pooled_list = []
            
            for i in range(2):
                tokenizer = tokenizers[i]
                encoder = text_encoders[i]
                
                inputs = tokenizer(
                    text, padding="max_length", max_length=77, 
                    truncation=True, return_tensors="pt"
                ).to(device)
                
                # SD3 uses the penultimate hidden state
                outputs = encoder(inputs.input_ids, output_hidden_states=True)
                clip_embeds_list.append(outputs.hidden_states[-2])
                clip_pooled_list.append(outputs.text_embeds)

            # Concat CLIPs on feature dim: [1, 77, 2048]
            clip_prompt_embeds = torch.cat(clip_embeds_list, dim=-1)
            pooled_prompt_embeds = torch.cat(clip_pooled_list, dim=-1)

            # 2. T5-XXL
            t5_inputs = tokenizers[2](
                text, padding="max_length", max_length=256, 
                truncation=True, return_tensors="pt"
            ).to(device)
            t5_prompt_embeds = text_encoders[2](t5_inputs.input_ids)[0]

            # 3. Padding and Sequence Concatenation
            # Pad CLIP (2048) to match T5 (4096) on the feature dimension
            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1])
            )
            
            # Concat along sequence dimension (dim -2): [1, 77 + 256, 4096]
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
            
            return prompt_embeds, pooled_prompt_embeds

        with torch.no_grad():
            cond_embeds, cond_pooled = encode_single(prompt)
            uncond_embeds, uncond_pooled = encode_single("")

        out = {
            "context": cond_embeds.to(dtype),
            "pooled_context": cond_pooled.to(dtype),
            "uncond_context": uncond_embeds.to(dtype),
            "uncond_pooled_context": uncond_pooled.to(dtype)
        }

        text_encoders = tokenizers = None
        gc.collect()
        torch.cuda.empty_cache()

        return out