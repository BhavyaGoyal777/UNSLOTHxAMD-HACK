import time
from typing import Optional, Union, List
from unsloth import FastLanguageModel
import torch


class QAgent(object):
    def __init__(self, **kwargs):
        model_name = "FINAL_MODELS/q_agent_llama/merged_16bit"

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=False,
        )

        FastLanguageModel.for_inference(self.model)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Set padding_side to left for decoder-only models
        self.tokenizer.padding_side = 'left'

    def generate_response(
        self, message: str | List[str], system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        # Keep system and user as separate messages, just like during training
        if isinstance(message, str):
            message = [message]

        all_messages = []
        for msg in message:
            messages = []
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            # Add user message
            messages.append({"role": "user", "content": msg})

            all_messages.append(messages)

        texts = []
        for messages in all_messages:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

        # Don't truncate - model needs full prompt like in training
        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.model.device)

        tgps_show_var = kwargs.get("tgps_show", False)

        if tgps_show_var:
            start_time = time.time()

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            pad_token_id=self.tokenizer.pad_token_id,
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("do_sample", True),
        )

        if tgps_show_var:
            generation_time = time.time() - start_time

        batch_outs = []
        if tgps_show_var:
            token_len = 0

        for i, (input_ids, generated_sequence) in enumerate(
            zip(model_inputs.input_ids, generated_ids)
        ):
            output_ids = generated_sequence[len(input_ids):].tolist()

            if tgps_show_var:
                token_len += len(output_ids)

            content = self.tokenizer.decode(
                output_ids, skip_special_tokens=True
            ).strip()
            batch_outs.append(content)

        if tgps_show_var:
            return (
                batch_outs[0] if len(batch_outs) == 1 else batch_outs,
                token_len,
                generation_time,
            )
        return batch_outs[0] if len(batch_outs) == 1 else batch_outs, None, None
