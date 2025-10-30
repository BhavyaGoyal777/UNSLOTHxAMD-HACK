import time
from typing import Optional, List, Tuple
from unsloth import FastLanguageModel
import torch


class QAgent(object):
    """Question generation model wrapper"""

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

        self.tokenizer.padding_side = 'left'

    def generate_response(
        self,
        message: str | List[str],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Tuple[str | List[str], int | None, float | None]:
        """Generate response(s) from message(s)"""
        if isinstance(message, str):
            message = [message]

        all_messages = []
        for msg in message:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": msg})
            all_messages.append(messages)

        texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            for messages in all_messages
        ]

        model_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False
        ).to(self.model.device)

        tgps_show = kwargs.get("tgps_show", False)
        start_time = time.time() if tgps_show else None

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=kwargs.get("max_new_tokens", 1024),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=kwargs.get("temperature", 0.3),
            top_p=kwargs.get("top_p", 0.9),
            do_sample=kwargs.get("do_sample", True),
            repetition_penalty=kwargs.get("repetition_penalty", 1.2),
        )

        generation_time = (time.time() - start_time) if tgps_show else None

        outputs = []
        total_tokens = 0

        for input_ids, generated_sequence in zip(model_inputs.input_ids, generated_ids):
            output_ids = generated_sequence[len(input_ids):].tolist()
            total_tokens += len(output_ids)
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(content)

        result = outputs[0] if len(outputs) == 1 else outputs
        return result, total_tokens if tgps_show else None, generation_time
