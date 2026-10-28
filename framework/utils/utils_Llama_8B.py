import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from vllm import LLM, SamplingParams
from typing import List
import json
import datetime
import os
import numpy as np

# device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "./meta-llama/Meta-Llama-3.1-8B-Instruct"

class APILLM:
    def __init__(self, log_path="./log/tokens_log"):
        self.model = LLM(
            model=model_id,
            dtype="bfloat16",
            tensor_parallel_size=1,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
            seed=12,
            enable_prefix_caching=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            self.tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
        ]
        
        # Initialize token statistics
        self.token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "all_total_tokens": 0,
            "call_count": 0,
            "history": []  # detailed record of every call
        }
        self.log_path = log_path
        self.log_count = 0

    def count_input_tokens(self, messages):
        """Count the tokens of the input messages."""
        if isinstance(messages, str):
            # If a plain string, encode it directly
            return len(self.tokenizer.encode(messages))
        else:
            # If a message list, apply the chat template and then count
            chat_template_text = self.tokenizer.apply_chat_template(
                conversation=messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return len(self.tokenizer.encode(chat_template_text))

    def count_output_tokens(self, text):
        """Count the tokens of the output text."""
        return len(self.tokenizer.encode(text))

    def generate(self, messages, do_sample=True, temperature=0.01, max_new_tokens=4096):
        # Count input tokens
        input_tokens = self.count_input_tokens(messages)
        
        gen_params = SamplingParams(
            temperature=temperature,  # 0-1
            top_p=0.9,  # nucleus sampling
            top_k=50,  # top-k sampling
            max_tokens=max_new_tokens,  # max tokens to generate
            stop_token_ids=self.terminators,
            n=1,  # number of samples
            presence_penalty=0.1,  # penalize topic repetition
            frequency_penalty=0.1,  # penalize token repetition
            repetition_penalty=1.1,  # repetition penalty factor
        )
        
        inputs = self.tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False, 
            add_generation_prompt=True
        )
        
        response = self.model.generate(
            inputs,
            sampling_params=gen_params
        )
        
        result = response[0].outputs[0].text.strip()
        
        # Count output tokens
        output_tokens = self.count_output_tokens(result)
        
        # Update statistics
        self._update_token_stats(input_tokens, output_tokens, messages, result)
        
        return result

    def _update_token_stats(self, input_tokens, output_tokens, input_messages, output_text):
        """Update token statistics."""
        call_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "all_total_tokens": input_tokens + output_tokens,
            "input_messages": input_messages,
            "output_text": output_text
        }
        
        self.token_stats["total_input_tokens"] += input_tokens
        self.token_stats["total_output_tokens"] += output_tokens
        self.token_stats["all_total_tokens"] += (input_tokens + output_tokens)
        self.token_stats["call_count"] += 1
        self.token_stats["history"].append(call_record)
        self.save_token_stats()

    def reset_token_stats(self):
        """Reset token statistics."""
        self.token_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "all_total_tokens": self.token_stats["all_total_tokens"],
            "call_count": 0,
            "history": []
        }

    def save_token_stats(self):
        """Save token statistics to file."""
        filepath=f"{self.log_path}_{self.log_count}.json"
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.token_stats, f, ensure_ascii=False, indent=2)
        
        print(f"Token statistics saved to: {filepath}")
        return filepath
    

if __name__ == "__main__":
    
    api_llm = APILLM(log_path="./log/tokens_log_1")
    
    # First call
    response1 = api_llm.generate(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Make sure you carefully and fully understand the details of user's requirements before you start solving the problem."
            },
            {
                "role": "user",
                "content": "Hello."
            }
        ],
        do_sample=True,
        temperature=0.01
    )
    print("Response 1:", response1)

    # Save statistics to file
    api_llm.save_token_stats()
    
    # Reset statistics (optional)
    # api_llm.reset_token_stats()