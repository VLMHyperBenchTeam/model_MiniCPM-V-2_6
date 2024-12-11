import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoProcessor

class MiniCPM_model:
    def __init__(self, cache_directory, model_name="openbmb/MiniCPM-V-2_6"):
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_directory,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_directory,
            fast_image_processor_class=True
        )

    def chat(self, image, msgs):
        return self.model.chat(image=image, msgs=msgs, tokenizer=self.tokenizer)
    
