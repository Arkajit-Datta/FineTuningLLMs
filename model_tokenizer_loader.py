import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftConfig,
    PeftModel,
    AutoPeftModelForCausalLM
)

class LoadModelandTokenizer:
    
    def __init__(self, Base_Model, Dataset_Name, New_Model, float_16_dtype, use_bf16, use_4bit_bnb, compute_dtype, bnb_config, peft_config): 
        self.BASE_MODEL = Base_Model
        self.DATASET_NAME = Dataset_Name
        self.NEW_MODEL = New_Model
        self.DEVICE_MAP = {"": 0}
        self.float_16_dtype = float_16_dtype
        self.use_bf16 = use_bf16
        self.use_4bit_bnb = use_4bit_bnb
        self.compute_dtype = compute_dtype
        self.bnb_config =bnb_config
        self.peft_config = peft_config
        
        self.check_gpu_and_set_float_dtype()
        
    def check_gpu_and_set_float_dtype(self):
        # Check GPU compatibility with bfloat16
        # If the gpu is 'bf16' compatible, set the flag to `True`
        if self.compute_dtype == torch.float16 and self.use_4bit_bnb:
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                print("=" * 80)
                print("Your GPU supports bfloat16: accelerate training with bf16=True")
                print("Changing floating point type to `torch.bfloat16`")
                self.float_16_dtype = torch.bfloat16
                self.use_bf16 = True
                print("=" * 80)
            else:
                print("Your GPU does not support bfloat16")
                
    def load_tokenizer(self):
        # Loading the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL, trust_remote_code=True)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Supress fast_tokenizer warning
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        return tokenizer
    
    def load_model(self):
        # Load the model
        # Loading the model
        if self.use_4bit_bnb:
            model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                quantization_config=self.bnb_config,
                device_map=self.DEVICE_MAP
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                device_map=self.DEVICE_MAP
            )
        model.config.use_cache = False
        model.config.pretraining_tp = 1
        # Some [optional] pre-processing which
        # helps improve the stability of the training
        for param in model.parameters():
            param.requires_grad = False  # freeze the model - train adapters later
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm) to fp32 for stability
                param.data = param.data.to(torch.float32)

        model.gradient_checkpointing_enable()  # reduce number of stored activations
        model.enable_input_require_grads()

        class CastOutputToFloat(torch.nn.Sequential):
            def forward(self, x): return super().forward(x).to(torch.float32)
        
        model.lm_head = CastOutputToFloat(model.lm_head)
        print("Printing the model params!")
        self.print_trainable_parameters(model)
        model = get_peft_config(model, self.peft_config)
        return model
    
    def print_trainable_parameters(self, model):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
    
    @classmethod
    def load(self):
        return self.load_tokenizer(), self.load_model()