import os
from peft import (
    get_peft_config,
    get_peft_model,
    LoraConfig,
    PeftConfig,
    PeftModel,
    AutoPeftModelForCausalLM
)
import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from trl import SFTTrainer
from typing import Optional
from accelerate import Accelerator
from utils import check_cuda_available_and_assign_device, login_huggingface, clean_objects_and_empty_gpu_cache
from dataset_loader import DatasetLoader
from model_tokenizer_loader import LoadModelandTokenizer

import locale
locale.getpreferredencoding = lambda: "UTF-8"

accelerator = Accelerator()

# Device map
DEVICE_MAP = {"": 0}
DEVICE = check_cuda_available_and_assign_device(accelerator)

#Model Config 
BASE_MODEL = "Mistral-7B-Instruct-v0.1"
DATASET_NAME = "sayan1101/identity_finetune_data_3"
NEW_MODEL = "finetuned_on_zephyr_sft_v_1"


LORA_TARGET_MODULES_LLAMA_2 = [
    "q_proj",
    "o_proj",
    "v_proj"
    "k_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
]

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=LORA_TARGET_MODULES_LLAMA_2,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

float_16_dtype = torch.float16
use_bf16 = True
use_4bit_bnb = False
compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit_bnb,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=float_16_dtype,
    bnb_4bit_use_double_quant=False,
)
# Training arguments
OUTPUT_DIR = "./results"
LEARNING_RATE = 1e-4

NUM_EPOCHS = 1
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 8 # effective backprop @ batch_size*grad_accum_steps
GRADIENT_CHECKPOINTING = True # speed down by ~20%, improves mem. efficiency

OPTIMIZER = "adamw_hf"
# OPTIMIZER = "adamw_torch_fused" # use with pytorch compile
WEIGHT_DECAY = 0.1
LR_SCHEDULER_TYPE = "cosine" # examples include ["linear", "cosine", "constant"]
MAX_GRAD_NORM = 1 # clip the gradients after the value
WARMUP_RATIO = 0.1 # The lr takes 3% steps to reach stability

SAVE_STRATERGY = "steps"
SAVE_STEPS = 10
SAVE_TOTAL_LIMIT = 5
LOAD_BEST_MODEL_AT_END = True

#REPORT_TO = "wandb"
LOGGING_STEPS = 1
EVAL_STEPS = SAVE_STEPS

PACKING = True
MAX_SEQ_LENGTH = 2048

def calculate_steps():
    dataset_size = len(train_dataset)
    steps_per_epoch = dataset_size / (BATCH_SIZE * GRAD_ACCUMULATION_STEPS)
    total_steps = steps_per_epoch * NUM_EPOCHS

    print(f"Total number of steps: {total_steps}")

calculate_steps()

# Load the model and tokenizer
tokenizer, model = LoadModelandTokenizer(BASE_MODEL, DATASET_NAME, NEW_MODEL, float_16_dtype=float_16_dtype, use_bf16=use_bf16, use_4bit_bnb=use_4bit_bnb, compute_dtype=compute_dtype, bnb_config=bnb_config, peft_config=peft_config).load()

# Load the dataset
dataset_loader = DatasetLoader(DATASET_NAME, tokenizer)
train_dataset = dataset_loader.get_dataset()

training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,

    optim=OPTIMIZER,
    weight_decay=WEIGHT_DECAY,
    max_grad_norm=MAX_GRAD_NORM,
    fp16=not use_bf16,
    bf16=use_bf16,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type=LR_SCHEDULER_TYPE,

    # torch_compile=False,
    group_by_length=False,

    save_strategy=SAVE_STRATERGY,
    save_steps=SAVE_STEPS,
    save_total_limit=SAVE_TOTAL_LIMIT,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,

    # evaluation_strategy=SAVE_STRATERGY,
    # eval_steps=EVAL_STEPS,

    dataloader_pin_memory=True,
    dataloader_num_workers=4,

    logging_steps=LOGGING_STEPS,
    # report_to=REPORT_TO,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    peft_config=peft_config,

    train_dataset=train_dataset,
    # eval_dataset=valid_dataset,
    dataset_text_field="text",

    args=training_arguments,
    max_seq_length=MAX_SEQ_LENGTH,
    packing=PACKING,
)

trainer.train()
trainer.model.save_pretrained(NEW_MODEL)