import os
import json
import argparse
import random
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()
def apply_template(
    tokenizer,
    messages,
    add_generation_prompt=False,
):
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

def build_preprocess_function(
    tokenizer,
    max_length,
):
    def preprocess(example):
        messages = example["messages"]
        if (
            not isinstance(messages, list)
            or len(messages) < 2
        ):
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "valid": False,
            }
        user_message = messages[0]
        assistant_message = messages[1]
        if (
            user_message.get("role") != "user"
            or assistant_message.get("role") != "assistant"
        ):
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "valid": False,
            }
        prompt_text = apply_template(
            tokenizer,
            [user_message],
            add_generation_prompt=True,
        )
        full_text = apply_template(
            tokenizer,
            [
                user_message,
                assistant_message,
            ],
            add_generation_prompt=False,
        )
        prompt_tokens = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )
        prompt_length = len(
            prompt_tokens["input_ids"]
        )
        full_tokens = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )
        input_ids = full_tokens["input_ids"]
        attention_mask = full_tokens[
            "attention_mask"
        ]
        labels = input_ids.copy()

        masked_length = min(
            prompt_length,
            len(labels),
        )
        for i in range(masked_length):
            labels[i] = -100
        valid_label_count = sum(
            1 for x in labels if x != -100
        )
        valid = valid_label_count > 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "valid": valid,
        }
    return preprocess
def print_token_stats(dataset):
    if len(dataset) == 0:
        return
    lengths = [
        len(x["input_ids"])
        for x in dataset
    ]
    label_lengths = [
        sum(
            token != -100
            for token in x["labels"]
        )
        for x in dataset
    ]
    lengths_sorted = sorted(lengths)
    print("\n" + "=" * 60)
    print("TOKEN STATISTICS")
    print("=" * 60)
    print(
        f"Samples              : {len(dataset)}"
    )
    print(
        f"Average total tokens : "
        f"{sum(lengths) / len(lengths):.2f}"
    )
    print(
        f"Median total tokens  : "
        f"{lengths_sorted[len(lengths_sorted)//2]}"
    )
    print(
        f"Maximum total tokens : "
        f"{max(lengths)}"
    )
    print(
        f"Average target tokens: "
        f"{sum(label_lengths) / len(label_lengths):.2f}"
    )
    print("=" * 80)
def main():
    args = parse_args()
    set_seed(args.seed)
    print("=" * 80)
    print("Qwen3-8B CSE LoRA SFT")
    print("=" * 80)
    print(f"Model       : {args.model_path}")
    print(f"Train file  : {args.train_file}")
    print(f"Output      : {args.output_dir}")
    print(f"Max length  : {args.max_length}")
    print(f"Epochs      : {args.epochs}")
    print(f"LR          : {args.learning_rate}")
    print(f"Batch       : {args.batch_size}")
    print(
        f"Grad accum  : "
        f"{args.gradient_accumulation_steps}"
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available."
        )
    print(
        f"\nGPU: "
        f"{torch.cuda.get_device_name(0)}"
    )
    total_mem = (
        torch.cuda.get_device_properties(0)
        .total_memory
        / 1024**3
    )
    print(
        f"GPU memory: {total_mem:.2f} GiB"
    )
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )
    tokenizer.padding_side = "right"
    print("\nLoading dataset...")
    dataset = load_dataset(
        "json",
        data_files=args.train_file,
        split="train",
    )
    print(
        f"Raw samples: {len(dataset)}"
    )
    split = dataset.train_test_split(
        test_size=args.val_ratio,
        seed=args.seed,
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(
        f"Train samples: {len(train_dataset)}"
    )
    print(
        f"Eval samples : {len(eval_dataset)}"
    )
    preprocess_fn = (
        build_preprocess_function(
            tokenizer,
            args.max_length,
        )
    )
    print("\nTokenizing training data...")
    train_dataset = train_dataset.map(
        preprocess_fn,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing train",
    )
    train_dataset = train_dataset.filter(
        lambda x: x["valid"]
    )
    train_dataset = train_dataset.remove_columns(
        ["valid"]
    )
    print("\nTokenizing validation data...")
    eval_dataset = eval_dataset.map(
        preprocess_fn,
        remove_columns=eval_dataset.column_names,
        desc="Tokenizing validation",
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["valid"]
    )
    eval_dataset = eval_dataset.remove_columns(
        ["valid"]
    )
    print(
        f"\nValid train samples: "
        f"{len(train_dataset)}"
    )
    print(
        f"Valid eval samples : "
        f"{len(eval_dataset)}"
    )
    print_token_stats(
        train_dataset
    )
    print("\nLoading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    print("\nApplying LoRA...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
    )
    model = get_peft_model(
        model,
        lora_config,
    )
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=(
            args.batch_size
        ),
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=True,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        optim="adamw_torch",
        max_grad_norm=1.0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    print("\n" + "=" * 80)
    print("START TRAINING")
    print("=" * 80)
    trainer.train()
    print("\nEvaluating...")
    metrics = trainer.evaluate()
    print("\nFinal evaluation:")
    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )
    final_dir = os.path.join(
        args.output_dir,
        "final_adapter",
    )
    print(
        f"\nSaving LoRA adapter to:"
        f"\n{final_dir}"
    )
    trainer.model.save_pretrained(
        final_dir
    )
    tokenizer.save_pretrained(
        final_dir
    )
    config_path = os.path.join(
        final_dir,
        "cse_training_config.json",
    )
    with open(
        config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            vars(args),
            f,
            ensure_ascii=False,
            indent=2,
        )
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
if __name__ == "__main__":
    main()