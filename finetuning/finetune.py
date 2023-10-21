import argparse
from dataclasses import dataclass
from datasets import load_dataset, load_metric
import json
from math import exp
import numpy as np
import os
from pathlib import Path
from quinine import Quinfig
import random
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler
import sys
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    AutoConfig, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    BatchEncoding
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import WandbCallback, rewrite_logs
from typing import Any, Callable, Dict, List, Optional, Tuple
import wandb

def main(quinfig, config_file, wandb_ids_cache, wandb_offline = False):
    HF_CACHE_DIR = #TODO
    os.environ["WANDB_PROJECT"] = quinfig.general.wandb_project
    os.environ["WANDB_WATCH"] = "all"
    # Set Randomness. Probably not needed because Trainer does this since we  use model_init?
    #print(f"Setting Random Seed to {quinfig.training.seed}!")
    #random.seed(quinfig.training.seed)
    #np.random.seed(quinfig.training.seed)
    #torch.manual_seed(quinfig.training.seed)
    def model_init():
        num_name_types_given = sum(["model_name" in quinfig.general,
                                    "huggingface_model_name" in quinfig.general,
                                    "model_config_name" in quinfig.general])
        assert num_name_types_given == 1, f"Please provide one of model_name, huggingface_model_name, model_config_name', you're currently giving {num_name_types_given}"
        if "model_name" in quinfig.general:
            assert "checkpoint_number" in quinfig.general, "Need a checkpoint number since we have a model_name!"
            checkpoint_path = f"pretrained_checkpoints/{quinfig.general.model_name}/checkpoint-{quinfig.general.checkpoint_number}"
            if "gpt2" in quinfig.general.model_type:
                model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            elif "bert" in quinfig.general.model_type:
                model = AutoModelForMaskedLM.from_pretrained(checkpoint_path)
        elif "huggingface_model_name" in quinfig.general:
            if "gpt2" in quinfig.general.model_type:
                model = AutoModelForCausalLM.from_pretrained(quinfig.general.huggingface_model_name)
            elif "bert" in quinfig.general.model_type:
                model = AutoModelForMaskedLM.from_pretrained(quinfig.general.huggingface_model_name)
        elif "model_config_name" in quinfig.general:
            print(f"Initializing new model from config at {quinfig.general.model_config_name}")
            # NOTE haven't really tested how autoconfig works and if this ok!
            config = AutoConfig.from_pretrained(quinfig.general.model_config_name)
            config.vocab_size = len(tokenizer)
            if "gpt2" in quinfig.general.model_type:
                model = AutoModelForCausalLM.from_config(config)
            elif "bert" in quinfig.general.model_type:
                model = AutoModelForMaskedLM.from_config(config)
        if quinfig.general.embeddings_strategy == "leave_embeddings_alone":
            print("Leaving embeddings as they are!")
            assert model.get_input_embeddings().num_embeddings >= len(tokenizer)
        elif "sample" in quinfig.general.embeddings_strategy:
            old_embeddings = model.get_input_embeddings()
            new_embeddings = nn.Embedding(len(tokenizer), model.config.hidden_size)
            if quinfig.general.embeddings_strategy == "sample":
                old_embeddings_sample_idxs = torch.randint(0, old_embeddings.num_embeddings, (new_embeddings.num_embeddings,))
            elif "just" in quinfig.general.embeddings_strategy:
                num_to_sample_from = int(quinfig.general.embeddings_strategy.split("just")[-1])
                assert num_to_sample_from <= old_embeddings.num_embeddings
                old_idxs_to_sample = np.random.choice(np.arange(old_embeddings.num_embeddings), num_to_sample_from, replace=False)
                old_embeddings_sample_idxs = np.random.choice(old_idxs_to_sample, (new_embeddings.num_embeddings,), replace=True)
                old_embeddings_sample_idxs = torch.LongTensor(old_embeddings_sample_idxs)
            new_embeddings.weight.data = old_embeddings.weight[old_embeddings_sample_idxs].data
            model.set_input_embeddings(new_embeddings)
            model.tie_weights()
            model.config.vocab_size = len(tokenizer)
            assert model.get_input_embeddings().num_embeddings >= len(tokenizer)
        elif "use_pretrained" in quinfig.general.embeddings_strategy:
            print(f"Using pretrained embeddings from {quinfig.data.embedding_model_name}")
            if "gpt2" in quinfig.data.embedding_model_name:
                emb_model = AutoModelForCausalLM.from_pretrained(quinfig.data.embedding_model_name)
            elif "bert" in quinfig.general.embedding_model_name:
                emb_model = AutoModelForMaskedLM.from_pretrained(quinfig.data.embedding_model_name)
            else:
                print("What model should we take the embeddings from?")
                sys.exit(1)
            pretrained_embeddings = emb_model.get_input_embeddings()
            new_embeddings = nn.Embedding(pretrained_embeddings.num_embeddings, pretrained_embeddings.embedding_dim)
            new_embeddings.weight.data = pretrained_embeddings.weight.detach()
            model.set_input_embeddings(new_embeddings)
            model.tie_weights()
            model.config.vocab_size = model.get_input_embeddings().num_embeddings
            assert model.get_input_embeddings().num_embeddings >= len(tokenizer)
            if quinfig.general.embeddings_strategy == "use_pretrained_frozen":
                print("Freezing embeddings!")
                model.get_input_embeddings().requires_grad_(False)
        else:
            print("Invalid embeddings strategy")
            sys.exit(1)
        if "trainable_params" in quinfig.general and quinfig.general.trainable_params == "embeddings":
            print("Only training word embeddings!")
            for name, param in model.named_parameters():
                if "word_embeddings" not in name:
                    param.requires_grad = False
        elif "trainable_params" in quinfig.general and quinfig.general.trainable_params == "bitfit":
            print("Only training bits!")
            for name, param in model.named_parameters():
                if "bias" not in name:
                    param.requires_grad = False
        return model
    tokenizer = AutoTokenizer.from_pretrained(quinfig.data.tokenizer_name, use_fast=True)
    dataset = load_dataset(quinfig.data.type, quinfig.data.name)
    hf_cache = Path(HF_CACHE_DIR)
    model = model_init()
    # TODO: make sure that this is ok for non-filename data types like wikitext?
    hf_tokenizer_cache = hf_cache / "preprocessed" / os.path.splitext(os.path.basename(quinfig.data.type))[0] / quinfig.data.name / quinfig.data.tokenizer_name.replace("/", "_")
    tokenize_dir = hf_tokenizer_cache / "tokenize"
    concat_dir = hf_tokenizer_cache / "concat"
    # TODO: change this back
    block_size = 512
    #block_size = model.config.max_position_embeddings
    tokenize_cache_files = {k: str(tokenize_dir / f"{k}-tokenized.hf") for k in dataset }
    concat_cache_files = {k: str(concat_dir / f"{k}-concated-stride{block_size}.hf") for k in dataset }
    tokenize_dir.mkdir(parents=True, exist_ok=True)
    concat_dir.mkdir(parents=True, exist_ok=True)
    def tokenize_function(examples):
        return tokenizer(examples["text"])
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"],
        cache_file_names=tokenize_cache_files)
    lm_dataset = tokenized_dataset.map(group_texts, batched=True, batch_size=1000,
        cache_file_names=concat_cache_files)
    if "crop_train" in quinfig.data and quinfig.data.crop_train > 0:
        print(f"Cropping training data to {quinfig.data.crop_train} rows")
        lm_dataset["train"] = lm_dataset["train"].select(range(quinfig.data.crop_train))
    lm_dataset.set_format(type = "torch")
    if "gpt2" in quinfig.general.model_type:
        data_collator = LMDataCollator(tokenizer=tokenizer)
    elif "bert" in quinfig.general.model_type:
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    run_name = f"{quinfig.general.nickname}"
    if "trainable_params" in quinfig.general:
        run_name = f"{run_name}_train-{quinfig.general.trainable_params}"
    if "embeddings_strategy" in quinfig.general and quinfig.general.embeddings_strategy != "sample":
        run_name = f"{run_name}_embs-{quinfig.general.embeddings_strategy}"
    if "seed" in quinfig.training:
        run_name = f"{run_name}_seed{quinfig.training.seed}"
        
    if torch.cuda.get_device_properties(0).total_memory > 40_000_000_000:
        quinfig.training.per_device_train_batch_size = 16
    elif torch.cuda.get_device_properties(0).total_memory > 20_000_000_000:
        quinfig.training.per_device_train_batch_size = 8
    else:
        quinfig.training.per_device_train_batch_size = 4
    quinfig.training["gradient_accumulation_steps"] = quinfig.general.effective_batch_size // quinfig.training.per_device_train_batch_size
    print(f"Device batch size {quinfig.training.per_device_train_batch_size}, grad acc {quinfig.training.gradient_accumulation_steps}, for effecitve batch size {quinfig.training.per_device_train_batch_size}*{quinfig.training.gradient_accumulation_steps}")

    has_started_before = False
    last_checkpoint = None
    if "resume" in quinfig.general and quinfig.general.resume:
        if os.path.isdir(os.path.join(quinfig.general.save_dir, run_name)):
            has_started_before = True
            last_checkpoint = get_last_checkpoint(os.path.join(quinfig.general.save_dir, run_name))
    training_arguments = TrainingArguments(
        run_name = run_name,
        output_dir = os.path.join(quinfig.general.save_dir, run_name),
        **quinfig.training.toDict())
    trainer = Trainer(
            args=training_arguments,
            model_init=model_init,
            train_dataset=lm_dataset["train"],
            # TODO change this for other experiments! There's gotta be a better way
            eval_dataset=lm_dataset["validation"].filter(lambda example, idx: idx % 10 == 0, with_indices=True),
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks = [WandbPplCallback])
    if wandb_ids_cache:
        wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                mode = "offline" if wandb_offline else "online",
                group = run_name.split("_seed")[0] if "wandb_seed_group" in quinfig.general and quinfig.general.wandb_seed_group else None,
                #group=quinfig.general.wandb_group if "wandb_group" in quinfig.general else None,
                name=run_name,
                id=wandb_ids_cache[config_file],
                resume = True if has_started_before else False)
    else:
        wandb.init(
                project=os.getenv("WANDB_PROJECT", "huggingface"),
                mode = "offline" if wandb_offline else "online",
                group = run_name.split("_seed")[0] if "wandb_seed_group" in quinfig.general and quinfig.general.wandb_seed_group else None,
                #group=quinfig.general.wandb_group if "wandb_group" in quinfig.general else None,
                name=run_name)
    wandb.config.update(quinfig, allow_val_change=True)

    if "hp_tune" in quinfig.general and quinfig.general.hp_tune == True:
        hp_tune_config = {
            "learning_rate": tune.loguniform(*quinfig.training_hp.learning_rate),
            "warmup_ratio": tune.uniform(*quinfig.training_hp.warmup_ratio),
            "gradient_accumulation_steps": \
                tune.choice([effective / quinfig.training.per_device_train_batch_size for \
                effective in quinfig.training_hp.effective_batch_size])
        }
        best_trial = trainer.hyperparameter_search(
            hp_space=lambda _: hp_tune_config,
            backend="ray", 
            local_dir="~/scr/ray_results",
            n_trials=10,
            keep_checkpoints_num=1,
            search_alg=HyperOptSearch(metric="objective", mode="min"),
            scheduler=ASHAScheduler(metric="objective", mode="min")
            )
        json.dump((best_trial, quinfig), open(os.path.join("hp_search_results", run_name + ".json"), "w"))
    elif last_checkpoint:
        print(f"Resuming from {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.evaluate()
        trainer.train()
    wandb.finish()

# Lifted from mistral
@dataclass
class LMDataCollator:
    tokenizer: PreTrainedTokenizerBase

    def __call__(self, examples: List[BatchEncoding]):
        batch = BatchEncoding(data={k: torch.cat([v[k].unsqueeze(0) for v in examples]) for k in examples[0].keys()})

        if "labels" in batch:
            labels = batch["labels"]
        else:
            labels = batch["input_ids"]

        if self.tokenizer.pad_token_id is not None:
            labels = labels.clone()
            labels[labels == self.tokenizer.pad_token_id] = -100

        batch["labels"] = labels
        return batch


class WandbPplCallback(WandbCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if not self._initialized:
            self.setup(args, state, model)
        if state.is_world_process_zero:
            logs = rewrite_logs(logs)
            if "eval/loss" in logs:
                logs["eval/perplexity"] = exp(logs["eval/loss"])
            self._wandb.log({**logs, "train/global_step": state.global_step})


from transformers.integrations import WandbCallback

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str)
    parser.add_argument('--wandb-offline', default=False, action='store_true')
    parser.add_argument('--wandb-ids-cache', type=str)
    args = parser.parse_args()
    print("Parsing quinfig...")
    quinfig = Quinfig(args.config_file)
    config_filename = os.path.splitext(os.path.basename(args.config_file))[0]
    wandb_ids_cache = json.load(open(args.wandb_ids_cache)) if args.wandb_ids_cache else None
    main(quinfig, config_filename, wandb_ids_cache, args.wandb_offline)


