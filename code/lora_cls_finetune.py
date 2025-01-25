import os, json, pickle, torch, wandb
if __name__ == "__main__":
    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    
torch.manual_seed(42)
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, TrainingArguments, Trainer, DefaultDataCollator, get_linear_schedule_with_warmup, BitsAndBytesConfig, TrainerCallback
from dataset import XNLIDatasetHF
from utils import models_map
from models import ModelForCLSWithLoRA
import evaluate, torch
import numpy as np
from typing import List, Tuple, Union, Any

class LoRAFineTuner:
    def __init__(self, device: torch.device, config: dict):
        self.config = config
        self.device = device
        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',  
            bnb_4bit_compute_dtype=torch.bfloat16,  
            bnb_4bit_use_double_quant=True,  
        ) if self.config["is_4bit_quant"] else None
        
        self.model_name = config["model_name"]
        self.model_name_srt = self.model_name.split("/")[-1]
        self.lang = config["lang"]
        self.method = config["method"]
        self.finetune_lang = config["finetune_lang"]
        self.train_ds = XNLIDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"], frac=self.config["train_frac"], is_train=True)
        self.eval_ds = XNLIDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"], frac=self.config["eval_frac"], is_train=False)
        self.data_collator = DefaultDataCollator(return_tensors="pt")
        self.tokenizer = self.train_ds.tokenizer

        self.batch_size = self.config["batch_size"]
        self.config["num_steps"] = len(self.train_ds)//self.batch_size
        self.num_steps = self.config["num_steps"]
        self.num_epochs = self.config["num_epochs"]
        self.project_name = f"{self.model_name.split('/')[-1]}_finetune_{self.config['task_name']}"
        os.environ["WANDB_PROJECT"] = self.project_name
        self.run_name = f"{self.method}/{self.lang}_finetune_{self.finetune_lang}_{self.config['train_frac']:.2f}_{self.config['initial_lr']:.1e}_r{self.config['lora_rank']}"
        self.output_dir = f"outputs/ckpt/{self.project_name}/{self.run_name}"

        self.frozen_neurons = self._get_frozen_neurons()
        self.model = ModelForCLSWithLoRA(device=self.device, tokenizer=self.tokenizer, model_name=self.model_name, sparse_alpha=None, num_class=self.config["num_class"], lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"], quant_config=self.quant_config, frozen_neurons=self.frozen_neurons)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=int(0.01 * self.num_steps), 
                                                         num_training_steps=int(self.num_epochs * self.num_steps))

        self.train_arg_config = {
            "output_dir": self.output_dir,
            "eval_strategy": "no",
            # "eval_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
            "per_device_train_batch_size": self.batch_size,
            "per_device_eval_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.config["grad_acc_steps"],
            "max_grad_norm": self.config["max_grad_norm"],
            "num_train_epochs": self.num_epochs,
            "logging_strategy": "steps",
            "logging_first_step": True,
            "logging_steps": self.batch_size,
            "save_strategy": "steps",
            "save_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
            "save_safetensors": False, 
            "save_total_limit": self.num_epochs * self.config["num_ckpt_per_epoch"] + 1,
            "save_only_model": False,
            "fp16": self.config["fp16"],
            "bf16": self.config["bf16"],
            "dataloader_drop_last": True,
            "run_name": self.run_name,
            "report_to": "wandb" if self.config["wandb_log"] else "none",
            "eval_on_start": False
        }

        self.training_args = TrainingArguments(**self.train_arg_config)
        self.trainer_config = {
            "model": self.model,
            "args": self.training_args,
            "data_collator": self.data_collator,
            "train_dataset": self.train_ds,
            # "eval_dataset": self.eval_ds,
            "tokenizer": self.tokenizer,
            "optimizers": (self.optimizer, self.scheduler),
            # "compute_metrics": self.compute_metrics
        }
        self.trainer = Trainer(**self.trainer_config)
    
    @staticmethod
    def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> dict:
        predictions, labels = eval_pred
        min_len = min(len(predictions), len(labels))
        predictions = np.argmax(predictions[:min_len], axis=1)
        labels = labels[:min_len]
        accuracy = evaluate.load("accuracy")
        output = accuracy.compute(predictions=predictions, references=labels) # keys: {"accuracy"}
        return output
    
    @staticmethod
    def compute_loss(self, model: torch.nn.Module, inputs: dict, outputs: torch.tensor = None):
        """Not to be used by LoRAFineTuner.
        Go to: ~/miniconda3/envs/env_name/lib/python3.8/site-packages/transformers/trainer.py
        Modify Trainer.compute_loss() function.
        Add this code snippet just before the return statement
        """
        # Custom: Begin.
        if model.training:
            total_norm = 0.0
            for p in model.parameters():
                if p.requires_grad:
                    param_norm = p.norm(2).item()
                    total_norm += param_norm ** 2
            total_norm = total_norm ** 0.5

            preds = outputs["logits"].detach() # outputs is already calculate before
            acc = (preds.argmax(axis=1) == inputs["labels"]).to(torch.float).mean().item()
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({"accuracy": acc, "param_norm": total_norm})
        # Custom: End.
    
    def _get_lang_neuron(self, method) -> dict:
        lang_neuron_path = Path(Path.cwd(), f"outputs/lang_neurons/{self.model_name_srt}/{method}/lang_neuron_data.pkl")
        if lang_neuron_path.exists():
            lang_neuron = pickle.load(open(lang_neuron_path, "rb"))
            print(f"The lang neurons data is loaded from {lang_neuron_path}")
        else:
            raise ValueError(f"{lang_neuron_path} doesn't exist!")
        return lang_neuron
    
    def _get_frozen_neurons(self) -> torch.tensor:
        if self.finetune_lang == "null":
            return None 
        
        lang_neuron = self._get_lang_neuron(method=self.method)
        all_neurons = torch.cartesian_prod(torch.arange(lang_neuron["L"]), torch.arange(lang_neuron["int_d"])).to(self.device) # (4Ld, 2)
        
        if self.finetune_lang == "all_mlp": 
            return all_neurons # !TODO: Caution! This is temporary change to see if MLP training helps
        elif "+" in self.finetune_lang:
            lang1, lang2 = self.finetune_lang.split("+")
            if "set" in lang1:
                lang_set, lang = lang1.split("_")
                lang_neuron1 = self._get_lang_neuron(method=f"lape/{lang_set}")
                ft_index1 = lang_neuron1["lang_to_neuron"][lang] # (N, 2)
            else:
                ft_index1 = lang_neuron["lang_to_neuron"][lang1] # (N, 2)
            ft_index2 = lang_neuron["lang_to_neuron"][lang2] # (N, 2)
            ft_index = torch.cat([ft_index1, ft_index2], dim=0)
        
        elif len(self.finetune_lang) in [2, 7]:
            if "set" in self.finetune_lang:
                lang_set, lang = self.finetune_lang.split("_")
                lang_neuron1 = self._get_lang_neuron(method=f"lape/{lang_set}")
                ft_index = lang_neuron1["lang_to_neuron"][lang] # (N, 2)
            else:
                ft_index = lang_neuron["lang_to_neuron"][self.finetune_lang] # (N, 2)
        else:
            raise ValueError("Invalid finetune lang!")
        
        all_neurons = all_neurons.to(self.device)
        ft_index = ft_index.to(self.device)
        frozen_index = all_neurons[~((all_neurons[:, None] == ft_index).all(-1).any(1))]
        return frozen_index
    
    def _save_config(self) -> None: 
        config_data = {
            "config": self.config,
            "output_dir": self.output_dir,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "train_arg_config": self.train_arg_config,
            "quant_config": self.quant_config,
            "frozen_neurons": self.frozen_neurons
        }
        with open(self.output_dir + "/master_config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
    
    @staticmethod
    def load_model(device: torch.device, config_path: Path, checkpoint_name: str) -> dict:
        with open(config_path, 'rb') as f:
            config_data = pickle.load(f)
        config = config_data["config"]
        model_name = config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ModelForCLSWithLoRA(device=device, tokenizer=tokenizer, model_name=model_name, sparse_alpha=None, num_class=config["num_class"], lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"], quant_config=config_data["quant_config"], frozen_neurons=config_data["frozen_neurons"])
        checkpoint_path = Path(config_path.parent, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return {"model": model, "config_data": config_data}

    def train(self) -> None:
        self._save_config()
        print(self.model)
        self.model.calc_num_lora_params()
        self.trainer.train(resume_from_checkpoint=False)

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name, "task_name": "XNLI-FT",
        "method": "lape/set1", "lang": "en", "finetune_lang": "null", # ["en", "vi", "en+vi", "null", "set1_en"]
        "num_epochs": 2, "num_steps": None, "batch_size": 8, "max_context_length": 256, # steps are auto calculated
        "train_frac": 0.25, "eval_frac": 0.1,
        "initial_lr": 1e-6, "num_class": 3, "lora_rank": 64, "lora_alpha": 128, "max_grad_norm": 10.0, "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999), "grad_acc_steps": 1, "num_ckpt_per_epoch": 4, "is_4bit_quant": True, "fp16": False, "bf16": True,
        "wandb_log": True
    }
    trainer = LoRAFineTuner(device=device, config=config)
    trainer.train()
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["aya23"], device=device)