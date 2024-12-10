import os, json, pickle, torch, wandb, tqdm
if __name__ == "__main__":
    wandb.login()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    
torch.manual_seed(42)
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from dataset import XNLIDatasetHF
from utils import models_map
from models import ModelForCLSWithLoRA, ModelForCLS
import evaluate, torch
import numpy as np
from typing import List, Tuple, Union, Any
from torch.utils.data import DataLoader

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
        self.train_ds = XNLIDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"], frac=self.config["train_frac"], is_train=True)
        self.eval_ds = XNLIDatasetHF(model_name=self.model_name, lang=self.lang, max_context_len=self.config["max_context_length"], frac=self.config["eval_frac"], is_train=False)
        self.tokenizer = self.train_ds.tokenizer

        self.batch_size = self.config["batch_size"]
        self.config["num_steps"] = len(self.train_ds)//self.batch_size
        self.num_steps = self.config["num_steps"]
        self.num_epochs = self.config["num_epochs"]
        self.wandb_log = self.config["wandb_log"]
        self.project_name = f"{self.model_name.split('/')[-1]}_finetune_{self.config['task_name']}"
        os.environ["WANDB_PROJECT"] = self.project_name
        self.run_name = f"{self.method}/{self.lang}_finetune_{self.config['train_frac']:.2f}_{self.config['initial_lr']:.1e}_r{self.config['lora_rank']}"
        self.output_dir = Path(Path.cwd(), f"outputs/ckpt/{self.project_name}/{self.run_name}")
        if not Path.exists(self.output_dir):
            Path.mkdir(self.output_dir, parents=True, exist_ok=True)

        self.L = 32
        self.int_d = 14336
        self.frozen_neurons, self.apply_lora_mlp = self._get_frozen_neurons()
        self.intervene_config = self._get_intervene_config(intervene_lang=self.config["intervene_lang"],
                                                           lang_neuron_method=self.config["lang_neuron_method"],
                                                           int_by=self.config["int_by"],
                                                           is_activate=True)
        if self.method == "only_cls":
            self.model = ModelForCLS(device=self.device, model_name=self.model_name, num_class=self.config["num_class"], quant_config=self.quant_config)
        else:
            self.model = ModelForCLSWithLoRA(device=self.device, tokenizer=self.tokenizer, model_name=self.model_name, sparse_alpha=None, num_class=self.config["num_class"], lora_rank=self.config["lora_rank"], lora_alpha=self.config["lora_alpha"], quant_config=self.quant_config, frozen_neurons=self.frozen_neurons, apply_lora_mlp=self.apply_lora_mlp)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.config['initial_lr'], weight_decay=self.config["weight_decay"], betas=self.config["adam_betas"])
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                         num_warmup_steps=int(0.01 * self.num_steps), 
                                                         num_training_steps=int(self.num_epochs * self.num_steps))
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.eval_dl = DataLoader(self.eval_ds, batch_size=self.batch_size, shuffle=False, drop_last=True)
        self.train_arg_config = {
            "output_dir": self.output_dir,
            "eval_strategy": "steps",
            "eval_steps": max(self.batch_size, self.num_steps//self.config["num_ckpt_per_epoch"]),
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
        if self.wandb_log:
            wandb.init(project=self.project_name, name=self.run_name, config=self.config)
            wandb.watch(self.model, log="all")
            wandb.define_metric("train/step")
            wandb.define_metric("val/step")
            wandb.define_metric("train/*", step_metric="train/step")
            wandb.define_metric("val/*", step_metric="val/step")
    
    def _get_frozen_neurons(self) -> torch.tensor:
        if "apply_lora_mlp" in self.method:
            frozen_neurons = "all"
            apply_lora_mlp = True
        elif "no_lora_mlp" in self.method:
            frozen_neurons = None
            apply_lora_mlp = False
        elif "only_cls" in self.method:
            frozen_neurons = None
            apply_lora_mlp = None
        elif "random_lora_mlp" in self.method:
            frozen_neurons = [(i,j) for i in range(self.L) for j in torch.randint(low=0, high=self.int_d, size=(10,))]
            frozen_neurons = torch.tensor(frozen_neurons)
            apply_lora_mlp = True
        return frozen_neurons, apply_lora_mlp
    
    def _get_intervene_config(self, intervene_lang: str, lang_neuron_method: str, int_by: str, is_activate: bool) -> dict:
        """intervene_lang = yy,
        lang_neuron_method = lape/set1, act_prob_95p,
        int_by = mean_p95_act, mean_p90_act, mean_p75_act, mean_mu_act"""
        if intervene_lang == "null":
            return None
        lang = intervene_lang
        lang_neuron_path = Path(Path.cwd(), f"outputs/lang_neurons/{self.model_name_srt}/{lang_neuron_method}/lang_neuron_data.pkl")
        if lang_neuron_path.exists():
            lang_neuron = pickle.load(open(lang_neuron_path, "rb"))
            print(f"The lang neurons data is loaded from {lang_neuron_path}")
        else:
            raise ValueError(f"{lang_neuron_path} doesn't exist!")

        act_data_path = Path(Path.cwd(), f"outputs/activation/{self.model_name_srt}/act_stat/rel_{lang}.pkl")
        if act_data_path.exists():
            act_data = pickle.load(open(act_data_path, "rb"))
            print(f"The activation data is loaded from {act_data_path}")
        else:
            raise ValueError(f"{act_data_path} doesn't exist!")
        
        mean_act = act_data[int_by].to(self.device) # (L, 4d)
        index = lang_neuron["lang_to_neuron"][lang].to(self.device) # (N, 2)
        value = mean_act[index[:, 0], index[:, 1]] # (N,)
        intervene_config = {
            "indices": index,
            "value": value if is_activate else torch.zeros_like(value)
        }
        return intervene_config
    
    def _find_norm(self) -> float:
        norm = 0
        for val in self.model.parameters():
            if val.requires_grad:
                k = val.grad if val.grad is not None else torch.tensor(0.0, device=self.device)
                norm += (k ** 2).sum().item()
        norm = norm ** 0.5  
        return norm
    
    def _save_checkpoint(self, ep: int) -> None:
        checkpoint = {"epoch": ep, 
                      "model_state": self.model.state_dict(), 
                      "opt_state": self.optimizer.state_dict(),
                      "config": self.config}   
        checkpoint_path = Path(self.output_dir, f"ckpt_ep_{ep}.pth")            
        torch.save(checkpoint, checkpoint_path)
        print(f"[SAVE] ep: {ep}/{self.num_epochs-1}, checkpoint saved at: {checkpoint_path}")
    
    def _forward_batch(self, batch: dict, is_train: bool) -> dict:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        labels = batch["labels"] # (b,)
        if is_train:
            self.model.train()
            with torch.amp.autocast(device_type="cuda"):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=self.intervene_config, labels=labels)
            out["logits"].requires_grad_(True)
            assert out["logits"].requires_grad == True
        else:
            self.model.eval()
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=self.intervene_config, labels=labels)       
        return out # (b, c)
        
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        return torch.tensor(acc.item()) # returns the tensor as a scalar number

    def _optimize_batch(self, loss: torch.tensor, ep: int, batch_index: int) -> None:  
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()     
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config["max_grad_norm"], norm_type=2.0)
        self.optimizer.step()
        self.scheduler.step()
    
    def _optimize_dataloader(self, ep: int) -> None:  
        with tqdm.tqdm(iterable=self.train_dl, desc=f"[TRAIN] ep: {ep}/{self.num_epochs-1}", total=len(self.train_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                out = self._forward_batch(batch=batch, is_train=True) # (b, c)  
                loss = out["loss"]
                pred_out = out["logits"] # (b, c)
                true_out = batch["labels"] # (b,) 
                gn = self._find_norm()
                lr = self.optimizer.param_groups[0]['lr']   
                self._optimize_batch(loss=loss, ep=ep, batch_index=i)
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                
                if self.wandb_log:
                    wandb.log({"train/loss": loss.item(), "train/accuracy": acc.item(), "train/learning_rate": lr, "train/grad_norm": gn, "train/epoch": ep, "train/step": self.train_step})
                    self.train_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc.item():.3f}", "lr": f"{lr:.3e}", "gn": f"{gn:.3f}"})                        
    
    def _validate_dataloader(self, ep: int) -> None:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[VAL] ep: {ep}/{self.num_epochs-1}", total=len(self.eval_dl), unit="step", colour="green") as pbar:
            for i, batch in enumerate(pbar):    
                out = self._forward_batch(batch=batch, is_train=False) # (b, c)  
                loss = out["loss"]
                pred_out = out["logits"] # (b, c)
                true_out = batch["labels"] # (b,)   
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                
                if self.wandb_log:
                    wandb.log({"val/loss": loss.item(), "val/accuracy": acc.item(), "val/epoch": ep, "val/step": self.val_step})
                    self.val_step += 1
                pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{acc.item():.3f}"})                        
    
    def _save_config(self) -> None: 
        config_data = {
            "config": self.config,
            "output_dir": self.output_dir,
            "project_name": self.project_name,
            "run_name": self.run_name,
            "train_arg_config": self.train_arg_config,
            "quant_config": self.quant_config,
            "frozen_neurons": self.frozen_neurons,
            "apply_lora_mlp": self.apply_lora_mlp
        }
        with open(Path(self.output_dir, "master_config.pkl"), 'wb') as f:
            pickle.dump(config_data, f)
    
    def train(self) -> None:
        self._save_config()
        print(self.model)
        self.model.calc_num_params()
        self.train_step = 0
        self.val_step = 0
        for ep in range(self.num_epochs):
            self._optimize_dataloader(ep=ep)
            self._validate_dataloader(ep=ep)
            self._save_checkpoint(ep=ep)
        if self.wandb_log:
            wandb.finish()
    
    @staticmethod
    def load_model(device: torch.device, config_path: Path, checkpoint_name: str) -> dict:
        with open(config_path, 'rb') as f:
            config_data = pickle.load(f)
        config = config_data["config"]
        model_name = config["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = ModelForCLSWithLoRA(device=device, tokenizer=tokenizer, model_name=model_name, sparse_alpha=None, num_class=config["num_class"], lora_rank=config["lora_rank"], lora_alpha=config["lora_alpha"], quant_config=config_data["quant_config"], frozen_neurons=config_data["frozen_neurons"], apply_lora_mlp=config_data["apply_lora_mlp"])
        checkpoint_path = Path(config_path.parent, checkpoint_name)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=False)
        
        print(f"Model loaded from checkpoint: {checkpoint_path}")
        return {"model": model, "config_data": config_data}

def main(model_name: str, device: torch.device) -> None:
    config = {
        "model_name": model_name, "task_name": "XNLI-MeanIntFT",
        "method": "no_lora_mlp/mean_int_set1_vi", "lang": "en", # ["apply_lora_mlp", "random_lora_mlp", "no_lora_mlp", "only_cls"]
        "intervene_lang": "vi", "lang_neuron_method": "lape/set1", "int_by": "mean_mu_act", # intervene lang could be "null" also
        "num_epochs": 1, "num_steps": None, "batch_size": 32, "max_context_length": 256, # steps are auto calculated
        "train_frac": 0.25, "eval_frac": 0.1,
        "initial_lr": 1e-5, "num_class": 3, "lora_rank": 8, "lora_alpha": 16, "max_grad_norm": 100.0, "weight_decay": 0.1,
        "adam_betas": (0.95, 0.999), "grad_acc_steps": 1, "num_ckpt_per_epoch": 4, "is_4bit_quant": True, "fp16": False, "bf16": True,
        "wandb_log": True
    }
    trainer = LoRAFineTuner(device=device, config=config)
    trainer.train()
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(models_map["llama3"], device=device)