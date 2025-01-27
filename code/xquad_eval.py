import os, wandb, torch, tqdm, sys, json, math, gc, pickle, argparse
# if __name__ == "__main__":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "7"
torch.manual_seed(42)   
from pathlib import Path
sys.path.append(Path(__file__).parent)
from typing import List, Tuple, Union, Any
from dataset import XQADatasetHF
from utils import models_map
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from lora_instruct_finetune import LoRAFineTuner
from collections import Counter

class Evaluator:
    def __init__(self, device: torch.device, config: dict):
        self.device = device
        self.config = config
        self.config_path = self.config["config_path"]
        out = LoRAFineTuner.load_model(config_path=self.config_path, checkpoint_name=self.config["ckpt_name"], device=self.device)
        self.model = out["model"]
        self.config_data = out["config_data"]
        
        self.model_name = self.config_data["config"]["model_name"]
        self.model_name_srt = self.model_name.split("/")[-1]
        self.task_name = self.config_data["config"]["task_name"]
        self.train_lang = self.config_data["config"]["lang"]
        self.finetune_lang = self.config_data["config"]["finetune_lang"]
        self.method = self.config["method"] #self.config_data["config"]["method"]
        self.eval_lang = self.config["eval_lang"]
        self.int_by = self.config["intervene_by"]
        
        self.eval_path = Path(Path.cwd(), f"outputs/task_eval/{self.model_name_srt}_finetune_{self.task_name}/{self.method}/{self.int_by}/{self.config['ckpt_name'].split('/')[0]}_train_{self.train_lang}_finetune_{self.finetune_lang}_eval_{self.eval_lang}_result.txt")
        if not self.eval_path.parent.exists():
            Path.mkdir(self.eval_path.parent, parents=True, exist_ok=True)
        
        self.Tmax = self.config_data["config"]["max_context_length"]
        self.eval_ds = XQADatasetHF(model_name=self.model_name, lang=self.eval_lang, max_context_len=self.Tmax, frac=self.config["eval_frac"], is_train=False)
        self.eval_dl = DataLoader(self.eval_ds, batch_size=self.config["batch_size"], shuffle=False, drop_last=True)
        self.tokenizer = self.eval_ds.tokenizer
        
    def _get_intervene_config(self, intervene_lang: str, is_activate: bool) -> dict:
        """intervene_lang = yy"""
        lang = intervene_lang
        lang_neuron_path = Path(Path.cwd(), f"outputs/lang_neurons/{self.model_name_srt}/{self.method}/lang_neuron_data.pkl")
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
        
        index = lang_neuron["lang_to_neuron"][lang].to(self.device) # (N, 2)
        if self.int_by == "zero":
            intervene_config = {
                "indices": index,
                "value": torch.zeros(size=(index.shape[0],))
            }
        elif self.int_by == "neg1":
            intervene_config = {
                "indices": index,
                "value": torch.ones(size=(index.shape[0],)) * (-1)
            }
        elif self.int_by == "neg10":
            intervene_config = {
                "indices": index,
                "value": torch.ones(size=(index.shape[0],)) * (-10)
            }
        elif self.int_by == "pos1":
            intervene_config = {
                "indices": index,
                "value": torch.ones(size=(index.shape[0],)) * (1)
            }
        elif self.int_by == "pos10":
            intervene_config = {
                "indices": index,
                "value": torch.ones(size=(index.shape[0],)) * (10)
            }
        else:
            mean_act = act_data[self.int_by].to(self.device) # (L, 4d)
            value = mean_act[index[:, 0], index[:, 1]] # (N,)
            intervene_config = {
                "indices": index,
                "value": value if is_activate else torch.zeros_like(value)
            }
        return intervene_config
   
    def _forward_batch(self, batch: dict, intervene_config: Union[dict, None]) -> torch.tensor:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=intervene_config)["logits"]
        return out # (b, T, V)
    
    def _generate_batch(self, batch: dict, intervene_config: Union[dict, None]) -> dict:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        answer_ids = batch["labels"].to(self.device) # (b, T)
        assert input_ids.shape[0] == 1, "Batch size should be 1"
        
        pred_token_list = []
        for i in range(100):
            out = self._forward_batch(batch=batch, intervene_config=intervene_config) # (b, T, V)
            pred_token_id = int(out[0, -1, :].argmax().item()) # scalar 
            input_ids = torch.concat([input_ids.squeeze(), torch.tensor([pred_token_id], device=self.device)], dim=0).unsqueeze(dim=0)
            attention_mask = torch.concat([attention_mask.squeeze(), torch.tensor([1], device=self.device)], dim=0).unsqueeze(dim=0)
            input_ids = input_ids[:, -self.Tmax:]
            attention_mask = attention_mask[:, -self.Tmax:]
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
            pred_token_list.append(pred_token_id)
            if pred_token_id == self.tokenizer.eos_token_id:
                break
        
        true_token_list = answer_ids[0].tolist()
        return {"true_tokens": true_token_list, "pred_tokens": pred_token_list}
    
    def _calc_metrics_batch(self, true_tokens: List[int], pred_tokens: List[int]) -> dict:
        # Exact Match: True if the sequences are identical
        exact_match = int(true_tokens == pred_tokens)
        
        # Count occurrences of tokens
        true_counter = Counter(true_tokens)
        pred_counter = Counter(pred_tokens)
        
        # Calculate intersection count (min of counts for each token)
        common_tokens = sum((true_counter & pred_counter).values())
        
        # Precision: |True ∩ Pred| / |Pred|
        precision = common_tokens / len(pred_tokens) if pred_tokens else 0.0
        
        # Recall: |True ∩ Pred| / |True|
        recall = common_tokens / len(true_tokens) if true_tokens else 0.0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
        
        return {
            "exact_match": exact_match,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
        
    def _compute_average_metrics(self, metrics_list: List[dict]) -> dict:
        # Initialize an empty dictionary to store the sum of each metric
        total_metrics = {key: 0.0 for key in metrics_list[0]}
        
        # Sum up the metrics for all examples
        for metrics in metrics_list:
            for key, value in metrics.items():
                total_metrics[key] += value
        
        # Divide by the number of examples to get the average
        num_examples = len(metrics_list)
        avg_metrics = {key: total / num_examples for key, total in total_metrics.items()}
        return avg_metrics
    
    def _evaluate_dataloader(self, intervene_config: Union[dict, None]) -> dict:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[EVAL] on lang {self.eval_lang}", total=len(self.eval_dl), unit="batch", colour="green") as pbar:
            metrics_list = []
            for i, batch in enumerate(pbar):   
                pred_out = self._generate_batch(batch=batch, intervene_config=intervene_config) # (b, T, V)     
                metrics = self._calc_metrics_batch(true_tokens=pred_out["true_tokens"], pred_tokens=pred_out["pred_tokens"])
                metrics_list.append(metrics)
                print("True: ", self.tokenizer.decode(pred_out["true_tokens"])) 
                print("Pred: ", self.tokenizer.decode(pred_out["pred_tokens"]))
                print(metrics)
        
        return self._compute_average_metrics(metrics_list=metrics_list)
    
    def evaluate(self) -> None:
        if self.eval_path.exists():
            print(f"The result already exists: {self.eval_path}")
            return None
        
        lang = self.eval_lang
        intervene_config = self._get_intervene_config(intervene_lang=self.eval_lang, is_activate=True)

        if self.config["is_zero_shot"]:
            metrics = self._evaluate_dataloader(intervene_config=None)
            if self.train_lang == lang:
                res1 = f"[RESULT] Train lang: {self.train_lang}, Finetune lang: {self.finetune_lang}, Eval lang: {self.eval_lang}, Direct acc: {metrics}"
            else:
                res1 = f"[RESULT] Train lang: {self.train_lang}, Finetune lang: {self.finetune_lang}, Eval lang: {self.eval_lang}, Zero shot acc: {metrics}"
        else:
            res1 = f"[RESULT] Train lang: {self.train_lang}, Finetune lang: {self.finetune_lang}, Eval lang: {self.eval_lang}, Zero shot acc: NOT CALCULATED"                
            
        print(res1)
        int_metrics = self._evaluate_dataloader(intervene_config=intervene_config)
        res2 = f"[RESULT] Train lang: {self.train_lang}, Finetune lang: {self.finetune_lang}, Eval lang: {self.eval_lang}, Intervene acc: {int_metrics}"
        print(res2)
        
        with open(self.eval_path, "w") as f:
            f.writelines("\n".join([res1, res2]))
              
def main(config: dict, device: torch.device) -> None:
    evaluator = Evaluator(device=device, config=config)
    evaluator.evaluate()
    print("DONE")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for language model")
    parser.add_argument("--method", type=str, required=True, help="Method")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--ckpt_id", type=str, required=True, help="Checkpoint id")
    parser.add_argument("--eval_lang", type=str, required=True, help="Language for evaluation")
    parser.add_argument("--is_zero_shot", type=int, required=True, help="Whether the evaluation is zero-shot")
    parser.add_argument("--intervene_by", type=str, required=True, help="Type of intervention - [mean_p95_act, mean_p90_act, mean_p75_act, mean_mu_act]")
    args = parser.parse_args()
    
    config = {
        "config_path": Path(Path.cwd(), f"{args.ckpt_path}/master_config.pkl"),
        "method": args.method,
        "ckpt_name": f"checkpoint-{args.ckpt_id}/pytorch_model.bin",
        "eval_lang": args.eval_lang,
        "batch_size": 1,
        "eval_frac": 1.0,
        "is_zero_shot": bool(args.is_zero_shot),
        "intervene_by": args.intervene_by
    }
    # ckpt_path = "/raid/speech/soumen/MS_Research/LangSpecificNeurons/outputs/ckpt/Meta-Llama-3.1-8B_finetune_XQUAD-FT/lape/set1/en_finetune_null_1.00_5.0e-05_r64"
    # ckpt_id = 1160
    # eval_lang = "vi"
    # is_zero_shot = False
    # intervene_by = "mean_p5_act"
    # method = "lape/set1"
    # config = {
    #     "config_path": Path(Path.cwd(), f"{ckpt_path}/master_config.pkl"),
    #     "method": method,
    #     "ckpt_name": f"checkpoint-{ckpt_id}/pytorch_model.bin",
    #     "eval_lang": eval_lang,
    #     "batch_size": 1,
    #     "eval_frac": 0.1,
    #     "is_zero_shot": is_zero_shot,
    #     "intervene_by": intervene_by
    # }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    main(config=config, device=device)
        

