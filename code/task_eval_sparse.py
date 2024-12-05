import os, wandb, torch, tqdm, sys, json, math, gc, pickle, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.manual_seed(42)   
from pathlib import Path
sys.path.append(Path(__file__).parent)
from typing import List, Tuple, Union, Any
from dataset import XNLIDatasetHF
from utils import models_map
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from lora_cls_finetune_sparse import LoRAFineTuner

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
        self.eval_lang = self.config["eval_lang"]
        
        self.eval_path = Path(Path.cwd(), f"outputs/task_eval/{self.model_name_srt}_finetune_{self.task_name}/{self.config_data['run_name']}_eval_{self.eval_lang}_result.txt")
        if not self.eval_path.parent.exists():
            Path.mkdir(self.eval_path.parent, parents=True, exist_ok=True)
   
    def _forward_batch(self, batch: dict, intervene_config: Union[dict, None]) -> torch.tensor:
        input_ids = batch["input_ids"].to(self.device) # (b, T)
        attention_mask = batch["attention_mask"].to(self.device) # (b, T)
        self.model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, intervene_config=intervene_config)["logits"]
        return out # (b, c)
    
    def _calc_acc_batch(self, pred_outputs: torch.tensor, true_outputs: torch.tensor) -> torch.tensor:
        pred_outputs = pred_outputs.to(self.device)
        true_outputs = true_outputs.to(self.device)
        assert pred_outputs.dim() == 2, f"pred_outputs.shape = {pred_outputs.shape} must be (b, c)"
        assert true_outputs.dim() == 1, f"true_outputs.shape = {true_outputs.shape} must be (b,)"
        acc = (pred_outputs.argmax(dim=-1) == true_outputs).to(torch.float32).mean()
        return torch.tensor(acc.item()) # returns the tensor as a scalar number 
    
    def _evaluate_dataloader(self, intervene_config: Union[dict, None]) -> float:
        with tqdm.tqdm(iterable=self.eval_dl, desc=f"[EVAL] on lang {self.eval_lang}", total=len(self.eval_dl), unit="batch", colour="green") as pbar:
            acc_list = []
            for i, batch in enumerate(pbar):   
                pred_out = self._forward_batch(batch=batch, intervene_config=intervene_config) # (b, c)  
                true_out = batch["labels"] # (b,)   
                acc = self._calc_acc_batch(pred_outputs=pred_out, true_outputs=true_out)
                acc_list.append(acc)
                pbar.set_postfix({"acc": f"{acc:.3f}"})  
        eval_acc = sum(acc_list)/len(acc_list)
        return float(eval_acc)
    
    def evaluate(self) -> None:
        if self.eval_path.exists():
            print(f"The result already exists: {self.eval_path}")
            return None
        
        lang = self.eval_lang
        self.eval_ds = XNLIDatasetHF(model_name=self.model_name, lang=lang, max_context_len=self.config_data["config"]["max_context_length"], frac=self.config["eval_frac"], is_train=False)
        self.eval_dl = DataLoader(self.eval_ds, batch_size=self.config["batch_size"], shuffle=False, drop_last=True)
        acc = self._evaluate_dataloader(intervene_config=None)
        res1 = f"[RESULT] Train lang: {self.train_lang}, Eval lang: {self.eval_lang}, Zero shot acc: {acc}"
        with open(self.eval_path, "w") as f:
            f.writelines("\n".join([res1]))
              
def main(config: dict, device: torch.device) -> None:
    evaluator = Evaluator(device=device, config=config)
    evaluator.evaluate()
    print("DONE")
    
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Evaluation script for language model")
    # parser.add_argument("--method", type=str, required=True, help="Method")
    # parser.add_argument("--ckpt_path", type=str, required=True, help="Checkpoint path")
    # parser.add_argument("--ckpt_id", type=str, required=True, help="Checkpoint id")
    # parser.add_argument("--eval_lang", type=str, required=True, help="Language for evaluation")
    # parser.add_argument("--is_zero_shot", type=int, required=True, help="Whether the evaluation is zero-shot")
    # parser.add_argument("--intervene_by", type=str, required=True, help="Type of intervention - [mean_p95_act, mean_p90_act, mean_p75_act, mean_mu_act]")
    # args = parser.parse_args()
    
    # config = {
    #     "config_path": Path(Path.cwd(), f"{args.ckpt_path}/master_config.pkl"),
    #     "method": args.method,
    #     "ckpt_name": f"checkpoint-{args.ckpt_id}/pytorch_model.bin",
    #     "eval_lang": args.eval_lang,
    #     "batch_size": 8,
    #     "eval_frac": 1.0,
    #     "is_zero_shot": bool(args.is_zero_shot),
    #     "intervene_by": args.intervene_by
    # }
    config = {
        "config_path": Path(Path.cwd(), f"outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI-FT-Sparse/en_finetune_sa_5.0e-09_0.25_1.0e-05_r8/master_config.pkl"),
        "ckpt_name": f"checkpoint-12268/pytorch_model.bin",
        "eval_lang": "vi",
        "batch_size": 8,
        "eval_frac": 1.0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    main(config=config, device=device)
        

