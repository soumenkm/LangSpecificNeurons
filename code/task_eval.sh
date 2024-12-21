CUDA_VISIBLE_DEVICES=3, nohup python code/task_eval.py --method "act_prob_90p" --ckpt_path "outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI-TN/act_prob_90p/en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "isi" > isi.out & 
CUDA_VISIBLE_DEVICES=4, nohup python code/task_eval.py --method "act_prob_90p" --ckpt_path "outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI-TN/act_prob_90p/en_finetune_en_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "isi" > isi.out & 
CUDA_VISIBLE_DEVICES=5, nohup python code/task_eval.py --method "act_prob_90p" --ckpt_path "outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI-TN/act_prob_90p/en_finetune_vi_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 0 --intervene_by "isi" > isi.out & 


