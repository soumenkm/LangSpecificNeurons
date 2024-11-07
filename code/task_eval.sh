CUDA_VISIBLE_DEVICES=0, nohup python code/task_eval.py --ckpt_path "outputs/ckpt/Meta-Llama-3.1-8B_finetune_XNLI-DGX4/lape/set1/en_finetune_null_0.25_1.0e-05_r8" --ckpt_id "12268" --eval_lang "vi" --is_zero_shot 1 --intervene_by "mean_mu_act" > eval1.out & 

