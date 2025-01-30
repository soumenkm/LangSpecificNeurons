import pickle, torch
from utils import models_map

model_name = models_map["mistral-nemo"].split("/")[-1]
lang_list = ["en", "vi", "hi", "ur", "zh"]
for lang in lang_list:
    filename = f"/raid/speech/soumen/MS_Research/LangSpecificNeurons/outputs/activation/{model_name}/act_stat/rel_{lang}.pkl"

    act = pickle.load(open(filename, "rb"))
    keys = ["mu", "p75", "p90", "p95", "p5", "p10", "p25"]

    string = ""
    for key in keys:
        string = string + " & " + str(round(act[f"mean_{key}_act"].mean().item(), 4))
    print(string)