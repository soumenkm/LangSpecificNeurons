import os, torch, sys, tqdm, pickle, datetime
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

torch.manual_seed(42)
from pathlib import Path
sys.path.append(Path(__file__).parent)
from transformers import BitsAndBytesConfig
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple, Union
import pandas as pd
from activation import NeuronRelevance
from utils import lang_map, models_map
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib_venn import venn3, venn3_circles

class LangNeuron:
    def __init__(self, device: Union[torch.device, None], model_name: str, lang_list: List[str], scoring_method: str):
        self.device = device
        self.model_name = model_name
        self.model_name_srt = self.model_name.split("/")[-1] 
        self.method = scoring_method
        self.cwd = Path.cwd()
        self.lang_neuron_path = Path(self.cwd, f"outputs/lang_neurons/{self.model_name_srt}/{self.method}/lang_neuron_data.pkl")
        self.lang_neuron_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.lang_neuron_path.exists():
            self.__dict__.update(pickle.load(open(self.lang_neuron_path, "rb")))
            print(f"{self.info()}: The lang neurons data is loaded from {self.lang_neuron_path}")
        else:
            self._init_attr(lang_list)
            pickle.dump(self.__dict__, open(self.lang_neuron_path, "wb"))
            print(f"{self.info()}: The lang neurons data is stored at {self.lang_neuron_path}")
    
    def _init_attr(self, lang_list):
        self.lang_list = lang_list
        self.quant_config = None 
        if "lape" in self.method:
            self.lang_to_neuron = self._get_lape_neurons()
        else:
            self.lang_to_neuron = self._get_set_indep_neurons()
    
    def info(self) -> str:
        return f"[INFO] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    def _get_lape_neurons(self) -> dict:
        sum_act_prob = 0
        act_prob_dict = {}
        for lang in self.lang_list:
            rel_obj = NeuronRelevance(model_name=self.model_name, device=self.device, lang=lang, quant_config=self.quant_config, scoring_method="act_prob_zero")
            rel = rel_obj.get_relevance_data(batch_size=None, data_frac=None)
            act_prob_dict[lang] = rel["mean_rel"].to(self.device) # (L, 4d)
            sum_act_prob += act_prob_dict[lang]
        
        norm_act_prob_dict = {}
        for lang, act_prob in act_prob_dict.items():
            norm_act_prob = act_prob/(sum_act_prob + 1e-10)
            norm_act_prob_dict[lang] = norm_act_prob
    
        act_prob = torch.stack(list(act_prob_dict.values()), dim=0) # (k, L, 4d)
        threshold_quantile = 0.95
        threshold = act_prob.flatten().quantile(threshold_quantile).item()
    
        norm_act_prob = torch.stack(list(norm_act_prob_dict.values()), dim=0) # (k, L, 4d)
        act_prob = torch.stack(list(act_prob_dict.values()), dim=0) # (k, L, 4d)
        epsilon = 1e-10
        lape = -1 * (norm_act_prob * torch.log(norm_act_prob + epsilon)).sum(dim=0) # (L, 4d)
        condition = ((act_prob > threshold).sum(dim=0) == 0) # (L, 4d)
        lape[condition] = torch.inf # Ignore those neurons positions where condition is not satisfied

        self.L = norm_act_prob[0].shape[0]
        self.int_d = norm_act_prob[0].shape[1]
        self.m = round(0.01 * self.L * self.int_d)
        
        lape_cutoff = lape.flatten().sort()[0][:self.m].max() # scalar
        lang_neurons = torch.nonzero(lape <= lape_cutoff) # (m, 2)
        
        neuron_to_lang = {}
        for m in range(lang_neurons.shape[0]):
            i, j = lang_neurons[m].tolist()
            lang_spec_list = []
            for lang, act_prob in act_prob_dict.items():
                if act_prob[i, j] > threshold:
                    lang_spec_list.append(lang)
            neuron_to_lang[(i,j)] = lang_spec_list
        
        lang_to_neuron = {lang: [] for lang in self.lang_list}
        for neuron, lang_list in neuron_to_lang.items():
            for lang in lang_list:
                lang_to_neuron[lang].append(list(neuron))
        for lang, neuron_list in lang_to_neuron.items():
            lang_to_neuron[lang] = torch.tensor(neuron_list)
        return lang_to_neuron
           
    def _get_set_indep_neurons(self) -> dict:
        rel_dict = {}
        for lang in self.lang_list:
            rel_obj = NeuronRelevance(model_name=self.model_name, device=self.device, lang=lang, quant_config=self.quant_config, scoring_method=self.method)
            rel = rel_obj.get_relevance_data(batch_size=None, data_frac=None)
            rel_dict[lang] = rel["mean_rel"].to(self.device) # (L, 4d)
        
        self.L = rel_dict[self.lang_list[0]].shape[0]
        self.int_d = rel_dict[self.lang_list[0]].shape[1]
        self.m = round(0.00125 * self.L * self.int_d)
        
        lang_to_neuron = {lang: None for lang in self.lang_list}
        for lang, rel in rel_dict.items():
            _, top_m_indices = torch.topk(rel.flatten(), k=self.m, largest=True)
            layer_indices, neuron_indices = torch.unravel_index(top_m_indices, (self.L, self.int_d))
            neuron = torch.stack((layer_indices, neuron_indices), dim=1) # (m, 2)
            lang_to_neuron[lang] = neuron
        return lang_to_neuron
    
    def get_lang_specific_neurons_dist(self, is_plot: bool) -> dict:
        neuron_dist = {}
        for lang, neuron_tensor in self.lang_to_neuron.items():
            neuron_dist[lang] = neuron_tensor.shape[0]
            
        if is_plot:
            languages = list(neuron_dist.keys())
            num_neurons = list(neuron_dist.values())
            plt.figure(figsize=(len(neuron_dist), 10))
            bars = plt.bar(languages, num_neurons, color='skyblue')
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), ha='center', va='bottom')
            
            plt.xlabel(f'Languages')
            plt.ylabel('Number of Language Specific Neurons')
            title = f'{self.model_name_srt}: Language Specific Neurons Distribution for {self.method}'
            
            plt.title(title, wrap=True)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = Path(self.lang_neuron_path.parent, "lang_neuron_dist.png")
            plt.savefig(str(save_path), format='png', dpi=300)
            plt.clf()
        return neuron_dist
    
    def get_layerwise_neurons_dist(self, is_plot: bool) -> dict:
        layer_neuron_dist = {}
        for lang, neuron_tensor in self.lang_to_neuron.items():
            layer_spec_dist = torch.zeros(size=(self.L,), dtype=torch.int64)
            if len(neuron_tensor) != 0:
                unique_val, counts = neuron_tensor[:, 0].unique(return_counts=True)
                for u, c in zip(unique_val, counts):
                    layer_spec_dist[u] = c
            layer_neuron_dist[lang] = layer_spec_dist
        
        neuron_dist = self.get_lang_specific_neurons_dist(is_plot=False)
        for lang, layer_tensor in layer_neuron_dist.items():
            assert layer_tensor.sum() == neuron_dist[lang], "Layerwise neurons count should match overall count!"

        if is_plot:
            df = pd.DataFrame(layer_neuron_dist, index=[f'{i}' for i in range(self.L)])
            df = df.iloc[::-1] # reverses the order of layers so that bottom (0) appears at bottom of matrix
            plt.figure(figsize=(len(layer_neuron_dist), 10))
            sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu", cbar=True, linewidths=.5)
            plt.xlabel(f'Languages')
            plt.ylabel('Layer Indices (0: bottom)')
            title = f'{self.model_name_srt}: Layerwise Neurons Distribution for {self.method}'
            
            plt.title(title, wrap=True)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            save_path = Path(self.lang_neuron_path.parent, "layerwise_lang_neurons_dist.png")
            plt.savefig(str(save_path), format='png', dpi=300)
            plt.clf()
        return layer_neuron_dist
    
    def get_neurons_overlap(self, is_plot: bool) -> dict:
        keys = list(self.lang_to_neuron.keys())
        matrix = pd.DataFrame(np.zeros((len(keys), len(keys)), dtype=np.int64), index=keys, columns=keys)
        for i, key1 in enumerate(keys):
            for j, key2 in enumerate(keys):
                if i <= j:  
                    data1 = {tuple(i) for i in self.lang_to_neuron[key1].tolist()}
                    data2 = {tuple(i) for i in self.lang_to_neuron[key2].tolist()}
                    common_elements = data1 & data2
                    count_common = len(common_elements)
                    matrix.at[key1, key2] = count_common
                    matrix.at[key2, key1] = count_common 
        
        neurons_dist = self.get_lang_specific_neurons_dist(is_plot=False)
        matrix_dict = matrix.to_dict()
        for key, val in matrix_dict.items():
            assert neurons_dist[key] == val[key], "Diagonal should match with number of lang specific neurons!"

        if is_plot:
            plt.figure(figsize=(len(self.lang_list), len(self.lang_list)))
            sns.heatmap(matrix, annot=True, cmap="YlGnBu", fmt="d", linewidths=.5)
            plt.xlabel(f'Languages')
            plt.ylabel(f'Languages')
            title = f'{self.model_name_srt}: Neurons Overlap Between Languages for {self.method}'
            
            plt.title(title, wrap=True)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=45, ha='right')
            plt.tight_layout()
            save_path = Path(self.lang_neuron_path.parent, 'lang_neurons_overlap.png')
            plt.savefig(str(save_path), format='png', dpi=300)
            plt.clf()
        return matrix.to_dict()
    
    def plot_3_lang_overlap_venn(self, languages: List[str]) -> None:
        assert len(languages) <= 3, "Venn diagrams only support up to 3 sets (languages)."
        for lang in languages:
            assert lang in self.lang_to_neuron, f"Language {lang} not found in lang neuron set."
            
        data = {}
        for lang in languages:
            data[lang] = {tuple(i) for i in self.lang_to_neuron[lang].tolist()}
        set1, set2, set3 = data[languages[0]], data[languages[1]], data[languages[2]]
        venn = venn3([set1, set2, set3], set_labels=languages)
        venn3_circles([set1, set2, set3])
        title = f'{self.model_name_srt}: Neurons Overlap Between Languages ({languages}) for {self.method}'
            
        plt.title(title, wrap=True)
        save_path = Path(self.lang_neuron_path.parent, f'neuron_overlap_{"_".join(languages)}.png')
        plt.savefig(str(save_path), format='png', dpi=300)
        plt.clf()

def main(model_name: str, device: torch.device) -> None:
    methods = ["act_prob_zero", "act_abs_mean", "act_prob_mean", "act_prob_95p", "lape/set1"]
    lang_neuron = LangNeuron(device=device, model_name=model_name, lang_list=lang_map["set1"], scoring_method="act_prob_zero")
    lang_neuron.get_lang_specific_neurons_dist(is_plot=True)
    lang_neuron.get_layerwise_neurons_dist(is_plot=True)
    lang_neuron.get_neurons_overlap(is_plot=True)
    lang_neuron.plot_3_lang_overlap_venn(languages=["en", "vi", "id"])
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}...")
    
    main(model_name=models_map["llama3"], device=device)