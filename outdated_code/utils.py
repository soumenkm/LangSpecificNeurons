lang_map = {
    "set1": ["en", "fr", "es", "vi", "id", "ja", "zh"],
    "set2": ["en", "fr", "vi", "zh", "bn", "hi", "te", "mr"],
    "set3": ["bn", "hi", "ta", "te", "mr", "ur", "kn", "ml", "pa"],
    "set4": ["en", "fr", "es", "vi", "id", "ja", "zh", "bn", "hi", "te"],
    "set5": ["en", "fr", "es", "vi", "id", "ja", "zh", "bn", "hi", "ta", "te", "mr", "ur", "kn", "ml", "pa"],
    "set6": ["en", "bn", "hi", "ta", "te", "mr", "ur", "kn", "ml", "pa"]
}

lang_triplet_map = {
    "set1": [["en", "fr", "es"], ["fr", "vi", "zh"], ["vi", "ja", "zh"]],
    "set2": [["fr", "vi", "zh"], ["bn", "hi", "mr"], ["bn", "hi", "te"], ["fr", "bn", "te"]],
    "set3": [["bn", "hi", "mr"], ["bn", "hi", "ur"], ["te", "kn", "ml"], ["hi", "pa", "mr"], ["ta", "te", "ml"], ["ta", "ml", "kn"], ["hi", "ta", "te"], ["bn", "te", "pa"]],
    "set4": [["bn", "fr", "ja"], ["fr", "vi", "zh"], ["vi", "te", "es"], ["bn", "hi", "te"], ["en", "hi", "zh"]],
    "set6": [["bn", "hi", "mr"], ["bn", "hi", "ur"], ["te", "kn", "ml"], ["hi", "pa", "mr"], ["ta", "te", "ml"], ["ta", "ml", "kn"], ["hi", "ta", "te"], ["bn", "te", "pa"]],
}

models_map = {
    "llama2": "meta-llama/Llama-2-7b-hf", # Done
    "llama3": "meta-llama/Meta-Llama-3.1-8B", # Done
    "mistral-nemo": "mistralai/Mistral-Nemo-Base-2407", # Done
    "bloomz": "bigscience/bloomz-7b1", # Done
    "sarvam": "sarvamai/sarvam-2b-v0.5", # Done
    "aya23": "CohereForAI/aya-23-8B",
    "aya101": "CohereForAI/aya-101"
}

token_repr_map = {
    "Llama-2-7b-hf": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "75", "hi": "75", "ta": "75", "te": "75", "mr": "75", "ur": "75", "kn": "75", "ml": "75", "pa": "75"},
    "Meta-Llama-3.1-8B": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "75", "hi": "75", "ta": "75", "te": "75", "mr": "37", "ur": "75", "kn": "75", "ml": "75", "pa": "75"},
    "bloomz-7b1": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "67", "hi": "52", "ta": "52", "te": "52", "mr": "18", "ur": "45", "kn": "30", "ml": "37", "pa": "18"},
    "sarvam-2b-v0.5": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "75", "hi": "37", "ta": "67", "te": "67", "mr": "22", "ur": "75", "kn": "37", "ml": "45", "pa": "22"},
    "aya-23-8B": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "75", "hi": "75", "ta": "60", "te": "75", "mr": "30", "ur": "60", "kn": "41", "ml": "45", "pa": "30"},
    "Mistral-Nemo-Base-2407": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "75", "hi": "75", "ta": "75", "te": "75", "mr": "37", "ur": "60", "kn": "55", "ml": "67", "pa": "37"},
    "aya-101": {"en": "75", "fr": "75", "es": "75", "vi": "75", "id": "75", "ja": "75", "zh": "75", "bn": "75", "hi": "75", "ta": "60", "te": "75", "mr": "30", "ur": "60", "kn": "41", "ml": "45", "pa": "30"},
}

lang_repr_map = {
    "Llama-2-7b-hf": {"en": "HIGH", "fr": "HIGH", "es": "HIGH", "vi": "HIGH", "id": "HIGH", "ja": "HIGH", "zh": "HIGH", "bn": "LOW", "hi": "LOW", "ta": "LOW", "te": "LOW", "mr": "LOW", "ur": "LOW", "kn": "LOW", "ml": "LOW", "pa": "LOW"},
    "Meta-Llama-3.1-8B": {"en": "HIGH", "fr": "HIGH", "es": "HIGH", "vi": "LOW", "id": "LOW", "ja": "LOW", "zh": "LOW", "bn": "LOW", "hi": "HIGH", "ta": "LOW", "te": "LOW", "mr": "LOW", "ur": "LOW", "kn": "LOW", "ml": "LOW", "pa": "LOW"},
    "bloomz-7b1": {"en": "HIGH", "fr": "HIGH", "es": "HIGH", "vi": "HIGH", "id": "HIGH", "ja": "LOW", "zh": "HIGH", "bn": "HIGH", "hi": "HIGH", "ta": "HIGH", "te": "HIGH", "mr": "HIGH", "ur": "HIGH", "kn": "HIGH", "ml": "HIGH", "pa": "HIGH"},
    "sarvam-2b-v0.5": {"en": "HIGH", "fr": "LOW", "es": "LOW", "vi": "LOW", "id": "LOW", "ja": "LOW", "zh": "LOW", "bn": "HIGH", "hi": "HIGH", "ta": "HIGH", "te": "HIGH", "mr": "HIGH", "ur": "LOW", "kn": "HIGH", "ml": "HIGH", "pa": "HIGH"},
    "aya-23-8B": {"en": "HIGH", "fr": "HIGH", "es": "HIGH", "vi": "HIGH", "id": "HIGH", "ja": "HIGH", "zh": "HIGH", "bn": "LOW", "hi": "HIGH", "ta": "LOW", "te": "LOW", "mr": "LOW", "ur": "LOW", "kn": "LOW", "ml": "LOW", "pa": "LOW"},
    "Mistral-Nemo-Base-2407": {"en": "HIGH", "fr": "HIGH", "es": "HIGH", "vi": "HIGH", "id": "HIGH", "ja": "HIGH", "zh": "HIGH", "bn": "HIGH", "hi": "HIGH", "ta": "HIGH", "te": "HIGH", "mr": "HIGH", "ur": "HIGH", "kn": "HIGH", "ml": "HIGH", "pa": "HIGH"},
    "aya-101": {"en": "HIGH", "fr": "HIGH", "es": "HIGH", "vi": "HIGH", "id": "HIGH", "ja": "HIGH", "zh": "HIGH", "bn": "LOW", "hi": "HIGH", "ta": "LOW", "te": "LOW", "mr": "LOW", "ur": "LOW", "kn": "LOW", "ml": "LOW", "pa": "LOW"},

}