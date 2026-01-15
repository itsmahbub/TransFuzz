import json

# MODELS = [
#     ("resnet50", "ImageNet", "mobilevit"),
#     ("mobilevit", "ImageNet", "resnet50"),
#     ("mitast", "speech_commands", "wav2vec2kws"),
#     ("wav2vec2kws", "speech_commands", "mitast"),
# ]

MODELS = [
    ("resnet50", "ImageNet", "robustresnet"),
    ("robustresnet", "ImageNet", "resnet50"),
]

model_mapping = {
    "resnet50": "ResNet-50",
    "mobilevit": "MobileViT",
    "mitast": "AST",
    "wav2vec2kws": "Wav2Vec2",
    "robustresnet": "ResNet-50 (AT)"
}

MODEL_PATH = None
TARGET_LABEL = None
SEEDS = [0, 1, 2]
N = 24
NoGrad = False # True if mutation is not gradient-guided

def generate_coverage_rows(json_path, batch_size, random_mutation=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    
    for model_name, dataset_name, attacked_model in MODELS:
        faults = []
        nlcs = []
        mtrs = []
        new_row = [model_mapping[model_name]]
        for seed in SEEDS:
            key = f"{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            # key = f"yuan-{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            if random_mutation:
                key = key + "-rand"
            
            entry = data.get(key, {})
            fault = entry.get("generated_count", 0)
            nlc = entry.get("diversity", {}).get("coverage", 0)
            mtr = entry.get(f"model_transfer-{attacked_model}", {}).get("success_rate", 0)

            faults.append(fault)
            nlcs.append(nlc)
            mtrs.append(mtr)
         

        fault_avg = sum(faults) / 3
        nlc_avg = sum(nlcs) / 3
        mtr_avg  = sum(mtrs) / 3
        new_row.append(f"{fault_avg:.0f}")
        new_row.append(f"{nlc_avg:.2f}")
        new_row.append(f"{mtr_avg:.2f} & $\\rightarrow$ & {model_mapping[attacked_model]} \\\\")
        print( " & ".join(new_row))



if __name__ == "__main__":
    json_path = "results.json"   # path to your JSON file
    batch_size = N              # set batch size here 

    generate_coverage_rows(
        json_path=json_path,
        batch_size=batch_size,
        random_mutation=NoGrad
    )
