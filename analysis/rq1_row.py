import json

MODELS = [
    ("resnet50", "ImageNet"),
    ("mobilevit", "ImageNet"),
    ("mitast", "speech_commands"),
    ("wav2vec2kws", "speech_commands"),
]

MODEL_PATH = None
TARGET_LABEL = None
SEEDS = [0, 1, 2]
N = 24
NoGrad = False # True if mutation is not gradient-guided

def generate_coverage_rows(json_path, batch_size, random_mutation=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    new_row = []

    for model_name, dataset_name in MODELS:
        new_cov = []
        faults = []

        for seed in SEEDS:
            key = f"{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            # key = f"yuan-{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            if random_mutation:
                key = key + "-rand"
            
            entry = data.get(key, {})
            faults.append(entry.get("generated_count", 0))

            cov = entry.get("coverage", {})
            new_cov.append(cov.get("new", 0))
         

        cov_avg = sum(new_cov) / 3
        new_row.append(f"{cov_avg:.2f}")

        fault_avg = sum(faults) / 3
        new_row.append(f"{fault_avg:.0f}")
        


    return new_row

if __name__ == "__main__":
    json_path = "results.json"   # path to your JSON file
    batch_size = N              # set batch size here 

    new_row = generate_coverage_rows(
        json_path=json_path,
        batch_size=batch_size,
        random_mutation=NoGrad
    )

    print( " & ".join(new_row))

