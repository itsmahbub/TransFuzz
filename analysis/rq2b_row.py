import json

MODELS = [
    ("mitast", "speech_commands"),
    ("wav2vec2kws", "speech_commands"),
]

MODEL_PATH = None
TARGET_LABEL = None
SEEDS = [0, 1, 2]

def generate_coverage_rows(json_path, batch_size, random_mutation=False):
    with open(json_path, "r") as f:
        data = json.load(f)

    new_row = []

    for model_name, dataset_name in MODELS:
        pesqs = []
        stois = []
        class_coverages = []
        entropies = []
        stabilities = []
        num_classes = 0
        for seed in SEEDS:
            key = f"{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            # key = f"yuan-{model_name}-{MODEL_PATH}-{dataset_name}-{TARGET_LABEL}-{batch_size}-{seed}"
            if random_mutation:
                key = key + "-rand"
            
            entry = data.get(key, {})
            num_classes =entry["number_of_classes"]
            pesqs.append(entry.get("naturalness", {}).get("mean_PESQ", 0))
            stois.append(entry.get("naturalness", {}).get("mean_STOI", 0))
            class_coverages.append(entry.get("diversity", {}).get("class_covered", 0))
            entropies.append(entry.get("diversity", {}).get("scaled_entropy", 0))
            stabilities.append(entry.get("stability", {}).get("preserved_adversarial_rate", 0))

        pesq = sum(pesqs)/3
        stoi = sum(stois)/3
        cc = sum(class_coverages)/ 3
        e = sum(entropies)/3
        s = sum(stabilities)/3

        new_row.append(f"{pesq:.3f}")
        new_row.append(f"{stoi:.3f}")
        new_row.append(f"{s:.2f}")
        new_row.append(f"{cc*100/num_classes:.1f}")
        new_row.append(f"{e:.2f}")
        

    return new_row

if __name__ == "__main__":
    json_path = "analysis/results-ablation.json"   # path to your JSON file
    batch_size = 24              # set batch size here 

    new_row = generate_coverage_rows(
        json_path=json_path,
        batch_size=batch_size,
        random_mutation=True
    )

    print( " & ".join(new_row))
