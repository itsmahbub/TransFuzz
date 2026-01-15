
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import json
from torch.utils.data import ConcatDataset
from diversity import calculate_diversity
from calculate_coverage import calculate_coverage
from calculate_stability import calculate_stability
from model_transfer import model_transfer_attack
from model_wrappers.resnet_wrapper import ResNetWrapper
from model_wrappers.mobile_vit_wrapper import MobileViTWrapper
from transfuzz import filter_incorrect
from model_wrappers.mit_ast_wrapper import MITASTWrapper
from model_wrappers.wav2vec2_kws_wrapper import Wav2Vec2KWSWrapper
from calculate_stability import calculate_adv_class_dist
import math
import argparse
from model_wrappers.robust_resnet_wrapper import RobustResNetWrapper

should_calculate_model_transfer = False
should_calculate_diversity = False
should_calculate_naturalness = True
should_calculate_stability = False
should_calculate_coverage = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-count", type=int, default=1000, help="Number of seeds used for fuzzing")
    parser.add_argument("--clean-seed-count", type=int, default=0, help="Number of clean seeds used for fuzzing")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size used for fuzzing")
    parser.add_argument("--target-label", type=int, default=None, help="Target label used for fuzzing")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model file (optional)")
    parser.add_argument("--model-name", type=str, default="resnet50", help="Model name")
    parser.add_argument("--dataset-name", type=str, default="UnsafeBench", help="Dataset name")
    parser.add_argument("--attacked-model-name", type=str, default="resnet50", help="Attacked Model name")
    parser.add_argument("--attacked-model-path", type=str, default=None, help="Path to the attacked model file (optional)")
    parser.add_argument("--time-budget", type=int, default=300, help="Time budget used for fuzzing")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes in the dataset")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    parser.add_argument("--random-mutation", action="store_true", help="Use random mutation instead of gradient-based")

    args = parser.parse_args()


    new_dataset_count = args.seed_count
    seed_count = args.seed_count
    clean_seed_count = args.clean_seed_count
    target_label = args.target_label
    batch_size = args.batch_size
    seed = args.seed
    rand = "-rand" if args.random_mutation else ""


    model_name = args.model_name

    model_path = args.model_path
    attacked_model_path = args.attacked_model_path
    attacked_model_name = args.attacked_model_name
    dataset_name = args.dataset_name
    num_classes = args.num_classes

    print("Attacked Model: ", attacked_model_name)
    result_key = f"{model_name}-{model_path}-{dataset_name}-{target_label}-{batch_size}-{seed}{rand}"
    # result_key = f"yuan-{model_name}-{model_path}-{dataset_name}-{target_label}-{batch_size}-{seed}"

    orig_root = f"outputs/{model_name}/{model_path}/{dataset_name}/NLC/{target_label}/{batch_size}/{seed}{rand}/orig"
    aes_root  = f"outputs/{model_name}/{model_path}/{dataset_name}/NLC/{target_label}/{batch_size}/{seed}{rand}/aes"
    poison_dir = f"outputs/{model_name}/{model_path}/{dataset_name}/NLC/{target_label}/{batch_size}/{seed}{rand}/poisons"


    # orig_root = f"outputs/yuan/{dataset_name}-{model_name}-NLC-{seed}/image/orig"
    # aes_root = f"outputs/yuan/{dataset_name}-{model_name}-NLC-{seed}/image/aes"
    # poison_dir = ""

    new_dataset_split = "train"
    seed_dataset_split = "val"

    with open("results.json", "r") as f:
        try:
            results = json.load(f)
        except:
            results = {}
    with open("initial_coverage.json", "r") as f:
        try:
            initial_coverages = json.load(f)
        except:
            initial_coverages = {}
    
    
    existing_results = results.get(result_key, {
        "seed_count": seed_count,
        "clean_seed": clean_seed_count,
        "time_budget": 300,
        "N": batch_size,
        "dataset": dataset_name,
        "target_label": target_label,
        "number_of_classes": num_classes
    })
    
    # if necessary metrics are already calculate, exit
    all_done = True
    if should_calculate_diversity and  "diversity" not in existing_results:
        all_done = False
    if should_calculate_naturalness and "naturalness" not in existing_results:
        all_done = False
    if should_calculate_stability and "stability" not in existing_results:
        all_done = False
    if should_calculate_model_transfer and f"model_transfer-{attacked_model_name}" not in existing_results:
        all_done = False
    if "adversarial_classes" not in existing_results:
        all_done = False
    if "generated_count" not in existing_results:
        all_done = False
    if "shared_perturbation" not in existing_results:
        all_done = False
    if "coverage" not in existing_results:
        all_done = False
    if f"{model_name}-{dataset_name}" not in initial_coverages:
        all_done = False
    if all_done:
        exit(0)




    if should_calculate_model_transfer:
        if attacked_model_name == "resnet50":
            attacked_model_wrapper = ResNetWrapper(model_path=attacked_model_path)
        elif attacked_model_name == "mobilevit":
            attacked_model_wrapper = MobileViTWrapper(model_path=attacked_model_path)
        elif attacked_model_name == "mitast":
            attacked_model_wrapper = MITASTWrapper(model_path=attacked_model_path)
        elif attacked_model_name == "wav2vec2kws":
            attacked_model_wrapper = Wav2Vec2KWSWrapper(model_path=attacked_model_path)
        elif attacked_model_name == "robustresnet":
            attacked_model_wrapper = RobustResNetWrapper(model_path=model_path)


    if model_name == "resnet50":
        model_wrapper = ResNetWrapper(model_path=model_path)
    elif model_name == "mobilevit":
        model_wrapper = MobileViTWrapper(model_path=model_path)
    elif model_name == "mitast":
        model_wrapper = MITASTWrapper(model_path=model_path)
    elif model_name == "wav2vec2kws":
        model_wrapper = Wav2Vec2KWSWrapper(model_path=model_path)
    elif model_name == "robustresnet":
        model_wrapper = RobustResNetWrapper(model_path=model_path)



    if model_name in ["resnet50", "mobilevit", "robustresnet"]:
        random_input = torch.randint(
            low=0, high=256, size=(1, 3, 224, 224), device=model_wrapper.device, dtype=torch.uint8
        )
        random_input = model_wrapper.preprocess(random_input)
    elif model_name in ["mitast", "wav2vec2kws"]:
        random_input = torch.randn(1, 16000).to(model_wrapper.device)
        random_input = model_wrapper.preprocess(random_input)
    

    if model_name in ["mitast", "wav2vec2kws"]:
        adv_dataset = model_wrapper.get_seeds(dataset_name="Adversarial", preprocessed=True, split="all", data_dir=aes_root, source_label2id=model_wrapper.model.config.label2id)
        if should_calculate_model_transfer:
            adv_dataset_for_transfer = attacked_model_wrapper.get_seeds(dataset_name="Adversarial", preprocessed=True, split="all", data_dir=aes_root, source_label2id=model_wrapper.model.config.label2id)
    else:
        adv_dataset = model_wrapper.get_seeds(dataset_name="Adversarial", preprocessed=True, split="all", data_dir=aes_root)
        if should_calculate_model_transfer:
            adv_dataset_for_transfer = attacked_model_wrapper.get_seeds(dataset_name="Adversarial", preprocessed=True, split="all", data_dir=aes_root)
   

    new_dataset = model_wrapper.get_seeds(dataset_name=dataset_name, preprocessed=False, split=new_dataset_split, count=new_dataset_count)
    seed_dataset = model_wrapper.get_seeds(dataset_name=dataset_name, preprocessed=True, split=seed_dataset_split, count=seed_count)
    seed_dataset_unprocessed = model_wrapper.get_seeds(dataset_name=dataset_name, preprocessed=False, split=seed_dataset_split, count=seed_count)

    if "generated_count" not in existing_results:
        existing_results["generated_count"] = len(adv_dataset)
    
    if f"{model_name}-{dataset_name}" not in initial_coverages:
        init_coverage = calculate_coverage(model_wrapper=model_wrapper, random_input=random_input, dataset=seed_dataset, batch_size=32)
        initial_coverages[f"{model_name}-{dataset_name}"] = init_coverage

    if should_calculate_coverage and "coverage" not in existing_results:
        adv_coverage = calculate_coverage(model_wrapper=model_wrapper, random_input=random_input, dataset=adv_dataset, batch_size=32)
        combined_dataset = ConcatDataset([seed_dataset, adv_dataset])
        new_coverage = calculate_coverage(model_wrapper=model_wrapper, random_input=random_input, dataset=combined_dataset, batch_size=32)
        existing_results["coverage"] = {
            "new": new_coverage,
            "adv": adv_coverage
        }

    if should_calculate_diversity and  "diversity" not in existing_results:
        diversity_results = calculate_diversity(aes_root, num_classes=num_classes)
        existing_results["diversity"] = diversity_results

    if should_calculate_naturalness and "naturalness" not in existing_results:
        
        if model_name in ["mitast", "wav2vec2kws"]:
            from audio_naturalness_v1 import calculate_audio_naturalness
            naturualness_results = calculate_audio_naturalness(orig_root, aes_root)
        else:
            from naturalness import calculate_naturalness
            naturualness_results = calculate_naturalness(orig_root, aes_root)
        existing_results["naturalness"] = naturualness_results

    if should_calculate_stability and "stability" not in existing_results:
        try:
            stability_results = calculate_stability(model_wrapper=model_wrapper, dataset=adv_dataset)
            existing_results["stability"] = stability_results
        except Exception as e:
            print("Stability calculation failed:", e)
            raise e
            
    if should_calculate_model_transfer and f"model_transfer-{attacked_model_name}" not in existing_results:
        try:
            model_transfer_results = model_transfer_attack(model_wrapper=attacked_model_wrapper, dataset=adv_dataset_for_transfer)
            existing_results[f"model_transfer-{attacked_model_name}"] = model_transfer_results
        except Exception as e:
            print("Model transfer calculation failed:", e)

    if "adversarial_classes" not in existing_results:
        adv_class_dist = calculate_adv_class_dist(model_wrapper=model_wrapper, dataset=adv_dataset)
        existing_results["adversarial_classes"] = adv_class_dist["aversarial_classes"]
    results[result_key] = existing_results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    with open("initial_coverage.json", "w") as f:
        json.dump(initial_coverages, f, indent=4)

if __name__ == "__main__":
    main()

# python analysis/main.py --seed-count 3923 --clean-seed-count 3121 --batch-size 24 --model-name mobilevit --dataset-name ImageNet --attacked-model-name mobilevit --time-budget 300 --num-classes 1000 --seed 0
# python analysis/main.py --seed-count 3923 --clean-seed-count 2483 --batch-size 32 --model-name robustresnet --dataset-name ImageNet --attacked-model-name  resnet50 --time-budget 300 --num-classes 1000 --seed 0

# python analysis/main.py --seed-count 3923 --clean-seed-count 3121 --batch-size 128 --model-name resnet50 --dataset-name ImageNet --attacked-model-name robustresnet --time-budget 300 --num-classes 1000 --seed 0
