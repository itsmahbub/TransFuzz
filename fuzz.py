from model_wrappers.resnet_wrapper import ResNetWrapper
from model_wrappers.mobile_vit_wrapper import MobileViTWrapper
import torch
import tool
import coverage_metrics
from fuzzer import Fuzzer
from torch.utils.data import DataLoader
import argparse
from model_wrappers.distil_hubert_wrapper import DistilHuBERTWrapper
from model_wrappers.mit_ast_wrapper import MITASTWrapper
from model_wrappers.wav2vec2_kws_wrapper import Wav2Vec2KWSWrapper
from model_wrappers.whisper_wrapper import WhisperTinyWrapper
from model_wrappers.blip_wrapper import BLIPWrapper
from model_wrappers.robust_resnet_wrapper import RobustResNetWrapper
from collections import Counter
from model_wrappers.utils import FilteredDataset, filter_incorrect
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from model_wrappers.wav2vec2_wrapper import Wav2Vec2Wrapper
from model_wrappers.hubert_asr_wrapper import HubertASRWrapper

hyper_map = {
    'NLC': None,
    'NC': 0,
    'KMNC': 5,
    'SNAC': None,
    'NBC': None,
    'TKNC': 10,
    'TKNP': 50,
    # 'CC': 10 if args.dataset == 'CIFAR10' else 1000,
    'LSA': 10,
    'DSA': 0.1,
    'MDSA': 10
}
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzz a DL model")
    parser.add_argument("--model", type=str, choices=["resnet50", "robustresnet", "mobilevit", "distilhubert", "whisper", "blip", "roberta_sentiment", "mitast", "wav2vec2kws", "wav2vec2asr", "hubertasr"], required=True, help="Model to fuzz")
    parser.add_argument("--model-path", type=str, default=None, help="Path to the model file (optional)")
    parser.add_argument("--dataset", type=str, choices=["UnsafeBench", "ImageNet", "CIFAR10", "ImageWoof", "speech_commands", "FLEURS", "CocoKarpathy", "TwitterSentiment", "LibriSpeech", "CommonVoice"], required=True, help="Dataset to use for seeds")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for DataLoader (default: 32)")
    parser.add_argument("--coverage-metric", type=str, choices=["NC", "KMNC", "TKNC", "TKNP", "NBC", "SNAC", "CC", "LSC", "DSC", "MDSC",  "NLC"], default="NLC", help="Coverage metric to use (default: NLC)")
    parser.add_argument("--split", type=str, choices=["train", "val", "validation", "test", "seed", "test-clean"], default="test", help="Dataset split to use (default: test)")
    parser.add_argument("--seed-count", type=int, default=-1, help="Number of seed inputs to use (-1 for all, default: -1)")
    parser.add_argument("--target-label", type=int, default=None, help="Target label for targeted attacks (optional)")
    parser.add_argument("--random-mutation", action="store_true", help="Use random mutation instead of gradient-based")
    parser.add_argument("--time-budget", type=int, default=-1,  help="Max time (seconds) to run before stopping (default: unlimited)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility (default: 0)")
    # parser.add_argument("--exploit-count", type=int, default=0, help="Number of times to exploit local region (default: 0)")
    args = parser.parse_args()

    ae_dir = f"adversarial-examples/{args.model}/{args.model_path}/{args.dataset}/{args.coverage_metric}/{args.target_label}/{args.batch_size}/{args.seed}"
    if args.random_mutation:
        ae_dir += "-rand"
    # if path exists then exit
    if os.path.exists(ae_dir):
        print(f"Adversarial examples already exist in {ae_dir}. Exiting.")
        exit(0)


    set_seed(args.seed)
    
    device =  torch.device('cpu')
    device = None # GPU if available
    inference_func = None
    if args.model == "mobilevit":
        model_wrapper = MobileViTWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 3, 224, 224).to(model_wrapper.device)
        noise_range=(-8, 8)
    elif args.model == "resnet50":
        model_wrapper = ResNetWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 3, 224, 224).to(model_wrapper.device)
        noise_range=(-8, 8)
    elif args.model == "robustresnet":
        model_wrapper = RobustResNetWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 3, 224, 224).to(model_wrapper.device)
        noise_range=(-8, 8)
        
    elif args.model == "distilhubert":
        model_wrapper = DistilHuBERTWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 16000).to(model_wrapper.device) # 1 second of audio at 16kHz
        noise_range=(-1, 1)
    elif args.model == "mitast":
        model_wrapper = MITASTWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 16000).to(model_wrapper.device) # 1 second of audio at 16kHz
        random_input = model_wrapper.preprocess(random_input)
        inference_func = model_wrapper.predict_outputs
        noise_range=(-1, 1)
    elif args.model == "wav2vec2kws":
        model_wrapper = Wav2Vec2KWSWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 16000).to(model_wrapper.device) # 1 second of audio at 16kHz
        random_input = model_wrapper.preprocess(random_input)
        inference_func = model_wrapper.predict_outputs
        noise_range=(-1, 1)
    elif args.model == "whisper":
        model_wrapper = WhisperTinyWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 16000).to(model_wrapper.device) # 1 second of audio at 16kHz
        random_input = model_wrapper.preprocess(random_input)
        inference_func = model_wrapper.predict_outputs
        noise_range=(-1, 1)
    elif args.model == "wav2vec2asr":
        model_wrapper = Wav2Vec2Wrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 16000).to(model_wrapper.device) # 1 second of audio at 16kHz
        random_input = model_wrapper.preprocess(random_input)
        inference_func = model_wrapper.predict_outputs
        noise_range=(-1, 1)
    elif args.model == "hubertasr":
        model_wrapper = HubertASRWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 16000).to(model_wrapper.device) # 1 second of audio at 16kHz
        random_input = model_wrapper.preprocess(random_input)
        inference_func = model_wrapper.predict_outputs
        noise_range=(-1, 1)
    elif args.model == "blip":
        model_wrapper = BLIPWrapper(model_path=args.model_path, device=device)
        random_input = torch.randn(1, 3, 384, 384).to(model_wrapper.device)
        inference_func = model_wrapper.predict_outputs
        noise_range=(-8, 8)
   
    
    

    layer_size_dict = tool.get_layer_output_sizes(model_wrapper.model, random_input,  inference_func=inference_func)

    seeds_dataset = model_wrapper.get_seeds(dataset_name=args.dataset, preprocessed=False, split=args.split, count=args.seed_count)
    seeds_dataset = filter_incorrect(seeds_dataset, model_wrapper, batch_size=args.batch_size, preprocessed=False)
    # labels = set([x[1].item() for x in seeds_dataset])
    # print("# Classes", len(labels))
   


    seeds_dataset_preprocessed = model_wrapper.get_seeds(dataset_name=args.dataset, preprocessed=True, split=args.split, count=args.seed_count)
    seeds_dataset_preprocessed = filter_incorrect(seeds_dataset_preprocessed, model_wrapper, batch_size=args.batch_size, preprocessed=True)

    seeds_loader = DataLoader(seeds_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=model_wrapper.collate_fn)
    seeds_loader_preprocessed = DataLoader(seeds_dataset_preprocessed, batch_size=args.batch_size, shuffle=True, collate_fn=model_wrapper.collate_fn)

    coverage_metric_class = getattr(coverage_metrics, args.coverage_metric)
    coverage_metric = coverage_metric_class(model_wrapper.model, layer_size_dict, device=model_wrapper.device, inference_func=inference_func, hyper=hyper_map[args.coverage_metric])
    coverage_metric.build(seeds_loader_preprocessed)
    if args.coverage_metric not in ['CC', 'TKNP', 'LSC', 'DSC', 'MDSC']:
        coverage_metric.assess(seeds_loader_preprocessed)

    print(f"Initial Coverage: {coverage_metric.current}")
    fuzzer = Fuzzer(seeds_loader, model_wrapper, coverage_metric, coverage_guided=True, target_label=args.target_label, epochs=-1, timeout=args.time_budget, ae_dir=ae_dir, random_mutation=args.random_mutation, noise_range=noise_range)
    fuzzer.run()
    print("Seed Count", len(seeds_dataset))

# python fuzz.py --model resnet50 --dataset ImageNet --batch-size 8  --split val  --time-budget 300 --coverage-metric KMNC
# python fuzz.py --model robustresnet --dataset ImageNet --batch-size 32  --split val  --time-budget 300 

# python fuzz.py --model resnet50 --model-path resnet50_unsafebench.pth --dataset UnsafeBench --batch-size 1 --split test --seed-count 1000 --time-budget 300 --target-label 0
# python fuzz.py --model resnet50 --model-path resnet50_cifar10.pth --dataset CIFAR10 --seed-count 2000
# python fuzz.py --model resnet50 --model-path resnet50_imagewoof.pth --dataset ImageWoof --split validation --seed-count 200 --time-budget 300 --batch-size 1
# python fuzz.py --model resnet50 --model-path resnet50_imagewoof.pth --dataset ImageWoof --split validation --seed-count 200 --target-label 1

# python fuzz.py --model resnet50 --model-path resnet50_imagewoof.pth --dataset ImageWoof --split validation --seed-count 50


# python fuzz.py --model mitast --dataset speech_commands --batch-size 32 --split test --time-budget 300 --seed-count 1000 --target-label 28 --seed 1
# python fuzz.py --model wav2vec2kws --dataset speech_commands --batch-size 16 --split test --time-budget 300 --seed-count 1000  --target-label 33

# python fuzz.py --model keywordspotting --dataset speech_commands --batch-size 16 --split test --time-budget 300 --seed-count 1000 --model-path Amirhossein75/Keyword-Spotting --target-label 0  
# python fuzz.py --model keywordspotting --dataset speech_commands --batch-size 16 --split test --time-budget 300 --seed-count 1000 --model-path MIT/ast-finetuned-speech-commands-v2 --target-label 28 --seed 0
# python fuzz.py --model roberta_sentiment --dataset TwitterSentiment --batch-size 32 --split test --time-budget 300
# python fuzz.py --model hubertasr --dataset LibriSpeech --batch-size 8 --split test-clean --time-budget 300 --seed-count -1
# python fuzz.py --model distilhubert --dataset speech_commands --batch-size 32 --split test --time-budget 300 --seed-count 2000
