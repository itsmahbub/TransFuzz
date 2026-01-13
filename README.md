# neuro-fuzz


### Seeds

ImageNet: https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000


## Comparison with Yuan et al.

python fuzz.py --model resnet50 --dataset ImageNet --split seed
Time 30 minutes
Seed 4000


# Experiments
## AutoAttack
python auto_attack.py --data seeds/imagenet-mini/seed/ --out-root adversarial-examples/resnet50/ImageNet/autoattack --time-budget 300 

rsync --ignore-existing -av   lab-230:~/code/neuro-fuzz/adversarial-examples/mobilevit/mobilevit_unsafebench/UnsafeBench/NLC/ adversarial-examples/mobilevit/mobilevit_unsafebench/UnsafeBench/NLC/

rsync -av  adversarial-examples/mitast/None/speech_commands/NLC/5/ lab-230:~/code/neuro-fuzz/adversarial-examples/mitast/None/speech_commands/NLC/5/


rsync -av  adversarial-examples/wav2vec2kws/None/speech_commands/NLC/33/24/ lab-230:~/code/neuro-fuzz/adversarial-examples/wav2vec2kws/None/speech_commands/NLC/33/24/

rsync -av adversarial-examples/resnet50/resnet50_unsafebench.pth/UnsafeBench/NLC/0/24 lab-230:~/code/neuro-fuzz/adversarial-examples/resnet50/resnet50_unsafebench.pth/UnsafeBench/NLC/0/24

rsync -av adversarial-examples/mobilevit/mobilevit_unsafebench/UnsafeBench/NLC/0/24 lab-230:~/code/neuro-fuzz/adversarial-examples/mobilevit/mobilevit_unsafebench/UnsafeBench/NLC/0/24


rsync -av adversarial-examples/wav2vec2asr/None/LibriSpeech/NLC/None/1/0/ lab-230:~/code/neuro-fuzz/adversarial-examples/wav2vec2asr/None/LibriSpeech/NLC/None/1/0/


rsync -av adversarial-examples/mobilevit/None/ImageNet/NLC/None/24/2/aes/ lab-230:~/code/neuro-fuzz/adversarial-examples/mobilevit/None/ImageNet/NLC/None/1/2/aes/

rsync -av adversarial-examples/mitast/None/speech_commands/NLC/None/1/1/aes/ lab-230:~/code/neuro-fuzz/adversarial-examples/mitast/None/speech_commands/NLC/None/1/1/aes/

rsync -av adversarial-examples/mobilevit/None/ImageNet/NLC/None/1/0/aes/ lab-230:~/code/neuro-fuzz/adversarial-examples/mobilevit/None/ImageNet/NLC/None/1/0/aes/


rsync -av adversarial-examples/resnet50/None/ImageNet/NLC/None/8-30minutes/0/coverage.json lab-230:~/code/neuro-fuzz/adversarial-examples/resnet50/None/ImageNet/NLC/None/8-30minutes/0/coverage.json

rsync --ignore-existing -av adversarial-examples/resnet50/None/ImageNet/NLC/None/1/0/poisons/ lab-230:~/code/neuro-fuzz/adversarial-examples/resnet50/None/ImageNet/NLC/None/1/0/poisons/