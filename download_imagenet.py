import kagglehub
import shutil

target = "./seeds/"

# Download latest version
path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
shutil.copytree(path, target, dirs_exist_ok=True)

print("Path to dataset files:", path)
