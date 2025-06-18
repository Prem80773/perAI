from huggingface_hub import snapshot_download

# Download the full model snapshot to a local directory
local_dir = snapshot_download(
    repo_id="openai/clip-vit-base-patch32",
    cache_dir="./models",
    local_dir="./models/clip-vit-base-patch32",
    local_dir_use_symlinks=False  # ensures actual files are copied
)

print(f"Model downloaded to: {local_dir}")
