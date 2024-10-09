from huggingface_hub import HfApi

user_name = "xcczach"
model_name = "YAGS"
repo_id = f"{user_name}/{model_name}"

api = HfApi()

api.create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
)

print("Uploading")
api.upload_folder(
    folder_path="to_upload",
    path_in_repo="",
    repo_id=repo_id,
    repo_type="model",
)
