"""
upload_to_hf.py — Push PyCompat model to Hugging Face Hub
==========================================================

Usage:
    # Set your token first
    export HF_TOKEN="hf_your_token_here"
    
    # Push to your repo
    python upload_to_hf.py your-username/pycompat-model

    # Or pass token directly
    python upload_to_hf.py your-username/pycompat-model hf_your_token_here
"""

import sys
import os

def main():
    if len(sys.argv) < 2:
        print("Usage: python upload_to_hf.py <repo_id> [hf_token]")
        print("Example: python upload_to_hf.py SibgathKhan777/pycompat-model")
        print("\nSet HF_TOKEN env var or pass token as second argument.")
        sys.exit(1)

    repo_id = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("HF_TOKEN")

    if not token:
        print("❌ No Hugging Face token provided.")
        print("   Set HF_TOKEN env var or pass as second argument.")
        print("   Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Check if model exists
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        print("❌ Model not found in ./model/")
        print("   Train first: python pycompat_model.py train data.json ./model")
        sys.exit(1)

    # Install huggingface_hub if needed
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("📦 Installing huggingface_hub...")
        os.system("pip3 install huggingface_hub")
        from huggingface_hub import HfApi, create_repo

    # Also upload pycompat_model.py so users can load the model
    import shutil
    upload_dir = os.path.join(os.path.dirname(__file__), "_hf_upload_tmp")
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    
    # Copy model files
    shutil.copytree(model_dir, upload_dir)
    
    # Copy the model class file
    shutil.copy(
        os.path.join(os.path.dirname(__file__), "pycompat_model.py"),
        os.path.join(upload_dir, "pycompat_model.py")
    )

    # Create a simple __init__.py
    with open(os.path.join(upload_dir, "__init__.py"), "w") as f:
        f.write("from .pycompat_model import PyCompatModel\n")

    # Create requirements file for the model
    with open(os.path.join(upload_dir, "requirements.txt"), "w") as f:
        f.write("scikit-learn>=1.3.0\nnumpy>=1.24.0\njoblib>=1.3.0\npandas>=2.0.0\n")

    # Create example usage script
    with open(os.path.join(upload_dir, "example.py"), "w") as f:
        f.write('''"""Example usage of PyCompat model from Hugging Face Hub."""

from pycompat_model import PyCompatModel

# Load from local directory (after downloading from Hub)
model = PyCompatModel.load(".")

# Single prediction
result = model.predict("boto3", "1.42.49", "3.12", "darwin_x86_64")
print(f"Compatible: {result['is_compatible']} (confidence: {result['confidence']:.0%})")

# Recommendations
print("\\nTop 5 recommendations for alembic on Python 3.9:")
for r in model.recommend("alembic", "3.9", top_n=5):
    s = "✅" if r["is_compatible"] else "❌"
    print(f"  v{r['version']} {s} ({r['confidence']:.0%})")

# Batch prediction
results = model.predict_batch([
    {"package": "boto3", "version": "1.42.49", "python_version": "3.12"},
    {"package": "alembic", "version": "1.18.4", "python_version": "3.9"},
    {"package": "azure-core", "version": "1.38.0", "python_version": "3.11"},
])
print("\\nBatch results:")
for r in results:
    s = "✅" if r["is_compatible"] else "❌"
    print(f"  {r['package']} v{r['version']} on Py{r['python_version']}: {s}")
''')

    print(f"🚀 Uploading to https://huggingface.co/{repo_id} ...")

    api = HfApi(token=token)
    
    # Create repo
    try:
        create_repo(repo_id, token=token, repo_type="model", exist_ok=True)
        print(f"   Repo created/verified: {repo_id}")
    except Exception as e:
        print(f"   Repo status: {e}")

    # Upload
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
    )

    # Cleanup
    shutil.rmtree(upload_dir)

    print(f"\n✅ Model pushed successfully!")
    print(f"   🔗 https://huggingface.co/{repo_id}")
    print(f"\n   To use in another project:")
    print(f"   from huggingface_hub import snapshot_download")
    print(f"   from pycompat_model import PyCompatModel")
    print(f"   path = snapshot_download('{repo_id}')")
    print(f"   model = PyCompatModel.load(path)")


if __name__ == "__main__":
    main()
