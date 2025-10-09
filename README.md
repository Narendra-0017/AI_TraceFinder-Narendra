# AI_TraceFinder

Local Streamlit app and model to detect scanner models from scanned images.

Quick start

1. Create a Python environment and install requirements:

```powershell
python -m pip install -r requirements.txt
python -m pip install tifffile
```

2. Run Streamlit:

```powershell
python -m streamlit run app.py
```

Diagnostics

- Use `diagnostics_from_urls.py` to download images from direct URLs listed in `urls.txt` and produce `diagnostics_results.csv`.

Pushing to GitHub

1. Create a new empty repository on GitHub (e.g. `TraceFinder`).
2. Add it as a remote and push:

```powershell
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/TraceFinder.git
git branch -M main
git push -u origin main
```

Note: Do NOT push large model files (e.g. `haar_hybridcnn.pth`) to GitHub. Use Git LFS or a cloud storage link instead.
