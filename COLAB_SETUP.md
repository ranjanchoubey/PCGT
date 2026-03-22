# Hybrid Workflow: Local Copilot + Colab GPU

## Overview
- **Local (Mac)**: Edit code with VS Code + Copilot + review agent
- **Colab/Kaggle**: Run GPU experiments fast
- **Sync**: Git push/pull

---

## 1. One-Time Setup (Local)

```bash
cd /Users/vn59a0h/thesis/PCGT
git init
git add -A
git commit -m "PCGT v4 — ready for GPU experiments"
```

Create a **private** GitHub repo, then:
```bash
git remote add origin git@github.com:YOUR_USERNAME/PCGT.git
git branch -M main
git push -u origin main
```

### .gitignore (add before first commit)
```
venv/
__pycache__/
*.pyc
data/
wandb/
*.pt
*.pth
results/
```

> Keep datasets out of git — download them on Colab separately.

---

## 2. Colab Notebook Template

Create a notebook `PCGT_Runner.ipynb` on Google Colab:

```python
# Cell 1: Clone repo (first time) or pull updates
import os
if not os.path.exists('PCGT'):
    !git clone https://github.com/YOUR_USERNAME/PCGT.git
else:
    %cd PCGT
    !git pull
    %cd ..

%cd PCGT
```

```python
# Cell 2: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install torch-geometric pyg-lib torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
!pip install pymetis scikit-learn ogb
```

```python
# Cell 3: Verify GPU
import torch
print(f"GPU: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}")
```

```python
# Cell 4: Run experiment (example — Cora)
!cd medium && python main.py \
    --method pcgt \
    --dataset cora \
    --num_partitions 7 \
    --num_reps 3 \
    --partition_method metis \
    --lr 0.01 \
    --num_layers 3 \
    --hidden_channels 64 \
    --weight_decay 5e-4 \
    --dropout 0.5 \
    --runs 10 \
    --epochs 500 \
    --device cuda:0
```

```python
# Cell 5: Run all medium datasets
!cd medium && bash ../FINAL_CONFIGS.sh
```

```python
# Cell 6: Save results back (optional — or just copy from output)
# If you want to push results back to GitHub:
!git add results/
!git commit -m "Results from Colab run"
!git push
```

---

## 3. Daily Workflow

### When editing code (local Mac):
```bash
# Edit in VS Code with Copilot
# When ready to test on GPU:
git add -A && git commit -m "description" && git push
```

### When running on Colab:
```python
# In Colab, just pull and run:
!cd PCGT && git pull
!cd PCGT/medium && python main.py --method pcgt --dataset cora ... --device cuda:0
```

### When results come back:
- Copy final numbers from Colab output
- Or push results from Colab → pull locally

---

## 4. For Large-Scale (ogbn-arxiv, ogbn-products)

Colab free tier: T4 (16GB VRAM), 12h session limit
Colab Pro ($12/mo): A100 (40GB), 24h sessions, priority queue

```python
# ogbn-arxiv needs ~4GB VRAM — T4 is fine
!cd PCGT/medium && python main.py \
    --method pcgt \
    --dataset ogbn-arxiv \
    --num_partitions 1000 \
    --num_reps 5 \
    --lr 0.001 \
    --hidden_channels 128 \
    --runs 10 \
    --device cuda:0

# ogbn-products needs A100 + batching
# Will implement when we get there
```

---

## 5. Kaggle Alternative (if Colab quota runs out)

- 30h/week GPU (T4 x2 or P100)
- No SSH, but notebooks work the same way
- Use Kaggle Datasets to upload your repo as a zip

```python
# Kaggle notebook:
!cp -r /kaggle/input/pcgt-code/* /kaggle/working/
%cd /kaggle/working/PCGT
!pip install pymetis torch-geometric ...
!cd medium && python main.py ... --device cuda:0
```

---

## 6. Tips

- **Save checkpoints** to Google Drive on Colab to survive disconnects:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  # Save: cp results/* /content/drive/MyDrive/PCGT_results/
  ```

- **Screen your experiments**: Run multiple datasets in sequence in one cell — if Colab disconnects mid-run, you keep earlier results

- **Don't install Jupyter locally for GPU** — it's not worth the complexity. The git push/pull loop takes 10 seconds.

---

## Priority Order

| Phase | Where | What |
|-------|-------|------|
| Now | Local Mac | Finish Deezer/Squirrel (already queued) |
| Next | Colab Free | Reproduce all 7 datasets on GPU (verify) |
| Then | Colab Free/Pro | ogbn-arxiv (K=1000) |
| Later | Colab Pro | ogbn-products (batching needed) |
| Paper | Local Mac | Writing, ablations on small datasets |
