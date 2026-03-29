import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


# ─────────────────────────────────────────────
# 1. CANVAS PREPROCESSING
# ─────────────────────────────────────────────

CANVAS_SIZE = 64
NUM_COLORS = 10

def grid_to_onehot(grid: list[list[int]]) -> np.ndarray:
    """Convert a raw ARC grid to a (NUM_COLORS, H, W) one-hot array."""
    arr = np.array(grid, dtype=np.int64)
    h, w = arr.shape
    oh = np.zeros((NUM_COLORS, h, w), dtype=np.float32)
    for c in range(NUM_COLORS):
        oh[c] = (arr == c).astype(np.float32)
    return oh

def place_on_canvas(oh: np.ndarray, scale: int, offset_y: int, offset_x: int) -> np.ndarray:
    """Place a scaled one-hot grid onto a black CANVAS_SIZE×CANVAS_SIZE canvas."""
    c, h, w = oh.shape
    canvas = np.zeros((c, CANVAS_SIZE, CANVAS_SIZE), dtype=np.float32)
    # nearest-neighbor upscale
    scaled = oh.repeat(scale, axis=1).repeat(scale, axis=2)
    sh, sw = h * scale, w * scale
    # clip to canvas bounds
    sh = min(sh, CANVAS_SIZE - offset_y)
    sw = min(sw, CANVAS_SIZE - offset_x)
    canvas[:, offset_y:offset_y+sh, offset_x:offset_x+sw] = scaled[:, :sh, :sw]
    return canvas

def preprocess_pair(inp: list, out: list, augment: bool = True):
    """
    Convert one (input, output) ARC pair into a (20, 64, 64) canvas tensor.
    Channels 0-9: input grid one-hot
    Channels 10-19: output grid one-hot
    """
    ih, iw = len(inp), len(inp[0])
    oh, ow = len(out), len(out[0])

    # pick scale so the larger grid fits comfortably
    max_dim = max(ih, iw, oh, ow)
    scale = max(1, (CANVAS_SIZE // 2) // max_dim)  # leave room for translation aug

    if augment:
        max_offset = CANVAS_SIZE - max_dim * scale
        max_offset = max(0, max_offset)
        oy = random.randint(0, max_offset)
        ox = random.randint(0, max_offset)
    else:
        oy, ox = 0, 0

    inp_oh = grid_to_onehot(inp)
    out_oh = grid_to_onehot(out)

    inp_canvas = place_on_canvas(inp_oh, scale, oy, ox)
    out_canvas = place_on_canvas(out_oh, scale, oy, ox)  # same position/scale

    return np.concatenate([inp_canvas, out_canvas], axis=0)  # (20, 64, 64)

def apply_color_permutation(canvas: np.ndarray) -> np.ndarray:
    """Randomly permute color channels (same permutation for input and output)."""
    perm = np.random.permutation(NUM_COLORS)
    inp = canvas[:NUM_COLORS][perm]
    out = canvas[NUM_COLORS:][perm]
    return np.concatenate([inp, out], axis=0)

def apply_spatial_aug(canvas: np.ndarray) -> np.ndarray:
    """Random flip/rotation of the whole canvas (input+output together)."""
    if random.random() < 0.5:
        canvas = np.flip(canvas, axis=2).copy()   # horizontal flip
    if random.random() < 0.5:
        canvas = np.flip(canvas, axis=1).copy()   # vertical flip
    k = random.randint(0, 3)
    canvas = np.rot90(canvas, k=k, axes=(1, 2)).copy()
    return canvas

def preprocess_task_file(task_path: str, augment: bool = True) -> list[np.ndarray]:
    """Load a task JSON and return a list of preprocessed (20,64,64) pair tensors."""
    with open(task_path) as f:
        task = json.load(f)
    pairs = []
    for pair in task["train"]:
        canvas = preprocess_pair(pair["input"], pair["output"], augment=augment)
        if augment:
            canvas = apply_color_permutation(canvas)
            canvas = apply_spatial_aug(canvas)
        pairs.append(canvas)
    return pairs  # list of (20, 64, 64) arrays

def preprocess_and_save(data_dirs: list[str], out_dir: str, augment: bool = True, n_aug: int = 5):
    """
    Preprocess all tasks and save to disk.
    Each task produces n_aug augmented versions of each pair.
    Saves: {out_dir}/{task_id}_pair{p}_aug{a}.npy
    Also saves a task index: {out_dir}/task_index.json
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    task_index = {}  # task_id -> list of saved pair file paths

    for d in data_dirs:
        for task_file in Path(d).glob("*.json"):
            task_id = task_file.stem
            task_index[task_id] = []
            with open(task_file) as f:
                task = json.load(f)
            for p_idx, pair in enumerate(task["train"]):
                for a_idx in range(n_aug if augment else 1):
                    canvas = preprocess_pair(pair["input"], pair["output"], augment=augment)
                    if augment:
                        canvas = apply_color_permutation(canvas)
                        canvas = apply_spatial_aug(canvas)
                    fname = f"{task_id}_pair{p_idx}_aug{a_idx}.npy"
                    np.save(out_path / fname, canvas)
                    task_index[task_id].append(fname)

    with open(out_path / "task_index.json", "w") as f:
        json.dump(task_index, f)
    print(f"Saved {len(task_index)} tasks to {out_dir}")
    return task_index


# ─────────────────────────────────────────────
# 2. DATASET
# ─────────────────────────────────────────────

class ARCPairDataset(Dataset):
    """Returns individual (20,64,64) pair tensors for autoencoder training."""
    def __init__(self, preprocessed_dir: str):
        self.dir = Path(preprocessed_dir)
        with open(self.dir / "task_index.json") as f:
            task_index = json.load(f)
        self.files = [f for files in task_index.values() for f in files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.dir / self.files[idx])
        return torch.from_numpy(arr)  # (20, 64, 64)


# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────

class StraightThrough(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        return grad  # pass gradients straight through

def straight_through_binarize(x):
    return StraightThrough.apply(x)


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class ConvTBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
        )
    def forward(self, x):
        return self.block(x)


class ARCEncoder(nn.Module):
    """
    Input:  (B, 20, 64, 64)
    Output: (B, latent_dim) binary codes
    """
    def __init__(self, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(20, 32),    # -> (B, 32,  32, 32)
            ConvBlock(32, 64),    # -> (B, 64,  16, 16)
            ConvBlock(64, 128),   # -> (B, 128,  8,  8)
            ConvBlock(128, 256),  # -> (B, 256,  4,  4)
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 * 4 * 4, latent_dim)

    def forward(self, x):
        h = self.conv(x)                    # (B, 256, 4, 4)
        h = self.dropout(h.flatten(1))      # (B, 256*4*4)
        z_cont = self.fc(h)                 # (B, latent_dim)  continuous
        z_bin  = straight_through_binarize(z_cont)  # (B, latent_dim)  binary
        return z_bin, z_cont


class ARCDecoder(nn.Module):
    """
    Reconstructs the OUTPUT grid only.
    Input conditioning: input grid channels (0:10) are encoded separately
    and concatenated with z at the bottleneck.
    Output: (B, 10, 64, 64) logits over 10 colors
    """
    def __init__(self, latent_dim: int = 128, inp_enc_dim: int = 256):
        super().__init__()
        # small encoder for the input grid (conditioning signal)
        self.inp_enc = nn.Sequential(
            ConvBlock(10, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, inp_enc_dim),   # -> (B, 256, 4, 4)
        )
        fused_dim = latent_dim + inp_enc_dim * 4 * 4
        self.fc = nn.Linear(fused_dim, 256 * 4 * 4)

        self.deconv = nn.Sequential(
            ConvTBlock(256, 128),   # -> (B, 128,  8,  8)
            ConvTBlock(128, 64),    # -> (B, 64,  16, 16)
            ConvTBlock(64, 32),     # -> (B, 32,  32, 32)
            ConvTBlock(32, 16),     # -> (B, 16,  64, 64)
            nn.Conv2d(16, 10, 1),   # -> (B, 10,  64, 64) logits
        )

    def forward(self, z, inp_canvas):
        """
        z:          (B, latent_dim)   binary latent
        inp_canvas: (B, 10, 64, 64)   input grid channels only
        """
        inp_enc = self.inp_enc(inp_canvas).flatten(1)   # (B, 256*4*4)
        fused   = torch.cat([z, inp_enc], dim=1)        # (B, latent_dim + 256*4*4)
        h       = self.fc(fused).view(-1, 256, 4, 4)    # (B, 256, 4, 4)
        return self.deconv(h)                            # (B, 10, 64, 64)


class ARCAutoEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.encoder = ARCEncoder(latent_dim)
        self.decoder = ARCDecoder(latent_dim)
        self.latent_dim = latent_dim

    def forward(self, canvas):
        """
        canvas: (B, 20, 64, 64)
        Returns: logits (B, 10, 64, 64), z_bin, z_cont
        """
        z_bin, z_cont = self.encoder(canvas)
        inp_canvas    = canvas[:, :10]              # input channels only
        logits        = self.decoder(z_bin, inp_canvas)
        return logits, z_bin, z_cont


# ─────────────────────────────────────────────
# 4. LOSS
# ─────────────────────────────────────────────

def arc_loss(logits, canvas):
    """
    Cross-entropy between predicted output and true output grid.
    logits:  (B, 10, 64, 64)
    canvas:  (B, 20, 64, 64)  — ground truth output is channels 10:20
    """
    out_oh    = canvas[:, 10:]                         # (B, 10, 64, 64)
    targets   = out_oh.argmax(dim=1)                   # (B, 64, 64)  class indices
    return F.cross_entropy(logits, targets)


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    preprocessed_dir: str,
    n_epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    latent_dim: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_path: str = "arc_ae.pt",
):
    dataset    = ARCPairDataset(preprocessed_dir)
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model      = ARCAutoEncoder(latent_dim).to(device)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            logits, z_bin, z_cont = model(batch)
            loss = arc_loss(logits, batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        avg = total_loss / len(loader)
        print(f"Epoch {epoch+1:3d}/{n_epochs}  loss={avg:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved model to {save_path}")
    return model


# ─────────────────────────────────────────────
# 6. RETRIEVAL / INFERENCE
# ─────────────────────────────────────────────

def embed_task(task_path: str, model: ARCAutoEncoder, device: str) -> np.ndarray:
    """
    Embed a task's demo pairs and return a single 128-bit task vector
    via majority vote over the pair embeddings.
    Returns: (latent_dim,) binary numpy array
    """
    model.eval()
    with open(task_path) as f:
        task = json.load(f)

    pair_codes = []
    with torch.no_grad():
        for pair in task["train"]:
            # no augmentation at inference time
            canvas = preprocess_pair(pair["input"], pair["output"], augment=False)
            t = torch.from_numpy(canvas).unsqueeze(0).to(device)  # (1, 20, 64, 64)
            _, z_bin, _ = model(t)
            pair_codes.append(z_bin.cpu().numpy()[0])              # (latent_dim,)

    codes = np.stack(pair_codes, axis=0)        # (n_pairs, latent_dim)
    majority = (codes.mean(axis=0) >= 0.5).astype(np.float32)  # majority vote
    return majority


def build_index(
    task_dir: str,
    model: ARCAutoEncoder,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Embed all tasks in task_dir and return an index matrix + task id list.
    Returns:
        index:    (N, latent_dim) binary float32 array
        task_ids: list of N task ids (filenames without .json)
    """
    task_ids, embeddings = [], []
    for task_file in sorted(Path(task_dir).glob("*.json")):
        emb = embed_task(str(task_file), model, device)
        embeddings.append(emb)
        task_ids.append(task_file.stem)
    index = np.stack(embeddings, axis=0)   # (N, latent_dim)
    return index, task_ids


def hamming_retrieve(
    query_path: str,
    index: np.ndarray,
    task_ids: list[str],
    model: ARCAutoEncoder,
    device: str,
    k: int = 5,
) -> list[tuple[str, int]]:
    """
    Retrieve top-k most similar tasks to the query task.
    Returns list of (task_id, hamming_distance) sorted ascending.
    """
    q = embed_task(query_path, model, device)                     # (latent_dim,)
    # Hamming distance: number of differing bits
    dists = (index != q[None, :]).sum(axis=1).astype(int)         # (N,)
    top_k_idx = np.argsort(dists)[:k]
    return [(task_ids[i], int(dists[i])) for i in top_k_idx]


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # --- Step 1: preprocess ---
    # Point these at your ARC-1 and ARC-2 training folders
    DATA_DIRS = [
        "data/arc1/training",
        "data/arc2/training",
    ]
    PREPROCESSED_DIR = "data/preprocessed"

    preprocess_and_save(DATA_DIRS, PREPROCESSED_DIR, augment=True, n_aug=5)

    # --- Step 2: train ---
    model = train(
        preprocessed_dir=PREPROCESSED_DIR,
        n_epochs=50,
        batch_size=64,
        lr=1e-3,
        latent_dim=128,
    )

    # --- Step 3: build retrieval index over training tasks ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load("arc_ae.pt", map_location=DEVICE))
    model = model.to(DEVICE)

    index, task_ids = build_index("data/arc1/training", model, DEVICE)
    np.save("task_index.npy", index)
    with open("task_ids.json", "w") as f:
        json.dump(task_ids, f)

    # --- Step 4: retrieve for a test task ---
    results = hamming_retrieve(
        query_path="data/arc1/evaluation/some_task.json",
        index=index,
        task_ids=task_ids,
        model=model,
        device=DEVICE,
        k=5,
    )
    print("Top 5 similar tasks:")
    for tid, dist in results:
        print(f"  {tid}  hamming={dist}")