# train.py - Train a ResNet-18 heatmap model for cephalometric landmark detection
# on the Aariz dataset (29 landmarks, CC BY 4.0, https://doi.org/10.6084/m9.figshare.27986417.v1).
#
# Honesty rules:
# - All metrics are in the resized 256x256 pixel analysis frame, NOT millimeters.
# - The validation split was used iteratively during development, so its metrics
#   are development metrics, not independent evidence.
# - The test split is never touched by this script.
#
# Usage:
#   python train.py DATA_ROOT [--epochs N] [--init-weights PATH] [--out PATH]
# DATA_ROOT must contain the Aariz 'train' and 'valid' split folders.
# --init-weights optionally warm-starts from a previously saved checkpoint.
import glob, json, os, sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

torch.manual_seed(42); np.random.seed(42)
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("data_root")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--init-weights", default=None, help="optional checkpoint to warm-start from")
ap.add_argument("--out", default="./aariz_resnet.pt", help="where to save the best checkpoint")
args = ap.parse_args()
ROOT = args.data_root
EPOCHS = args.epochs
SIZE = 256
NL = 29
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, flush=True)

def find_split(root, name):
    cands = glob.glob(os.path.join(root, "**", name), recursive=True)
    for c in cands:
        if os.path.isdir(os.path.join(c, "Cephalograms")):
            return c
    cands = glob.glob(os.path.join(root, "*", name))
    for c in cands:
        if os.path.isdir(os.path.join(c, "Cephalograms")):
            return c
    raise SystemExit("split not found: " + name + " under " + root)

def load_split(root, mode):
    base = find_split(root, mode)
    img_dir = os.path.join(base, "Cephalograms")
    ann_dir = os.path.join(base, "Annotations", "Cephalometric Landmarks")
    senior = os.path.join(ann_dir, "Senior Orthodontists")
    junior = os.path.join(ann_dir, "Junior Orthodontists")
    imgs, pts = [], []
    for f in sorted(os.listdir(img_dir)):
        stem = os.path.splitext(f)[0] + ".json"
        sp = os.path.join(senior, stem)
        jp = os.path.join(junior, stem)
        if not (os.path.exists(sp) and os.path.exists(jp)):
            continue
        im = Image.open(os.path.join(img_dir, f)).convert("L")
        w, h = im.size
        im = im.resize((SIZE, SIZE))
        def rd(p):
            d = json.load(open(p))
            return np.array([[l["value"]["x"], l["value"]["y"]] for l in d["landmarks"]], dtype=np.float32)
        s, j = rd(sp), rd(jp)
        pt = np.ceil(0.5 * (s + j)).astype(np.float32)
        pt[:, 0] *= SIZE / w
        pt[:, 1] *= SIZE / h
        if pt.shape[0] != NL:
            continue
        imgs.append(np.array(im, dtype=np.float32)[None] / 255.0)
        pts.append(pt)
    X = torch.from_numpy(np.stack(imgs))
    Y = torch.from_numpy(np.stack(pts))
    print(mode, "loaded:", X.shape, flush=True)
    return X, Y

Xtr, Ytr = load_split(ROOT, "Train")
Xva, Yva = load_split(ROOT, "Valid")

def heatmaps(points, size=SIZE, sigma=4.0):
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    ys, xs = ys.to(points.device), xs.to(points.device)
    out = torch.zeros((points.shape[0], NL, size, size), device=points.device)
    for i, p in enumerate(points):
        out[i] = torch.exp(-((xs.unsqueeze(0) - p[:, 0].view(-1, 1, 1)) ** 2 + (ys.unsqueeze(0) - p[:, 1].view(-1, 1, 1)) ** 2) / (2 * sigma ** 2))
    return out.cpu()

print("building heatmaps...", flush=True)
# heatmaps are generated per batch to keep memory bounded
Hva = heatmaps(Yva)

mean_pt = Ytr.mean(0, keepdim=True)
md = (mean_pt - Yva).norm(dim=2)
md_case = md.mean(1)
print("MEAN-BASELINE valid MRE_px: %.2f within2px: %.1f%% within10px: %.1f%%" % (md.mean().item(), (md<=2).float().mean().item()*100, (md<=10).float().mean().item()*100), flush=True)
print("MEAN-BASELINE per-case mean/median: %.2f %.2f" % (md_case.mean(), md_case.median()), flush=True)

import torchvision
bb = torchvision.models.resnet18(weights=None)
bb.fc = nn.Identity()

def up(cin, cout):
    return nn.Sequential(nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.ReLU(inplace=True))

class ResNetHeat(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.stem = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.l1, self.l2, self.l3, self.l4 = bb.layer1, bb.layer2, bb.layer3, bb.layer4
        self.dec = nn.Sequential(up(512,256), up(256,128), up(128,64), up(64,32), up(32,16))
        self.head = nn.Conv2d(16, NL, 1)
        self.norm = torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    def forward(self, x):
        x = self.norm(x.repeat(1,3,1,1))
        f = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        return self.head(self.dec(f))

model = ResNetHeat(bb).to(DEV)
if args.init_weights:
    model.load_state_dict(torch.load(args.init_weights, map_location=DEV))
    print("warm-started weights from", args.init_weights, flush=True)
else:
    print("training from scratch (random init)", flush=True)
print("params(M): %.1f" % (sum(p.numel() for p in model.parameters())/1e6,), flush=True)

def wloss(pred, target, alpha=99.0):
    w = 1.0 + alpha * target
    return (w * (pred - target) ** 2).mean()

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
B = 16
ntr = Xtr.shape[0]
best_mre = 1e9; best_state = None; best_ep = -1
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(ntr)
    tot = 0.0
    for i in range(0, ntr, B):
        idx = perm[i:i+B]
        xb, hb = Xtr[idx].to(DEV), heatmaps(Ytr[idx]).to(DEV)
        loss = wloss(model(xb), hb)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(idx)
    model.eval()
    with torch.no_grad():
        pxs = []
        for b in range(0, Xva.shape[0], 32):
            pv = model(Xva[b:b+32].to(DEV))
            Bv, L, Hh, Ww = pv.shape
            idx = pv.view(Bv, L, -1).argmax(2)
            pxs.append(torch.stack([(idx % Ww).float(), (idx // Ww).float()], 2).cpu())
            del pv
        px = torch.cat(pxs)
        dc = (px - Yva).norm(dim=2)
        mre = dc.mean().item()
        SYMS=["A","ANS","B","Me","N","Or","Pog","PNS","Pn","R","S","Ar","Co","Gn","Go","Po","LPM","LIT","LMT","UPM","UIA","UIT","UMT","LIA","Li","Ls","N`","Pog`","Sn"]
        severe = {c: (dc[:,c].mean().item(), (dc[:,c]>20).sum().item()) for c in (7,28,10,23,11,8)}
    sev = " ".join(f"{SYMS[c]}:{v[0]:.1f}px/{v[1]}>20" for c,v in severe.items())
    print(f"epoch {ep+1}/{EPOCHS} train_loss={tot/ntr:.5f} val_MRE_px={mre:.2f} | {sev}", flush=True)
    if mre < best_mre:
        best_mre, best_ep = mre, ep + 1
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

torch.save(best_state if best_state is not None else model.state_dict(), args.out)
print(f"SAVED {args.out} (best epoch {best_ep}, best val_MRE_px {best_mre:.2f})", flush=True)

model.load_state_dict(best_state); model.eval()
with torch.no_grad():
    pxs = []
    for b in range(0, Xva.shape[0], 32):
        pv = model(Xva[b:b+32].to(DEV))
        idx = pv.view(pv.shape[0], NL, -1).argmax(2)
        pxs.append(torch.stack([(idx % 256).float(), (idx // 256).float()], 2).cpu())
        del pv
    px = torch.cat(pxs)
    D = (px - Yva).norm(dim=2)
d_case = D.mean(1)
print("valid MRE mean/std/p25/p50/p75/p90: %.2f %.2f %.2f %.2f %.2f %.2f" % (
    d_case.mean(), d_case.std(), *np.percentile(d_case.numpy(), [25, 50, 75, 90])), flush=True)
print("valid within2px: %.1f%% within4px: %.1f%% within10px: %.1f%%" % (
    (D<=2).float().mean().item()*100, (D<=4).float().mean().item()*100, (D<=10).float().mean().item()*100), flush=True)
wins = (d_case < md_case).float().mean().item()*100
print("beats landmark-mean baseline per-case: %.1f%% of %d validation cases (baseline mean 18.11 px)" % (wins, len(d_case)), flush=True)
pl_base = md.mean(0); pl_model = D.mean(0)
for k in range(NL):
    print("LM%02d baseline %.1f model %.1f" % (k, pl_base[k].item(), pl_model[k].item()), flush=True)
print("DONE", flush=True)
