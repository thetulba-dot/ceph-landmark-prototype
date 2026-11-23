# V4 visualization + metrics dump. Reuses train4.py loader/model code VERBATIM (comments trimmed) so the pipeline is identical.
# Selection rule (deterministic, stated in output): sort 150 valid cases by per-case MRE ascending, take ranks [0, 37, 74, 112, 149] = min, Q1, median, Q3, max.
# evaluate.py - Aggregate validation metrics for a trained landmark checkpoint.
# Metrics are in the resized 256x256 pixel analysis frame, NOT millimeters.
# The validation split was used iteratively during development: these are
# development metrics, not independent evidence.
#
# Usage: python evaluate.py DATA_ROOT --weights PATH
# DATA_ROOT must contain the Aariz 'valid' split folder.

import glob, json, os, argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

SIZE = 256
NL = 29
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ap = argparse.ArgumentParser()
ap.add_argument("data_root")
ap.add_argument("--weights", required=True)
args = ap.parse_args()
ROOT = args.data_root

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
    imgs, pts, stems = [], [], []
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
        stems.append(os.path.splitext(f)[0])
    X = torch.from_numpy(np.stack(imgs))
    Y = torch.from_numpy(np.stack(pts))
    print(mode, "loaded:", X.shape, flush=True)
    return X, Y, stems


Xva, Yva, stems = load_split(ROOT, "valid")

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
model.load_state_dict(torch.load(args.weights, map_location=DEV))
model.eval()
with torch.no_grad():
    pxs = []
    for b in range(0, Xva.shape[0], 32):
        pv = model(Xva[b:b+32].to(DEV))
        idx = pv.view(pv.shape[0], NL, -1).argmax(2)
        pxs.append(torch.stack([(idx % 256).float(), (idx // 256).float()], 2).cpu())
        del pv
    px = torch.cat(pxs)
D = (px - Yva).norm(dim=2)
d_case = D.mean(1).numpy()
print("cases:", len(d_case))
print("per-case MRE px mean/std/p25/p50/p75/p90: %.2f %.2f %.2f %.2f %.2f %.2f" % (
    d_case.mean(), d_case.std(), *np.percentile(d_case, [25, 50, 75, 90])))
print("within2px: %.1f%% within4px: %.1f%% within10px: %.1f%%" % (
    (D<=2).float().mean().item()*100, (D<=4).float().mean().item()*100, (D<=10).float().mean().item()*100))
pl = D.mean(0).numpy()
print("per-landmark mean err px (channel index = Aariz landmark array order):")
for k in range(NL):
    print("  LM%02d %.2f" % (k, pl[k]))
print("DONE")
