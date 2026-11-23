# ceph-landmark-prototype

Research prototype for automatic cephalometric landmark detection on lateral cephalograms.

A ResNet-18 backbone with a transposed-convolution decoder predicts one heatmap per landmark (29 landmarks), trained on the public Aariz Cephalometric Dataset. This is an early research prototype, not a medical device, and it is not validated for clinical or diagnostic use.

## Results (development metrics)

Measured on the 150-case Aariz validation split, in the 256x256 resized analysis frame (pixels, NOT millimeters):

- Mean radial error 4.12 px (median 3.38 px, p90 8.19 px)
- 42.8% of placements within 2 px, 81.1% within 4 px, 96.9% within 10 px
- Per-landmark mean error is at or below 9.7 px for all landmarks except dataset symbol "Sn" (15.0 px); next worst are symbols "LIA" (9.7 px) and "Go" (7.6 px)

The validation split was used iteratively during development, so these numbers are development metrics, not independent evidence. The Aariz test split was never used.

Known limitations: 256x256 analysis resolution (no millimeter calibration), one architecture/configuration family tried so far, small single-source dataset, and a visible failure tail on the worst cases. The model's error in millimeters on full-resolution films has not been established here.

## Data

Training used the Aariz Cephalometric Dataset (CC BY 4.0), Khalid et al. 2025, https://doi.org/10.6084/m9.figshare.27986417.v1. No images, annotations, or other dataset files are redistributed in this repository; download the dataset from figshare and point the scripts at it.

## Usage

```
pip install -r requirements.txt
python train.py /path/to/Aariz --epochs 40 --out ./aariz_resnet.pt
python evaluate.py /path/to/Aariz --weights ./aariz_resnet.pt
```

`train.py` trains from scratch by default; `--init-weights PATH` warm-starts from a saved checkpoint. Checkpoints are saved as plain PyTorch state_dicts. GPU recommended for training; evaluation runs on CPU.

## License

Code in this repository is MIT licensed (see LICENSE). The Aariz dataset remains under its own CC BY 4.0 license with attribution to its authors.
