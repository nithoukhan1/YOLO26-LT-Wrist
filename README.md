# YOLO26-LT-Wrist

YOLO26 with Long-Tail Adaptations for Pediatric Wrist Fracture Detection on GRAZPEDWRI-DX.

## Project Overview

This repository contains the implementation of YOLO26-LT, an enhanced version of Ultralytics YOLO26 (Jan 2026) targeting the long-tailed class distribution problem in pediatric wrist fracture detection. Built on the canonical Ju et al. patient-level split of the GRAZPEDWRI-DX dataset.

## Novel Contributions (planned)

1. First application of YOLO26's NMS-free architecture to GRAZPEDWRI-DX
2. Tau-normalization at inference for long-tail class confidence calibration
3. LDAM-DRW class-margin loss for rare-class learning
4. Multi-seed rigorous evaluation (seeds 42, 1234, 2024)

## Base Code

Built on Ultralytics YOLO (https://github.com/ultralytics/ultralytics), commit reference: [add later].

## Dataset

GRAZPEDWRI-DX with Ju et al. patient-level split (6,091 patients, 14,204/4,094/2,029 train/val/test).
Dataset: https://figshare.com/articles/dataset/GRAZPEDWRI-DX/14825193
Split: https://ruiyangju.github.io/GRAZPEDWRI-DX_JU/

## Status

In development. See `experiments/` for run logs.
