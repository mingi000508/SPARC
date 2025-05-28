# Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models

This is the official implementation of the paper:
**"Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models."**

## Abstract

Detailed image captioning is essential for tasks like data generation and aiding visually impaired individuals. High-quality captions require a balance between precision and recall, which remains challenging for current multimodal large language models (MLLMs). In this work, we hypothesize that this limitation stems from weakening and increasingly noisy visual attention as responses lengthen. To address this issue, we propose SPARC (Selective Progressive Attention ReCalibration), a training-free method that enhances the contribution of visual tokens during decoding. SPARC is founded on three key observations: (1) increasing the influence of all visual tokens reduces recall; thus, SPARC selectively amplifies visual tokens; (2) as captions lengthen, visual attention becomes noisier, so SPARC identifies critical visual tokens by leveraging attention differences across time steps; (3) as visual attention gradually weakens, SPARC reinforces it to preserve its influence. Our experiments, incorporating both automated and human evaluations, demonstrate that existing methods improve the precision of MLLMs at the cost of recall. In contrast, our proposed method enhances both precision and recall with minimal computational overhead.

## Installation


```bash
# Create a conda environment
conda create -n sparc python=3.10 -y
conda activate sparc

# LLaVA installation
cd LLaVA
pip install --upgrade pip  # Enables PEP 660 support
pip install -e .

# install additional
cd ..
pip install -r requirements.txt
```
## Evaluation

### IIW-400 Dataset (CLAIR Evaluation)
1. Download the dataset from [IIW-400](https://google.github.io/imageinwords/).
2. Edit the dataset path in `script/llava/captioning_iiw400.sh`.
3. Run caption generation:
   ```bash
   bash scripts/llava/captioning_iiw400.sh
   ```
4. Update the dataset path in `script/clair_iiw_eval.sh`.
5. Run CLAIR evaluation:
   ```bash
   bash scripts/clair_iiw_eval.sh
   ```

---

### DOCCI Dataset (CLAIR Evaluation)
1. Download the dataset from [DOCCI](https://google.github.io/docci/).
2. Edit the dataset path in `script/llava/captioning_docci.sh`.
3. Run caption generation:
   ```bash
   bash scripts/llava/captioning_docci.sh
   ```
4. Update the dataset path in `script/clair_docci_eval.sh`.
5. Run CLAIR evaluation:
   ```bash
   bash scripts/clair_docci_eval.sh
   ```

---

### CHAIR Evaluation
1. Download the COCO 2014 validation images and annotation files.
2. Edit the dataset path in `script/llava/captioning_coco.sh`.
3. Run caption generation:
   ```bash
   bash scripts/llava/captioning_coco.sh
   ```
4. Update the dataset path in `script/chair_eval.sh`.
5. Run CHAIR evaluation:
   ```bash
   bash scripts/chair_eval.sh
    ```


## Acknowledgement

- This project is based on the [LLaVA](https://github.com/haotian-liu/LLaVA) codebase.
- Evaluation codes are based on [CHAIR](https://github.com/Maxlinn/CHAIR-metric-standalone), [CLAIR](https://github.com/DavidMChan/clair)

## Citation

If you find this work useful for your research, please cite:


```bibtex
@misc{jung2025visualattentionfadesselective,
  title={Visual Attention Never Fades: Selective Progressive Attention ReCalibration for Detailed Image Captioning in Multimodal Large Language Models},
  author={Mingi Jung and Saehuyng Lee and Eunji Kim and Sungroh Yoon},
  year={2025},
  eprint={2502.01419},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2502.01419}
}
```