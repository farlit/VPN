### VPN: Visual Prompt Navigation

#### Shuo Feng, Zihan Wang, Yuchen Li, Rui Kong, Hengyi Cai, Shuaiqiang Wang, Gim Hee Lee, Piji Li, Shuqiang Jiang

This repository is the official implementation of **[VPN: Visual Prompt Navigation](https://arxiv.org/pdf/2508.01766).**

>While natural language is commonly used to guide embodied agents, the inherent ambiguity and verbosity of language often hinder the effectiveness of language-guided navigation in complex environments. To this end, we propose Visual Prompt Navigation (VPN), a novel paradigm that guides agents to navigate using only user-provided visual prompts within 2D top-view maps. This visual prompt primarily focuses on marking the visual navigation trajectory on a top-down view of a scene, offering intuitive and spatially grounded guidance without relying on language instructions. It is more friendly for non-expert users and reduces interpretive ambiguity. We build VPN tasks in both discrete and continuous navigation settings, constructing two new datasets, R2R-VP and R2R-CE-VP, by extending existing R2R and R2R-CE episodes with corresponding visual prompts. Furthermore, we introduce VPNet, a dedicated baseline network to handle the VPN tasks, with two data augmentation strategies: view-level augmentation (altering initial headings and prompt orientations) and trajectory-level augmentation (incorporating diverse trajectories from large-scale 3D scenes), to enhance navigation performance. Extensive experiments evaluate how visual prompt forms, top-view map formats, and data augmentation strategies affect the performance of visual prompt navigation.

## Requirements for VPN

1. Install Matterport3D simulator and Python Environment for `R2R-VP`: follow instructions [here](https://github.com/cshizhe/VLN-DUET).

2. Download annotations, preprocessed features, trained models and preprocessing code from [Baidu Netdisk](https://pan.baidu.com/s/11hLnDKq3uvg_ni5fWvY5MA?pwd=rznf) (You should the folder "datasets" in "VPN/").

3. Training & Evaluation for R2R-VP:
```setup
conda activate vlnduet
cd map_nav_src
bash scripts/run_r2r.sh 
```

## Requirements for VPN-CE

1. Install Habitat simulator and Python Environment for `R2R-CE-VP`: follow instructions [here](https://github.com/MarSaKi/ETPNav).

2. Download annotations, preprocessed features, trained models and preprocessing code from [Baidu Netdisk](https://pan.baidu.com/s/1Y7ACK9By8DcEY5y4F0TyOQ?pwd=qg3m) (You should the folder "data" in "VPN/VPN_CE/").

3. Training & Evaluation for R2R-CE-VP:
```setup
conda activate vlnce
cd VPN_CE
CUDA_VISIBLE_DEVICES=0,1 bash run_r2r/main.bash train 2333  # training
CUDA_VISIBLE_DEVICES=0,1 bash run_r2r/main.bash eval  2333  # evaluation
```

## Citation
If you find some useful for your work, please consider citing our paper:
```bibtex
Feng S, Wang Z, Li Y, et al. VPN: Visual Prompt Navigation[J]. arXiv preprint arXiv:2508.01766, 2025.
  ```
Besides, if you 

## Contact
Feel free to contact Shuo Feng via email fengshuo@nuaa.edu.cn for more support.

## Acknowledgments
Our code is based on [VLN-DUET](https://github.com/cshizhe/VLN-DUET), [ETPNav](https://github.com/MarSaKi/ETPNav) and [ScaleVLN](https://github.com/wz0919/ScaleVLN). Thanks for their great works!

