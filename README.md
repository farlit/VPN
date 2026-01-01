### VPN: Visual Prompt Navigation

#### Shuo Feng, Zihan Wang, Yuchen Li, Rui Kong, Hengyi Cai, Shuaiqiang Wang, Gim Hee Lee, Piji Li, Shuqiang Jiang

This repository is the official implementation of **[VPN: Visual Prompt Navigation](https://arxiv.org/pdf/2508.01766).**

>While natural language is commonly used to guide embodied agents, the inherent ambiguity and verbosity of language often hinder the effectiveness of language-guided navigation in complex environments. To this end, we propose Visual Prompt Navigation (VPN), a novel paradigm that guides agents to navigate using only user-provided visual prompts within 2D top-view maps. This visual prompt primarily focuses on marking the visual navigation trajectory on a top-down view of a scene, offering intuitive and spatially grounded guidance without relying on language instructions. It is more friendly for non-expert users and reduces interpretive ambiguity. We build VPN tasks in both discrete and continuous navigation settings, constructing two new datasets, R2R-VP and R2R-CE-VP, by extending existing R2R and R2R-CE episodes with corresponding visual prompts. Furthermore, we introduce VPNet, a dedicated baseline network to handle the VPN tasks, with two data augmentation strategies: view-level augmentation (altering initial headings and prompt orientations) and trajectory-level augmentation (incorporating diverse trajectories from large-scale 3D scenes), to enhance navigation performance. Extensive experiments evaluate how visual prompt forms, top-view map formats, and data augmentation strategies affect the performance of visual prompt navigation.

## Requirements for VPN

1. Install Matterport3D simulator and Python Environment for `R2R-VP`: follow instructions [here](https://github.com/cshizhe/VLN-DUET).

2. Download annotations, preprocessed features, trained models and preprocessing code from [Baidu Netdisk](https://pan.baidu.com/s/11hLnDKq3uvg_ni5fWvY5MA?pwd=rznf).

You should the folder "datasets" in "VPN/".


## Requirements for VPN-CE

1. Install Matterport3D simulator for `R2R-VP`: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```

2. Install requirements:
```setup
conda create --name GridMM python=3.8.0
conda activate GridMM
pip install -r requirements.txt
```

3. Download annotations, preprocessed features, and trained models from [Baidu Netdisk](https://pan.baidu.com/s/1jRshMRNAhIx4VtCT0Lw1DA?pwd=beya).

4. Install Habitat simulator for `R2R-CE`: follow instructions [here](https://github.com/YicongHong/Discrete-Continuous-VLN) and [here](https://github.com/jacobkrantz/VLN-CE).
