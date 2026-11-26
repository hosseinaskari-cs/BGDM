# Third-party notices

The following notices apply to the third-party code and adaptations identified here.

| Component | Origin | License |
| --- | --- | --- |
| Guided diffusion U-Net and numerical helpers in `bgdm/_guided/` | [OpenAI Guided Diffusion](https://github.com/openai/guided-diffusion/tree/22e0df8183507e13a7813f8d38d51b072ca1e67c), built on [OpenAI Improved Diffusion](https://github.com/openai/improved-diffusion/tree/1bc7bbbdc414d83d4abf2ad8cc1446dc36c4e4d5) | [MIT, Copyright 2021 OpenAI](licenses/OpenAI-MIT.txt) |
| Attention pooling in the guided diffusion U-Net | [OpenAI CLIP](https://github.com/openai/CLIP/blob/d05afc436d78f1c48dc0dbf8e5980a9d471f35f6/clip/model.py), as credited in OpenAI Guided Diffusion | [MIT, Copyright 2021 OpenAI](licenses/OpenAI-MIT.txt) |
| Diffusion sampling routines adapted through the original BGDM implementation | [Denoising Diffusion Null-Space Model (DDNM)](https://github.com/wyhuai/DDNM/tree/00b58eac7843a4c99114fd8fa42da7aa2b6808af) | [MIT, Copyright 2023 Yinhuai Wang](licenses/DDNM-MIT.txt) |

The guided diffusion U-Net also credits [Jonathan Ho's TensorFlow DDPM implementation](https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py) for the attention block design.

## Research acknowledgments

- [Diffusion Posterior Sampling (DPS)](https://github.com/DPS2022/diffusion-posterior-sampling)
- [Score-based medical imaging inverse problems (ScoreMed)](https://github.com/yang-song/score_inverse_problems)
- [Score-based diffusion models for accelerated MRI (Score-MRI)](https://github.com/hyungjin-chung/score-MRI)
- [RePaint](https://github.com/andreas128/RePaint)

The notices above apply to software. Checkpoint and dataset terms are set by their respective providers.
