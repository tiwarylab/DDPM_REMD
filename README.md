# DDPM_REMD
Denoising diffusion probabilistic models for replica exchange MD simulation


Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> for data from replica exchange MD. This implementation was transcribed from the  Tensorflow version <a href="https://github.com/hojonathanho/diffusion">Ho, J., Jain, A. and Abbeel, P., 2020. arXiv:2006.11239.</a> and a modified Pytorch version <a href="https://github.com/lucidrains/denoising-diffusion-pytorch">here</a>. 


## Package requirement
```bash
pip3 install einops
pip3 install pillow
pip3 install torchvision
pip3 install tqdm
```

## Usage
To train the model, you can run the `run_training.py` after modifying the parameters insde the script.
```bash
python run_training.py
```
To use the trained model to generate figures, you can run the `gen_sample.py` after modifying the parameters insde the script.
```bash
python gen_sample.py
```

## Citations

```bibtex
@article{wang2021denoising,
  title={Denoising diffusion probabilistic models for replica exchange},
  author={Wang, Yihang and Tiwary, Pratyush},
  journal={arXiv preprint arXiv:2107.07369},
  year={2021}
}
```

```bibtex
@misc{ho2020denoising,
    title={Denoising Diffusion Probabilistic Models},
    author={Jonathan Ho and Ajay Jain and Pieter Abbeel},
    year={2020},
    eprint={2006.11239},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
