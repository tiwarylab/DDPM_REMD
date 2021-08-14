# DDPM_REMD
Denoising diffusion probabilistic models for replica exchange


Implementation of <a href="https://arxiv.org/abs/2006.11239">Denoising Diffusion Probabilistic Model</a> in Pytorch.  This implementation was transcribed from the official Tensorflow version <a href="https://github.com/hojonathanho/diffusion">here</a> and a modified Pytorch version <a href="https://github.com/lucidrains/denoising-diffusion-pytorch">here</a>. 


##Package requirement
```bash
pip3 install einops
pip3 install pillow
pip3 install torchvision
pip3 install tqdm
```

##Usage
To train the model, you can run the `run_training.py` after modifying the parameters insde the script.
```bash
python run_training.py
```
To use the trained model to generate figures, you can run the `gen_fig.py` after modifying the parameters insde the script.
```bash
python run_training.py
```

## Citations

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

```bibtex
@inproceedings{
    anonymous2021improved,
    title={Improved Denoising Diffusion Probabilistic Models},
    author={Anonymous},
    booktitle={Submitted to International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=-NEXDKk8gZ},
    note={under review}
}
```
