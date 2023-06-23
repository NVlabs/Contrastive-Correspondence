# [Learning Contrastive Representation for Semantic Correspondence](https://arxiv.org/abs/2109.10967)

Taihong Xiao, Sifei Liu, Shalini De Mello, Zhiding Yu, Jan Kautz, Ming-Hsuan Yang

## Data Preparation

You can download the required datasets using the following links and put them into the directory `Datasets_SCOT`.

- [PF-PASCAL](https://drive.google.com/open?id=1OOwpGzJnTsFXYh-YffMQ9XKM_Kl_zdzg)
- [PF-WILLOW](https://drive.google.com/open?id=1tDP0y8RO5s45L-vqnortRaieiWENQco_)
- [SPair-71k](https://drive.google.com/open?id=1KSvB0k2zXA06ojWNvFjBv0Ake426Y76k)

Also, we need to prepare the ImageNet dataset.


## Environment

- pytorch==1.5.1
- torchvision==0.6.1
- opencv-contrib-python
- scipy==1.2.1
- scikit-image
- pandas
- requests
- gluoncv-torch

## Pretrained Models

We here provide pretrained models of ImageNet. You may download the pretrained models using the following command.

```bash
mkdir pretrained_models
cd pretrained_models
wget http://vllab1.ucmerced.edu/~taihong/ContrastiveCorrespondence/pretrained_models/moco.pth.tar
```

## Training, Validation and Testing

Please refer the `training` and `validation_and_test` directory for detailed instructions.
