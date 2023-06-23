# Instruction for Validation and Testing

## Validation

### Prepare the validation set

In order to conduct the beamsearch without using the validation ground truth, we employ the cycle loss
as the validation performance indicator. The cycle loss requires the augmentated images from the first
image similar to what we do in the training. Different from the online augmentation during traiing, we
need to perform augmentations offline to obtain permanent augmented images. To this end, we can run

```
python create_val_images_labels.py \
    --dataset pfpascal \
    --split val \
    --arch resnet50 \
    --modelpath ../pretrained_models/moco.pth.tar \
    --downscale 16 \
    --patch_size 128 \
    --rotate 10 \
    --moco_dim 128 \
    --temp 0.0007 \
    --gpu 0 \
    --save_dir ./crop/
```
Then we obtain a folder named `crop` containing augmented images with the pseudo correpsondence.
We need to move the folder into the PF-PASCAL dataset directory. Note that we can do similarly thing
to the other two datasets by replacing the `--dataset pfpascal` to `--dataset pfwillow` or
`--dataset spair`.

Alternatively, you may download the augmented images processed by us using the following links.

- [PF-PASCAL](http://vllab1.ucmerced.edu/~taihong/ContrastiveCorrespondence/datasets_crop/PF-PASCAL-crop.tar)
- [PF-WILLOW](http://vllab1.ucmerced.edu/~taihong/ContrastiveCorrespondence/datasets_crop/PF-WILLOW-crop.tar)
- [SPair-71k](http://vllab1.ucmerced.edu/~taihong/ContrastiveCorrespondence/datasets_crop/SPair-71k-crop.tar)

Then, untar each file separately into the directory of each dataset. The resulting directory looks like the
following, where the `crop` folders contain processed images.
```
├── PF-PASCAL
│   ├── Annotations
│   ├── crop
│   ├── html
│   ├── index.html
│   ├── JPEGImages
│   ├── parsePascalVOC.mat
│   ├── ShowMatchingPairs
│   ├── test_pairs.csv
│   ├── trn_pairs.csv
│   ├── val_images
│   └── val_pairs.csv
├── PF-WILLOW
│   ├── car(G)
│   ├── car(M)
│   ├── car(S)
│   ├── crop
│   ├── duck(S)
│   ├── motorbike(G)
│   ├── motorbike(M)
│   ├── motorbike(S)
│   ├── test_pairs.csv
│   ├── winebottle(M)
│   ├── winebottle(wC)
│   └── winebottle(woC)
└── SPair-71k
    ├── crop
    ├── devkit
    ├── ImageAnnotation
    ├── JPEGImages
    ├── Layout
    ├── PairAnnotation
    ├── README
    └── Segmentation
```


### Beam Search

#### Beam Search on PF-PASCAL

We can perform beam search without ground truth correspondence. Take the PF-PASCAL dataset as an
example, the following command will output `[2,12,13,15]` as the best hyperpixel layer combination.

```
python beamsearch_loss.py \
    --dataset pfpascal \
    --backbone resnet50 \
    --modelpath ../pretrained_models/pf_pascal.pth.tar \
    --num_classes 128 \
    --thres img \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 0
```
Here are the detailed explanation of different options for `--sim`
- `cos`: raw similarity
- `cosGeo`: using regularized Hough matching (RHM) only as the post-processing step
- `OT`: applying the optimal transport (OT) as the post-processing step
- `OTGeo`: employing OT and RHM


For comparison, we can also run beam search using the validation ground truth correspondence,
which would outputs `[2,11,12,15]` as the best hyperpixel layer combination.
```
python beamsearch.py \
    --dataset pfpascal \
    --backbone resnet50 \
    --modelpath ../pretrained_models/pf_pascal.pth.tar \
    --num_classes 128 \
    --thres img \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 0
```

#### Beam Search on PF-WILLOW

Similarly, we can perform beam search on PF-WILLOW.
```
python beamsearch_loss.py \
    --dataset pfwillow \
    --backbone resnet50 \
    --modelpath ../pretrained_models/pf_willow.pth.tar \
    --num_classes 128 \
    --thres bbox \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 0
```

#### Beam Search on SPair-71k

Similarly, we can perform beam search on SPair-71k.
```
python beamsearch_loss.py \
    --dataset spair \
    --backbone resnet50 \
    --modelpath ../pretrained_models/spair_71k.pth.tar \
    --num_classes 128 \
    --thres bbox \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 0.5 \
    --classmap 0
```


## Testing

### Testing on PF-PASCAL

Given the best layer combination `[2,12,13,15]` by beam search above, we can run the testing as following:
```
python evaluate_map_CAM.py \
    --dataset pfpascal \
    --thres img \
    --backbone resnet50 \
    --num_classes 128 \
    --modelpath ../pretrained_models/pf_pascal.pth.tar \
    --hyperpixel '(2,12,13,15)' \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 0.5 \
    --eps 0.05 \
    --gpu 0 \
    --classmap 1 \
    --split test \
    --alpha 0.05
```
Note that the `--alpha 0.05` could be changed to other thresholds (e.g., 0.1, 0.15).

We can also add `--vis_dir /path/to/vis_dir` to save correspondence visualization results.

### Testing on PF-WILLOW

Given the best layer combination `[2,11,12,13,14]` by beam search above, we can run the testing as following:
```
python evaluate_map_CAM.py \
    --dataset pfwillow \
    --thres bbox \
    --backbone resnet50 \
    --num_classes 128 \
    --modelpath ../pretrained_models/pf_willow.pth.tar \
    --hyperpixel '(2,11,12,13,14)' \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 0.5 \
    --eps 0.05 \
    --gpu 0 \
    --classmap 1 \
    --split test \
    --alpha 0.05
```

### Testing on SPair-71k

Given the best layer combination `[3,5,8,14]` by beam search above, we can run the testing as following:
```
python evaluate_map_CAM.py \
    --dataset spair \
    --thres bbox \
    --backbone resnet50 \
    --num_classes 128 \
    --modelpath ../pretrained_models/spair_71k.pth.tar \
    --hyperpixel '(3,5,8,14)' \
    --sim OTGeo \
    --exp1 1.0 \
    --exp2 1.0 \
    --eps 0.05 \
    --gpu 0 \
    --classmap 1 \
    --split test \
    --alpha 0.05
```

