# FSRCNN

This repository is implementation of the ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367).

<center><img src="./thumbnails/fig1.png"></center>

## Differences from the original

- Added the zero-padding
- Used the Adam instead of the SGD

## Requirements

- PyTorch 1.0.0
- Numpy 1.15.4
- Pillow 5.4.1
- h5py 2.8.0
- tqdm 4.30.0
- Skimage

## Train

The 91-image, Set5 dataset converted to HDF5 can be downloaded from the links below.

| Dataset | Scale | Type | Link |
|---------|-------|------|------|
| 91-image | 2 | Train | [Download](https://www.dropbox.com/s/01z95js39kgw1qv/91-image_x2.h5?dl=0) |
| 91-image | 3 | Train | [Download](https://www.dropbox.com/s/qx4swlt2j7u4twr/91-image_x3.h5?dl=0) |
| 91-image | 4 | Train | [Download](https://www.dropbox.com/s/vobvi2nlymtvezb/91-image_x4.h5?dl=0) |
| Set5 | 2 | Eval | [Download](https://www.dropbox.com/s/4kzqmtqzzo29l1x/Set5_x2.h5?dl=0) |
| Set5 | 3 | Eval | [Download](https://www.dropbox.com/s/kyhbhyc5a0qcgnp/Set5_x3.h5?dl=0) |
| Set5 | 4 | Eval | [Download](https://www.dropbox.com/s/ihtv1acd48cof14/Set5_x4.h5?dl=0) |

Otherwise, you can use `prepare.py` to create custom dataset.

```bash
python train.py --train-file "BLAH_BLAH/91-image_x3.h5" \
                --eval-file "BLAH_BLAH/Set5_x3.h5" \
                --outputs-dir "BLAH_BLAH/outputs" \
                --scale 3 \
                --lr 1e-3 \
                --batch-size 16 \
                --num-epochs 20 \
                --num-workers 8 \
                --seed 123                
```

## Test

Pre-trained weights can be downloaded from the links below.

| Model | Scale | Link |
|-------|-------|------|
| FSRCNN(56,12,4) | 2 | [Download](https://www.dropbox.com/s/1k3dker6g7hz76s/fsrcnn_x2.pth?dl=0) |
| FSRCNN(56,12,4) | 3 | [Download](https://www.dropbox.com/s/pm1ed2nyboulz5z/fsrcnn_x3.pth?dl=0) |
| FSRCNN(56,12,4) | 4 | [Download](https://www.dropbox.com/s/vsvumpopupdpmmu/fsrcnn_x4.pth?dl=0) |

The results are stored in the same path as the query image.

```bash
python test.py --weights-file "BLAH_BLAH/fsrcnn_x3.pth" \
               --image-file "data/butterfly_GT.bmp" \
               --scale 3
```

## Results

PSNR was calculated on the Y channel.

### Set5

| Eval. Mat | Scale | Paper | Ours (91-image) |
|-----------|-------|-------|-----------------|
| PSNR | 2 | 36.94 | 37.12 |
| PSNR | 3 | 33.06 | 33.22 |
| PSNR | 4 | 30.55 | 30.50 |

<table>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3 (34.66 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/lenna.bmp""></center>
    	</td>
    	<td>
    		<center><img src="./data/lenna_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/lenna_fsrcnn_x3.bmp"></center>
    	</td>
    </tr>
    <tr>
        <td><center>Original</center></td>
        <td><center>BICUBIC x3</center></td>
        <td><center>FSRCNN x3 (28.55 dB)</center></td>
    </tr>
    <tr>
    	<td>
    		<center><img src="./data/butterfly_GT.bmp""></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_bicubic_x3.bmp"></center>
    	</td>
    	<td>
    		<center><img src="./data/butterfly_GT_fsrcnn_x3.bmp"></center>
    	</td>
    </tr>
</table>
