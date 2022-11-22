# CRCNet

##  Install dependencies

* torch
* torchvision 
* opencv
* Albumentations

## DataSets
kvasir-seg：https://datasets.simula.no/kvasir-seg/

CVC-ClinicDB：http://www.cvc.uab.es/CVC-Colon/index.php/databases/

##  Usage

####  1. Training

```bash
python train.py --dataset "dataset-name" --batch_size batch-size --load_ckpt "/path-to-check-point" --epoch_start epoch-start
```



####  2. Inference

```bash
python test.py --dataset "dataset-name" --batch_size 1 --load_ckpt "/path-to-check-point"
```
