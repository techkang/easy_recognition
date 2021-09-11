# Easy-Recognizer

## Introduce
This is a framework that is user-friendly for text recognition task.

## Feature
- automatic resize all images to the minimum shape in a batch `data/dataloader.py:resize_collector`
- three level configuration, from default to user-defined: `config/default.py` -> `experiment/*.yaml` -> `command line args`
- tensorboard support for visualize acc/loss curve and inference result
- fast lmdb packing `data/scripts/dataset_to_lmdb.py`
- lmdb visualize tools with Flask `data/scripts/lmdb_vis.py`
- reimplement of bezier align and bezier control point fit in ABCNet `tools/bezier.py`
- default lr scheduler with warmup and cosine learning rate decay `lr_scheduler/build_lr_scheduler.py`

## Usage
```shell
git clone ...
cd easy_recognizer
pip install -r requirements.txt
python train.py --config-file experiment/base/crnn.yaml num_gpus 4 dataloader.batch_size 12
```