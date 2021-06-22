# DER.ClassIL.Pytorch
This repo is the official implementation of [DER: Dynamically Expandable Representation for Class Incremental Learning
](https://arxiv.org/abs/2103.16788) (CVPR 2021)



#### Dataset
* ImageNet100
Refer to [ImageNet100_Split](https://github.com/arthurdouillard/incremental_learning.pytorch/tree/master/imagenet_split)

## Training
* Change to corresponding directory and run the following commands
```
sh scripts/run.sh
```

## Inference
Inference command:
```
sh scripts/inference.sh
```





## Tips
* create a new exp folder
```
rsync -rv --exclude=tensorboard --exclude=logs --exclude=ckpts --exclude=__pycache__ --exclude=results --exclude=inbox ./codes/base/ ./exps/der_womask/10steps/trial0
```


## Acknowledgement
Thanks for the great code base from https://github.com/arthurdouillard/incremental_learning.pytorch.




## Citation
If you are using the DER in your research or with to refer to the baseline results published in this repo, please use the following BibTex entry.
```
@article{yan2021dynamically,
  title={DER: Dynamically Expandable Representation for Class Incremental Learning},
  author={Yan, Shipeng and Xie, Jiangwei and He, Xuming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```