# ECCV2022: Enhanced Accuracy and Robustness via Multi-Teacher Adversarial Distillation
The Offical Code of ECCV2022: [Enhanced Accuracy and Robustness via Multi-Teacher Adversarial Distillation](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640577.pdf)

by Shiji Zhao, Jie Yu, Zhenlong Sun, Bo Zhang, Xingxing Wei.

For the teacher model, the [WideResNet-34-10](https://drive.google.com/file/d/10sHvaXhTNZGz618QmD5gSOAjO3rMzV33/view) TRADES-pretrained and [WideResNet-70-16](https://github.com/deepmind/deepmind-research/tree/master/adversarial_robustness) model is also following the [RSLAD](https://github.com/zibojia/RSLAD). The clean teacher model checkpoint is [here](https://drive.google.com/file/d/1i9PdN-Nt10Ckhaj3KqFrEE9SICSIEYCe/view?usp=drive_link).

Our checkpoint is [here](https://drive.google.com/file/d/1QIdqSLAgTXiWmC_HAF-AylnYcRam1x4n/view?usp=drive_link).

### the running environment

```bash
python=3.8 
pytorch=1.6
cuda = 11.3
numpy=1.19
```

### training resnet18 on cifar10:

```bash
python mtard_resnet18_cifar10.py
```

### training resnet18 on cifar100:

```bash
python mtard_resnet18_cifar100.py
```


### Citation

```bash
@inproceedings{Zhao2022Enhanced,
title={Enhanced Accuracy and Robustness via Multi-Teacher Adversarial Distillation},
author={Shiji Zhao and Jie Yu and Zhenlong Sun and Bo Zhang and Xingxing Wei},
booktitle={European Conference on Computer Vision},
year={2022},
}
```
