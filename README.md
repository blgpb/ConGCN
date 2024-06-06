# Hyperspectral Image Classification With Contrastive Graph Convolutional Network

Official Code Repository for our paper - Hyperspectral Image Classification With Contrastive Graph Convolutional Network

## Requirement
- Python 3.8.11
- TensorFlow-gpu 2.6.0
- numpy 1.19.5
- pillow 9.0.1
- networkx 2.6.3
- scipy 1.7.1


## Parameter description

- `-r`: the number of repetitions of the experiment (_i.e._, 10)
- `-t`: the name of dataset (_e.g._, ‘IP’)


## Run
Follow the command line to run the experiment.

```Python
$ python auto.py
```

# Citation
If our work is helpful to you, please cite our work. Thank you very much!

```
@ARTICLE{10032180,
  author={Yu, Wentao and Wan, Sheng and Li, Guangyu and Yang, Jian and Gong, Chen},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Hyperspectral Image Classification With Contrastive Graph Convolutional Network}, 
  year={2023},
  volume={61},
  number={},
  pages={1-15},
  keywords={Convolutional neural networks;Convolution;Hyperspectral imaging;Representation learning;Adaptation models;Generative adversarial networks;Training;Contrastive learning;graph augmentation;graph convolutional network (GCN);hyperspectral image (HSI) classification},
  doi={10.1109/TGRS.2023.3240721}}
```