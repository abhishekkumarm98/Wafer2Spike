# Wafer2Spike
# ITC2024 : Wafer2Spike: Spiking Neural Network for Wafer Map Pattern Classification

## Envrionment

* Python 3.10.12
* Numpy 1.23.5
* Pandas 1.5.3
* Torch 2.1.0+cu118
* Sklearn 1.2.2
* GPU : Tesla T4


## To run


### To install the necessary modules

```
pip install -r requirements.txt
```

### To download WM-811k dataset, [click here][1]

### To generate the results (Augmentation will take rougly 20 minutes)


#### Split ratio = 8:2
```
python main.py --splitRatio '8:2' --epoch 13 --modelType "Wafer2Spike_4C"
python main.py --splitRatio '8:2' --epoch 14 --modelType "Wafer2Spike_3C"
python main.py --splitRatio '8:2' --epoch 14 --modelType "Wafer2Spike_2C"
```

#### Split ratio = 8:1:1
```
python main.py --splitRatio '8:1:1' --epoch 12 --modelType "Wafer2Spike_4C"
python main.py --splitRatio '8:1:1' --epoch 11 --modelType "Wafer2Spike_3C"
python main.py --splitRatio '8:1:1' --epoch 10 --modelType "Wafer2Spike_2C"
```

#### Split ratio = 7:3
```
python main.py --splitRatio '7:3' --epoch 12 --modelType "Wafer2Spike_4C"
python main.py --splitRatio '7:3' --epoch 11 --modelType "Wafer2Spike_3C"
python main.py --splitRatio '7:3' --epoch 12 --modelType "Wafer2Spike_2C"
```

#### Split ratio = 6:1:3
```
python main.py --splitRatio '6:1:3' --epoch 12 --modelType "Wafer2Spike_4C"
python main.py --splitRatio '6:1:3' --epoch 10 --modelType "Wafer2Spike_3C"
python main.py --splitRatio '6:1:3' --epoch 12 --modelType "Wafer2Spike_2C"
```


## Citation

If you find our work useful, please cite our paper.
```
@INPROCEEDINGS{10766711,
  author={Mishra, Abhishek and Kumar, Suman and Lingamoorthy, Anush and Das, Anup and Kandasamy, Nagarajan},
  booktitle={2024 IEEE International Test Conference (ITC)}, 
  title={Wafer2Spike: Spiking Neural Network for Wafer Map Pattern Classification}, 
  year={2024},
  volume={},
  number={},
  pages={16-20},
  keywords={Integrated circuit synthesis;Accuracy;Pattern classification;Spiking neural networks;Computer architecture;Benchmark testing;Computational efficiency;Manufacturing;Open source software;Wafer map pattern classification;spiking neural networks;neuromorphic computing},
  doi={10.1109/ITC51657.2024.00011}}

```






[1]: https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
