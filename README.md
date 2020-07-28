# Mutual Information Maximization for Effective Lip Reading
## Introduction
Code and model for paper [Mutual Information Maximization for Effective Lip Reading](https://arxiv.org/abs/2003.06439). 

Some codes of this respository is based on [mpc001's](https://github.com/mpc001/end-to-end-lipreading) & [T. Stafylakis and G. Tzimiropoulos's](https://github.com/tstafylakis/Lipreading-ResNet) implementations and thanks for their inspiring works.
## Dependencies
* python 3.5 
* pytorch 1.0.0 
* opencv-python 4.1

## Dataset
- **LRW**
> The results obtained with the proposed model on the [LRW dataset](https://github.com/mpc001/end-to-end-lipreading). The coordinates for cropping mouth ROI are suggested as (x1, y1, x2, y2) = (80, 116, 175, 211) in Matlab. Please note that the fixed cropping mouth ROI (F x H x W) = [:, 115:211, 79:175] in python.

- **LRW-1000**
[LRW-1000 dataset](http://vipl.ict.ac.cn/view_database.php?id=14) had cropped the mouth ROI, we directly sent them to the model.

##Training
In order to better illustrate our proposed GLMIM, we trained the Baseline firstly, then the LMIM was applied to the Baseline. Finally, the GMIM was applied to them. We trained the model with the LRW  as an example.
- **Baseline (LRW/Baseline/)**
The configurations has been explained with annotations throughly at the top of `main.py`. After filling your own confugurations at each stage, you can start training the model with
```
python main.py
```  

- **Baseline + LMIM (/LRW/Baseline_LMIM)**
After training the Baseline, please reload the Baseline model and train it with the LMIM simultaneously
- **Baseline + LMIM +GMIM (/LRW/Baseline_GLMIM)**
Reloading the two models you obtained at previous stage, training the total models.

The GLMIM will be dropped after training.

**Tips**
- To obtain better results, you could decay the learing rate while the accuracy is not on the increase.
- After training several epochs, testing the model a few times during each epoch rather than testing the model at the end of each epoch.

##Models
Models are availbale at [GooleDrive](https://drive.google.com/drive/folders/1injmbeusVXCEHQUftRhosfb3aBtE5qGg?usp=sharing). To evaluate the model, set `path` as the <font color='green'>file location</font> and `test` as <font color='#6060B1'>True</font>:
```
parser.add_argument('--path', default=r'file/location', type=str, help='path to Baseline, empty for training')
parser.add_argument('--test', default=True, action='store_false', help='perform on the test phase')
```
## Citation
```bibtex
@article{zhao2020mutual,
  title={Mutual Information Maximization for Effective Lip Reading},
  author={Zhao, Xing and Yang, Shuang and Shan, Shiguang and Chen, Xilin},
  journal={arXiv preprint arXiv:2003.06439},
  year={2020}
}
```

