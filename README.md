# **繁體中文場景文字辨識競賽－初階：場景文字檢測**

# **教育部全國大專校院人工智慧競賽(AI CUP)-機器閱讀紀錄-課程挑戰賽**

[AI CUP Competition](https://tbrain.trendmicro.com.tw/Competitions/Details/13)

Team Name : 報名系統一致\
Team Member : 吳亦振 (Wu, Yi-Chen)\
Final Rank : 10th/341

## `Run Scripts`

#### Data download and unzip

Please download data from [aicup](https://tbrain.trendmicro.com.tw/Competitions/Details/13) and [ReCTS](https://rrc.cvc.uab.es/?ch=12).
Notice that ReCTS dataset need register

1. unzip aicup dataset to `./data/aicup/` folder
2. unzip ReCTS dataset to  `./data/ReCTS/` folder
3. copy (or move) images to `./dataset/train/` folder

```
$cp -a ./data/aicup/. ./dataset/train/
$cp -a ./data/ReCTS/. ./dataset/train/
```

#### Data preprocess
```
$ python preprocess.py
```

#### Train
```
$ python train.py
```

#### Eval 

Please download the model from [Resnet](https://drive.google.com/file/d/1QFdObTYPnh7dF_Xtu4O9JZCVQbDzzFB4/view?usp=sharing) and upload to `./model/` folder.
```
$ python eval.py
```
