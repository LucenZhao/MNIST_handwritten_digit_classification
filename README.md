# MNIST_handwritten_digit_classification

This is the code for MATH4432 Project 1: Multi-Class Classification on MNIST Database. For details of this project, please refer to this report.  
![#f03c15](**The current version of code is modified after deadline. In commits after deadline, I just wrote the README.md file and separated codes into different files to make it more readable.**)`#f03c15`  
![#f03c15]For the version before deadline, please [click here](https://github.com/LucenZhao/MNIST_handwritten_digit_classification/tree/71374ac3ddf85072b848ccd2011ba29cdfb681bb).`#f03c15`


## Requirements
* Python 3.5.4
* NumPy 1.12.1
* scikit-learn 0.19.1
* scikit-image 0.13.1
* SciPy 1.0.0
* OpenCV 3.3.1
* Matplotlib 2.2.0

## Download Data
The MNIST database can be downloaded from the following links:  
Info: [https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.info.txt](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.info.txt)  
Training (1.7M): [https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.info.txt)  
Test (429K): [https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz](https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz)  
Please **decompress** the data before run the code.  
**Please do not use data from other sources because the data I used is different from the original MNIST database.**

## Produce Results
**To produce my final results, simply run this command:**
```
python main.py
```

To produce other results, you can modify the **config.py** file.
* Model selection: set `select_model = True`
* Feature selection: set `select_feature = True`
* Produce final results: set `produce_results = True`
* Visualize wrongly-classified images: set `visualize_error = True` (only works when producing final results)
* Draw ROC curves for LDA model: set `draw_ROC = True`

You could change the features by modifying the feature settings in **config.py**. The settings for my final results are shown below:
```python
plain = True
pool = {'take': False, 'class': 'max'}
hist = {'take': False, 'h': [4], 'w': [4]}
grad = {'take': True, 'class': 'hist'}
chain = {'take': True, 'class': 'hist'}
```
  
The results are saved in the output **results.txt** file. 