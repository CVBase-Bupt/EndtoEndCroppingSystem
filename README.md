# End-to-End Cropping System

This is an offical implemenation for 
**An End-to-End Neural Network for Image Cropping by Learning Composition from Aesthetic Photos**.

Given a source image, our algorithm could take actions step by step to find almost the best cropping window on source image.

## Get Start
Install the python libraries. (See Requirements).

Download the code from GitHub:
```
git clone https://github.com/CVBase-Bupt/EndtoEndCroppingSystem.git

cd EndtoEndCroppingSystem
```

Run the python script:
```
python demo.py [your image path]
```
Before you run, please download our pre-trained models.We have released 6 models based on different scale (224,384,512) and ratio (square or not). If you want to use any of them, just:


link：https://pan.baidu.com/s/11m4mNhUdFUlTThRXDL7XLA 
password：vzwb 

Put the weight file under the dirctory ```weights```

Fix the config file ```models/config.py```, change the ```self.ratio``` and ```self.scale``` in the  ```__init__```function. 

## Requirement

Python

keras(we use version 2.2.4)

tensorflow 1.13.1

opencv-python 2.4.9

## Performance
On FLMS:


 |   model    | IOU | BDE | 
 |    :-:     | :-: | :-: | 
 | model__224 | 0.846 | 0.026 |
 | model__384 | 0.844 | 0.027 |
 | model__512 | 0.845 | 0.028 |
 | model_square_224 | 0.840 | 0.028 |
 | model_square_384 | 0.843 | 0.028 |
 | model_square_512 | 0.842 | 0.028 |
 
 
 On CUHK-ICD:
 
 
 
 |   model    | IOU | BDE | IOU | BDE | IOU | BDE |
 |    :-:     | :-: | :-: | :-: | :-: | :-: | :-: |
 | model__224 | 0.822 | 0.032 | 0.815 | 0.034 | 0.803 | 0.036 |
 | model__384 | 0.823 | 0.032 | 0.818 | 0.034 | 0.804 | 0.036 |
 | model__512 | 0.825 | 0.032 | 0.820 | 0.034 | 0.806 | 0.036 |
 | model_square_224 | 0.825 | 0.032 | 0.818 | 0.034 | 0.805 | 0.036 |
 | model_square_384 | 0.827 | 0.032 | 0.817 | 0.034 | 0.804 | 0.036 |
 | model_square_512 | 0.828 | 0.032 | 0.822 | 0.034 | 0.806 | 0.036 |
 
 
 On FCD:
 
 
  |  model    | IOU | BDE | 
 |    :-:     | :-: | :-: | 
 | model__224 | 0.673 | 0.058 |
 | model__384 | 0.670 | 0.059 |
 | model__512 | 0.664 | 0.060 |
 | model_square_224 | 0.672 | 0.059 |
 | model_square_384 | 0.670 | 0.059 |
 | model_square_512 | 0.665 | 0.061 |
 
 
