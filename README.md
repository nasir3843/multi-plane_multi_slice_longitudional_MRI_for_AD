## Multi-Plane Multi-Slice Longitudinal MRI for Deep Ensemble Progression Detection based on Enhanced Residual Multi-Head Self-Attention 

##This is a Keras implementation of the study conducted for the early detection of progression from MCI to PMCI/AD.

Please refer to the requirements.txt file to install all necessary libraries for running this code.

## Training:
Run the following command to train any desired CNN backbone for feature extraction with a Bayesian optimized classification head. You can also use a CBAM module as an intermediate attention module by specifying a flag in the given command.
```

python train.py --cnnbb [EfficientNet, ResNet, ConvNext, DenseNet, XceptionNet] EfficientNet --ch [mlp, lstm, multihead_attention] multihead_attention --cbam [True, False] --data_dir "path\to\root\folder" 

```


