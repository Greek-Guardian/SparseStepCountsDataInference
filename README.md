# codes
This is the implement of MLP-GRU proposed in "Infering activity patterns from sparse step counts data with Recurrent Neural Network". The data folder includes python scripts on data preprocessing, multi-granular activity patterns labeling, and data down-sampleing. The models folder includes the implement of our model and three state-of-the-art methods.

To run our model, you need to add our datasset (http://health.sjtu.edu.cn/infer/) to data folder. Then run MLP_GRU.py to train the model. We also provide trained MLP-GRU models on three kinds of activity patterns in /models/sparsity_granularity/. You can load the trained model and infer acticity patterns of your own sparse step counts data directly like sparsity_gry.py. You can also change the parameters in multi_granular_labeling.py to construct the activity patterns you want, and retrain the model with your own activity patterns.

