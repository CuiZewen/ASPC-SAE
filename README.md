# DEC-SAE
Deep Embedded Clustering with Sparse Autoencoder (DEC-SAE)
Based on [ASPC-DA](https://github.com/CuiZewen/ASPC-DA)  ASPC-DA:  acc(fmnist) = 0.60 acc(mnist) = 0.98  
my model DEC-DA acc(fmnist) = 0.90339  acc(mnist) = 0.97553


两天没有睡觉了，大致写一下，明天起来再继续完善(前天从linux换成win10，很多地方还没完善)
I haven't slept for two days, so I will write it down roughly, and I will continue to improve it tomorrow when I get up (I just want to sleep now).
（目前我还不会使用git更新项目，第一次用，不知道咋样）



conda create -n aspc python=3.6 -y

conda activate aspc  # Windows


pip install tensorflow-gpu==1.10 scikit-learn h5py munkres pydotplus  graphviz

git clone https://github.com/CuiZewen/DEC-SAE.git DEC-SAE

cd DEC-SAE

python run_exp.py


