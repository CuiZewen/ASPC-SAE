# Deep Embedded Clustering with Sparse Autoencoder (DEC-SAE)  


## Usage  

### 1. Prepare environment  
```
conda create -n aspc python=3.6 -y
source activate aspc  #Linux
# or
conda activate aspc  # Windows
```
Install required packages:  
```
pip install tensorflow-gpu==1.10 scikit-learn h5py munkres pydotplus  graphviz
```
### 2. Clone the code and prepare the datasets  
```
git clone https://github.com/CuiZewen/DEC-SAE.git DEC-SAE
cd DEC-SAE
```
### 3. Run experiments
```
python run_exp.py
```

改动部分  
1.修改了损失函数，添加了对隠层的kl散度约束，使隠层具备稀疏性，编码器修改relu为sigmoid激活函数  

2.对关键代码添加中文注释，对模型结构添加了可视化输出  
（调用了keras.utils.plot_model，用于输出模型结构，如果报错或者嫌安装麻烦可以将autoencoder的plot_out设置为False，  
最后修改keras.utils.vis_utils中的所有pydot为pydotplus，ctrl+R替换全部就好  
支持ubuntu，不支持为win0，win10配置不同）

3.对模型训练过程做了细微调整,修改优化器为Nadam，效果更好  

4.添加了模型的调用接口，可以输出并查看中间层的结果，也可调用预测  
增加了聚类标签匹配的函数用于还原聚类标签,采用了kuhn-Murunkres算法映射聚类标签  

5.根据acc保留最优权重  

6.(pip install munkres)该包实现了聚类与实际标签的映射算法 
