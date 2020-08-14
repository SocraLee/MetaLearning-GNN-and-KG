# 1. Papers

这里归档了小组成员阅读过的各个文章



### Graph Attention Networks

作者受到attention机制的启发，将attention机制思想应用到图卷积网络（GCN）中

这里的attention机制是指：为每个节点的邻居节点分配权重，从而关注作用较大的节点，忽略作用较小的节点

阅读参考：

https://blog.csdn.net/LIYUO94/article/details/105187349

https://www.jianshu.com/p/8078bf1711e7


### Efficient Estimation of Word Representations in Vector Space

文章写作目的：试图从数以十亿计单词的巨大数据集学习高质量的单词向量

文章主要提出了两个计算模型：New Log-linear Models

                     1.Continuous Bag-of-Words Model；
                     训练复杂度：Q = N × D + D × log 2 (V ). (N x D是映射层矩阵，V是输出层维度)
                     
                     2.Continuous Skip-gram Model；
                     训练复杂度：Q = C × (D + D × log 2 (V )). (C为最大词距，V是输出层维度)
                     
NNLM和RNNLM中最大复杂度来自于非线性隐层相关的计算。

于是本文第一个模型CBOW就去掉了Feedforward NNLM的非线性隐层，所有词共享映射层，形成一个词袋。

第二个模型是另一个概念，是根据另外一个词来预测同句中这个词的类别。

结果不错。

阅读参考：

https://www.cnblogs.com/shuzirank/p/6519888.html

https://www.jianshu.com/p/b838a3e7f94b
                   
