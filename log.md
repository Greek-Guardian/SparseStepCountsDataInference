## 问题

已经能够用数值方法分类，为什么还要训练网络来分类

mlp-gru和gru-d有什么异同？

为什概率和不为1

接下来要干什么

## TODO:

1. 尝试一下补全的效果
2. 对于补全任务，会不会倾向于阶梯式补全（围绕loss值补全）
3. 在无完整数据集场景下的稀疏数据（训练集也是稀疏的，或者说没有label，label很难确定）
4. 步数“模式"的定义
5. 以一个人为“模式"研究的对象
6. 以人的模式的规律性来研究（周中vs周末）

## 学习记录

### 数据缺失类型：

1.Missing Completely at Random(MCAR)。完全随机的缺失。
2.Missing at Random(MAR)。数据的缺失依赖于某个完全变量（完全变量：已知的、不包含缺失值的变量）
3.Missing Not at Random(MNAR)。数据的缺失依赖于某个未知变量

MCAR可以根据其出现情况删除缺失值的数据，同时，随机缺失可以通过已知变量对缺失值进行估计。
在MNAR下，删除包含缺失值的数据可能会导致模型出现偏差，同时，对数据进行填充也需要格外谨慎。

### [高斯过程：](https://blog.csdn.net/ting_qifengl/article/details/121535264#:~:text=%E9%AB%98%E6%96%AF%E8%BF%87%E7%A8%8B%EF%BC%88Gaussian%20Processes%2C,GP%EF%BC%89%E6%98%AF%E6%A6%82%E7%8E%87%E8%AE%BA%E5%92%8C%E6%95%B0%E7%90%86%E7%BB%9F%E8%AE%A1%E4%B8%AD%E9%9A%8F%E6%9C%BA%E8%BF%87%E7%A8%8B%E7%9A%84%E4%B8%80%E7%A7%8D%EF%BC%8C%E6%98%AF%E5%A4%9A%E5%85%83%20%E9%AB%98%E6%96%AF%E5%88%86%E5%B8%83%20%E7%9A%84%E6%89%A9%E5%B1%95%EF%BC%8C%E8%A2%AB%E5%BA%94%E7%94%A8%E4%BA%8E%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E3%80%81%E4%BF%A1%E5%8F%B7%E5%A4%84%E7%90%86%E7%AD%89%E9%A2%86%E5%9F%9F%E3%80%82)
