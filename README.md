# HW_final
这是一个关于日常作业及二崔思考的代码仓；纪念一下2021-2022年这个痛不欲生、拔苗助长的交接期；顺便摸清这个GitHub的门道，看吧，我总是能将相关系数为0的两个变量硬生生牵拉在一起，美其名曰“一箭双雕”

此时我的世界只有：选题，写报告，跑代码，编编，编不下去就ctrl c和v；然后顶着double肿的脸看尽票圈的幸福人生和繁华世界；

奥对，还有入职后一堆东西要学，本以为是带有傲慢与偏见光环的主角，后知后觉，小丑竟是我自己，整个团队都是大佬，菜鸡只有默默打工的份，关键上一周干了个啥，啥也没干，各种平台都没看，啥也不懂，人家干数据分析的好歹还知道取数的步骤和平台，还能有模有样地讲出来，我只有一些现在不觉得以后会认为极度幼稚和无知的发言。

想着1.9号之后可能会舒服点吧

一定是2021年干的缺德事儿和打脸的事儿太多了，以至于肿成这个鬼样；2022，让二崔慢慢赎罪吧，救赎自己，救赎他人，我看长大这件事对我来说挺难的，那也别难为自己了。
----2021尾末作业写不完刚拔完智齿脸double肿的二崔


## 统计基础-经典线性回归模型对房价的预测及改进
### house price prediction.ipynb
数据来源于https://archive.ics.uci.edu/ml/datasets.php 上的2012-2013年台湾省台北市及其周边的一组房价数据，数据集包含414条房价信息，每条信息由“交易日期”、“房龄”、“到MRT站（地铁站）的距离”、“便利店数目”、“纬度”、“经度”、“单位平方房价”7个因子组成，经初步探索性分析判断该数据集近似符合线性回归模型的基本假设，将清洗、归一化处理后的数据集，输入经典的回归模型：多元线性回归、岭回归、稳健回归、度为2、3的多项式回归模型。通过比较可决系数、预测标准差、综合评价指标Evalue 检验模型的优劣性，发现
随着模型复杂度的提高，在训练集上的拟合性越好，而在预测集上的误差就越大。为在拟合优度与预测准度上寻求平衡，本文调用PolynomialFeature函数构建degree=3的解释变量组合全集，以拟合优度检验系数为依据通过逐步回归选择最优的自变量组合，经验证相较一般线性模型，本复合模型准确度上提高了18.5%，综合指标也在所有模型中表现最优，可用于样本量>> 自变量数目的房
价数据集的预测。从不同角度考虑，本模型可为开发商选址定价、大众买房居住或投资提供决策参考。

### 二崔说
gang真，都是老掉牙的统计模型，推荐两个很值得的库，statsmodel（经典统计模型）和sklearn(机器学习)；这个作业一方面我是想嗯...完成作业，另外一方面好好了解一下回归模型的拓展，毕竟多高大上的模型，下游任务也就那几类：回归，分类，聚类；毕竟多那个啥的数据分析报告，也还是逃不了数据清洗-数据预处理-描述性统计分析-数据可视化-再处理-建模-得出结论-瞎编乱造的基本框架；比较有价值的点或许就在模型求解过程中优化算法的选取与创新，可惜，只有这方面的解释我看不懂，api也不会调，调了后数据跑不出来，千辛万苦跑出来后，发现，诶，还不如直接最小二乘来的痛快。一个作业，快把所有合适的不合适的线性回归模型全套进去了，最后搞了个【多元+逐步】的模型，自认为还可，实际工程上并没什么用

## 疫情时代国际贸易数据可视化 
### International trade visualization.ipynb

基于2015-2020年各国之间的往来贸易数据，本文以“新冠肺炎疫情对进出口贸易的影响”为分析目的，以Year与Value两列数据为主线，着重探索数据集各特征上的交易值在不同年份的表现，主要分析工具为数据可视化，根据数据类别与大小调用合适的图表依次对各变量进行数据可视化，结合可视化结果给出在疫情常态化趋势下，如何保持国与国之间的正常贸易往来。

### 二崔说
不知我可爱的助教是从哪儿找的数据，这个数据初看哇时间序列（可算找到一个序列型数据练练手了），可视化以后，god，怎么会有长这个模样的时间序列图，这不就是我那个紊乱的心电图叠加叠加再叠加嘛；再看，老老实实搞个条形图或者数据分布或者箱线图之类的吧，胡乱搞出来几个，着重夸一下dataframe.groupby这个函数，我直呼这类数据的救星，问题来了，文字部分怎么解决，这几张图只是为了出生而出生，没有什么意义，好吧，只好ctrl+c和v。进出口贸易，白皮书上我觉得用excel分析的很到位了，数据量不大的话，就不要互相折磨了

## 编程作业
### customer-segmentation.ipynb
从Kaggle 上选择customer-segmentation 数据集，旨在通过顾客的行为特征进行定位分类。基于选取的数据集，在pandas框架下对类别特征数据进行编码、归一化处理，将处理后的数据集输入经典的逻辑回归、决策树、等分类模型，调用sklearn 中的GridSearchCV函数进行Xgboost
与lightgbm超参数的寻优。基于优化后的模型，创建customer 类，属性为客户的特征，通过异常构造函数的参数检查，若不合理，则触发异常拒绝生成对象，通过定义分类方法实现客户类别群的预判。该类将优化后的分类模型封装在一起，实现了客户类别预测的功能

详情见链接：https://github.com/Rabbicui/customer-segmentation

##pytorch自建简单多项式回归网络
### pytorch编写多项式回归-五折交叉验证-earlystop方法.ipynb
1. 使用模型随机生成 100个数据点(x, y)，要求x均匀地分布在[-1,1]内。然后把所有样本随机等分成训练集和测试集。
请训练n次多项式回归模型，用于给定x时预测y，分别对n=1，2，3，8。对不同的n，画出训练误差和测试误差随着训练迭代步数而变化的曲线。训练算法使用基于梯度下降的Adam算法，使用默认参数，并迭代5000 步。

2. 观察上题中是否出现过拟合，并用5折交叉验证法找出最合适的模型，即最优的n。并最终在测试集上报告测试误差。

3. 再生成50个点作为验证集，对于n=8，使用earlystopping方法得到一个多项式回归模型，用于解决过拟合问题。并最终在测试集上报告测试误差。Earlystopping 方法是找到验证集误差达到最小点时的迭代步，并将此刻的模型作为最终模型。

##全连接网络实现经典的手写数字识别，基于Keras
### 作业2_1.ipynb
从网站https://www.kaggle.com/c/digit-recognizer/overview 了解手写体字符数据集，利用训练数据train.csv 训练一个多层全连接前馈神经网络+Softmax 分类器的模型，在测试数据test.csv上预测，参考sample_submission.csv 的格式，在网站上提交你的预测结果，记录得分。注意数据特征表示的是0-255 的像素值，可先将数据标准化至[0,1]。若计算资源有限，可考虑先进行PCA 降维（可选）。
## xgboost实现预测人口收入分类
### 作业2_2.ipynb
 从网站https://archive.ics.uci.edu/ml/datasets/Adult 下载人口收入调查数据集。
1) 了解此数据集；
2) 准备数据，对特征进行预处理；
3) 在adult.data 训练集上训练任何你在本课学过的模型，并想办法选择合适的超参数；
4) 在adult.data 训练集和adult.test 测试集上都计算分类准确率和F1 score，并尽可能提高模型在测试集上的表现。


