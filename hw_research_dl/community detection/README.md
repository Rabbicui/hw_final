## graph algorithm applied in community detection
This REPORT firstly summarizes the graph algorithm according to algorithm function. As an important application scenario of graph algorithms, community detection has developed over a long period of time from traditional detection techniques such as clustering to deep neural networks; general community detection methods can be roughly divided into four categories: traditional clustering, The splitting algorithm of edge betweenness, the aggregation algorithm based on modularity optimization, the unsupervised clustering based on graph representation learning and neural network, in order to explore the applicability of the above methods in various scenarios, this paper constructs and Three data sets from simple to complex are selected, and preliminary conclusions are drawn: the unsupervised clustering algorithm based on deep learning performs well on all data sets, especially when the number of communities is small and the community discrimination is relatively high. On low graph datasets; relatively speaking, the division effect of algorithms such as louvain is not as good as graph representation learning + clustering; it can be seen that there is still a long way to go for the application of graph neural networks in community detection.

## env
python---3.+

tensorflow--1.+

pip install community

