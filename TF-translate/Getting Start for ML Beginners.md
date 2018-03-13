#Getting Started for ML Beginners
这篇文章将讲述如何使用机器学习算法区分不同花的种类。这篇文章深入介绍了如何TensorFlow代码来达到我们想要的效果。
如果不符合下列条件，那么你真的来对地方了
* 对机器学习了解甚少
* 你想学习如何使用机器学习编程
* 你能够使用Python编程（至少会一点）

花种分类问题
-----------------------------
想象一下，假如你是一位植物学家，并且想要寻找一种权威的方法来区分鸢尾花。机器学习算法提供了许多种给花分类的方法。例如，一个复杂的机器学习程序可以基于图像对花朵进行分类。我们也许并不需要如此的效果，至少能够基于花的[萼片](https://en.wikipedia.org/wiki/Sepal)和[花瓣](https://en.wikipedia.org/wiki/Petal)区分是否是鸢尾花就够了。

然而鸢尾花约有300多种，但是我们只需要区分以下三种花就足够了
* Iris setosa (鸢尾花的一种)
* Iris virginica  (鸢尾花的一种)
* Iris versicolor (鸢尾花的一种)

![Iris setosa](https://www.tensorflow.org/images/iris_three_species.jpg) ![Iris virginica](https://www.tensorflow.org/images/iris_three_species.jpg) ![Iris versicolor](https://www.tensorflow.org/images/iris_three_species.jpg)

从左到右分别是 Iris setosa，Iris virginica， Iris versicolor

幸运的是已经有人利用花萼和花瓣的测量数据创建了[120种鸢尾花数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)，这些数据集已经成为了介绍机器学习分类算法的规范介绍之一。（ [MNIST database](https://en.wikipedia.org/wiki/MNIST_database)数据集，包含手写字体识别数据，与另一个非常著名的问题相关），数据的前五行看起来像这样：

|花萼长度|花萼宽度|花朵长度|花朵宽度|花种代号|
|-------|--------|--------|------|--------|
|6.4     |2.8      |5.6      |2.2    |2       |
|5.0     |2.3      |3.3      |2.2    |2       |
|6.4     |2.8      |4.5      |2.2    |2       |
|6.4     |2.8      |1.7      |2.2    |2       |

让我们介

