Getting Started for ML Beginners
=============================
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

![Iris setosa](https://www.tensorflow.org/images/iris_three_species.jpg)

从左到右分别是 Iris setosa，Iris virginica， Iris versicolor

幸运的是已经有人利用花萼和花瓣的测量数据创建了[120种鸢尾花数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)，这些数据集已经成为了介绍机器学习分类算法的规范介绍之一。（ [MNIST database](https://en.wikipedia.org/wiki/MNIST_database)数据集，包含手写字体识别数据，与另一个非常著名的问题相关），数据的前五行看起来像这样：

|花萼长度|花萼宽度|花朵长度|花朵宽度|花种代号|
|-------|--------|--------|------|--------|
|6.4     |2.8      |5.6      |2.2    |2       |
|5.0     |2.3      |3.3      |2.2    |2       |
|6.4     |2.8      |4.5      |2.2    |2       |
|6.4     |2.8      |1.7      |2.2    |2       |

让我们介绍一些专有名词：

* 最后一列(花种代号)称为标记([label](https://developers.google.com/machine-learning/glossary/#label))，第一列被称为特征([featurs](https://developers.google.com/machine-learning/glossary/#feature)),**特征**是样例的属性的一种，而标记正式我们需要预测的值

* 一个样例由一种花的多种属性和标记组成，预测的表格中展示了从120种样例中选出的5种

每个标记都是一串字符(例如：“setosa”)，但是机器学习算法依赖于数值，因此，有人将这些字符标记转化成了相应的数字值，这里有一些表示方法的例子：

* 0 代表 setosa
* 1 代表 versicolor
* 2 代表 virginica

Models and training(模型与训练)
------------------------------
模型描述的是一种实体属性和标签间的联系。对于鸢尾花这个问题而言，这种模型定义了鸢尾花种类和花瓣及花萼测量数据值间的关系，一些简单的模型可以利用几行代数关系是来表示，更加复杂的学习模型可以包含大量难以用数学方式总结的交错的公式和参数（涉及到公式和算法数量达到一定级别后，内部调度的复杂关系很难从整体上处理，例如非监督型神经网络的价值判别过程）

在不使用机器学习的的方式下如何区分鸢尾花四种属性和花种类间的关系？你能够使用传统的编程技巧（例如成堆的条件声明）创建一个模型？你可以花足够多的时间去定义一个确定花萼和花瓣与鸢尾花种类的关系的方法，然而，一个好的机器学习算法专门为你定制一个模型。如果你能够提供足够多的具有代表性的、符合模型类型的关系数据，机器学习算法就能够确定花萼，花瓣与鸢尾花种类的关系。

训练是模型不断被优化（算法程序对数据进行学习）的过程，鸢尾花的分类问题是一个典型的[监督性学习](https://developers.google.com/machine-learning/glossary/#supervised_machine_learning)的例子，模型根据带有**标签**的数据不断训练优化，然而在[非监督机器学习](https://developers.googl.com/machine-learning/glossary/#unsupervised_machine_learning),训练数据是不带有标签的，反之，模型需要自己从属性数据中寻找特定的关系。

