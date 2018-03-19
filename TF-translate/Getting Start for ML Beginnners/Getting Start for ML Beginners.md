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

Get the sample program
--------------------------------
在运行示例代码之前，首先需要做以下的工作：

1. [安装 **TensorFlow**](https://www.tensorflow.org/install/index)
2. 如果使用 **Anaconda** 或者 **virtualenv** 安装 **TensorFlow** 环境，激活它
3. 使用以下命令安装或升级你的 pandas 库
```python
  pip install pandas
```

按照以下几个步骤获取实例程序

1. 从github上利用以下命令克隆 TensorFlow 模型仓库 
```bash
 git clone https://github.com/tensorflow/models
```
2. 切换当前目录到包含示例程序和说明文档的文件夹中
```bash
 cd models/samples/core/get_started/
```
在 **get_started** 文件夹中，能够找到一个名字叫做 **premade_estimator.py** 的文件

Run the sample program(运行示例程序)
-----------------------------------
你可以像运行其他python程序那样运行我们的示例程序，在命令行中输入以下命令在运行 **premade_estimators.py** 文件
```bash
 python premade_estimator.py
```
运行程序后的输出结果中会包含以下三行：
```bash
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"
```

如果程序执行后产生了错误而非预测结果，看看整个过程中是否出现了以下的错误：
1. 是否正确安装了 **TensorFlow**
2. **TensorFlow** 的版本是否正确，正常运行示例程序要求 **TensorFlow** 的运行版本在1.4或以上
3. 如果你使用 **virtualenv**  或者 **Anaconda** 安装TensorFlow ，你是否正常激活了环境

The TensorFlow programming stack(TensorFlow 编程技术栈)
------------------------------------------------------

正如下面插图所描述的那样，TensorFlow 的API技术栈分为以下多个层
![TensorFlow 技术栈](https://www.tensorflow.org/images/tensorflow_programming_environment.png)

##TensorFlow 的程序环境

如果你开始写TensorFlow程序，我们强烈建议着重学习以下两个高级API：
* Estimators (估计)
* Datasets (数据集)

当然，我们偶尔也可以从其他的API函数中感受到封装的便利，但是这篇文档更加注重前面提到的两个API

程序
-------
感谢你的耐心，让我们深入到程序内部看一下， **premade_estimator.py** 以及其他的 TensorFlow 的流程如下：

* 导入和转化处理数据集
* 创建描述数据属性的数据列
* 选择适合的问题模型
* 带入数据集训练模型
* 评估模型的效率
* 利用训练好的模型进行数据预测

以下的几个部分对上面每个步骤做了详细的介绍。

Import and parse the data sets(导入和转化处理数据集)
---------------------------------------------------
鸢尾花的分类问题需要从以下两个 .csv 文件中导入数据

* http://download.tensorflow.org/data/iris_training.csv, 包含训练数据集。
* http://download.tensorflow.org/data/iris_test.csv,包含测试数据集。

训练数据集包含训练模型所需的数据，测试数据集则包含用来评估训练后的模型效果的数据。

训练和测试数据集都单独作为数据文件存在，之后，程序对训练数据集做了分割，大部分数据用来对模型进行训练，而剩下的小部分作为测试数据存在，增加训练数据通常来看会得到较好的模型结果，然而，增加更多的测试数据能够使我们的模型效率更高，不管怎样分割数据，测试数据样例总要从训练数据中抽选出一部分，否则你无法准确的得到模型的效率。

**premade_estimators.py** 这个程序文件依赖于 在[**iris_data.py**]() **load_data** 这个函数从训练和测试数据集中读取数据。下面是函数的部分示例

```python
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

...

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)
```

**keras** 是另一款开源机器学习库，**tf.keras** 是一个 **TensorFlow** 对于 **Keras**接口的封装，**premade_estimator.py** 这个程序仅调用了一个 tf.keras 函数; tf.keras.utils.get_file 作为一个功能上的封装，能够从远程复制 **CSV**文件到本地

针对 **laod_data** 的调用返回两个值(属性，标签)，分别对应训练和测试的数据集
```python
    # Call load_data() to parse the CSV file.
    (train_feature, train_label), (test_feature, test_label) = load_data()
```
**Pandas** 是TensorFlow依赖的另一个数据处理函数库。一个Pandas [DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)(数据表)类似于数据库的数据表，包含了数据列和表头，从 load_data 函数中返回的数据被 **DataFrame** 记录并整理， 例如：test_feature看起来像是这样
```python
    SepalLength  SepalWidth  PetalLength  PetalWidth
0           5.9         3.0          4.2         1.5
1           6.9         3.1          5.4         2.1
2           5.1         3.3          1.7         0.5
...
27          6.7         3.1          4.7         1.5
28          6.7         3.3          5.7         2.5
29          6.4         2.9          4.3         1.3
```
Describe the data
----------------
一个**feature column**(属性列)告诉你的模型如何对应每种属性的数据。在鸢尾花的分类问题中，我们想让模型能够将花萼和花瓣这两个属性和相应的数据对应我们希望模型将属性的值解释为计算机可以处理的浮点数据值，然而在其他的机器学习问题当中，我们通常不会从字面上解释数据。利用属性列解释数据仍是一个热点话题，因此我们有一个[文档](https://www.tensorflow.org/get_started/feature_columns)专门介绍它

从代码的角度来看，你通过对**tf.feature_column**模块的调用创建了一系列**feature_column**(属性列)对象，每个对象描述了一组针对模型的输入，通过对**tf.feature_column.numeric_column**的调用告知模型利用浮点数值类型处理数据。在 **premade_estimator.py**文件中，所有的四个属性都会被当作浮点型数值尽心处理，所以创建属性列的代码看起来像下面这样：

```python
    # Create feature columns for all features.
    my_feature_columns = []
    for key in train_x.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
```
下面的代码和上面的功能完全相同，只是看起来没那么简洁：
```python
my_feature_columns = [
    tf.feature_column.numeric_column(key='SepalLength'),
    tf.feature_column.numeric_column(key='SepalWidth'),
    tf.feature_column.numeric_column(key='PetalLength'),
    tf.feature_column.numeric_column(key='PetalWidth')
]
```

选择模型的种类
--------------
我们需要事先选定被训练的模型的种类。当前已经有处理多种不同问题的不同模型，每种模型都有其相对应的适合的问题。我们需要经验(大量的数学知识)来判断选择何种模型。好在我们已经选定了利用**Neural networks**神经网络来处理鸢尾花问题。[**Neural networks**](https://developers.google.com/machine-learning/glossary/#neural_network)可以从标记和对应的属性间寻找到非常复杂的关系(也许非监督学习模型的求解根本在于多维坐标系的建立和关系映射分析即转化为监督学习？)。一个神经网络是一张高度结构化的图，包含一层或多层[hidden layers](https://developers.google.com/machine-learning/glossary/#hidden_layer)(隐含层),每个隐含层包含一个或多个神经元。当今已有许多种神经网络，我们使用的是[fully connected neural network](https://developers.google.com/machine-learning/glossary/#fully_connected_layer)(全链接型神经网络)，意味着每层神经网络的输入取决于上一层神经网络的输出，下图是一个有着三个隐含层的神经网络的示例：

* 第一层隐含层包含四个神经元
* 第二层隐含层包含三个神经元
* 第三个隐含层包含两个神经元

![layer figure](https://www.tensorflow.org/images/simple_dnn.svg)
**A neural network with three hidden layers.**

为了制定一个模型的种类，需要实例化一个[Estimator](https://developers.google.com/machine-learning/glossary/#Estimators)(评价器)类， **TensorFlow** 提供了两种评价器：

* [pre-made Estimators](https://developers.google.com/machine-learning/glossary/#pre-made_Estimator),某人事先写好的评价器类

* [custom Estimators](https://developers.google.com/machine-learning/glossary/#custom_estimator)，自定义评价其，需要手动完成全部或部分功能的定义

为了实现一个神经网络，**premade_estimators.py**中的程序使用了一个叫做[tf.estimator.DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)的**pre-made Estimators**(预先定义的评价器)，这个评价器创建一个神经网络实现对样例的分类。以下代码调用了一个 DNNClassifier(深度神经网络分类器)实例：
```python
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=3)
```
使用 **hidden_units** 参数定义隐含神经元的层数和每层等神经元数量：
```python
 hidden_units=[10, 10],
```
**hidden_units**列表的长度为隐含神经元的层数(例子中为两层)，列表中每项的值为每层隐含层中包含的神经元的个数，如果你想要更换每层隐含神经元的个数，只需要改变参数的值即可。

**n_calss**参数定义了神经网络需要预测的值的维度个数，既然鸢尾花分类问题定义了三个种类，我们将n_class赋值为3

**tf.Estimator.DNNClassifier**的构造器还需要一个叫做**optimizer**(优化器)的参数,在我们的示例中并没有给他赋值，代表着我们将使用其默认值(有时为空，但是这里不是)， [**optimizer**](https://developers.google.com/machine-learning/glossary/#optimizer)(优化器)决定了如何训练我们的模型，随着对机器学习的深入，优化器和[learning rate](https://developers.google.com/machine-learning/glossary/#learning_rate)学习速率变得愈加重要。

Train the model(训练模型)
------------------------
实例化一个 **tf.Estimator.DNNClassifier** 创建了一个学习模型的框架，我们仅仅是创建了一个网络结构，但是并没有带入数据训练。我们需要调用 **Estimator** 对象的 **train** 方法。 例如：
```python
    classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, args.batch_size),
        steps=args.train_steps)
```
**steps**参数决定了 **train**函数在一系列计数后停止训练，增大 **steps** 的值会使训练时间变长。与我们的直觉相反( Counter-intuitively),长时间的训练模型并不会生成一个较好的模型。默认的**args.train_steps** 参数值为1000。训练的总步数是一个可以不断调优的[**hyperparameter**](https://developers.google.com/machine-learning/glossary/#hyperparameter)超参数。选择正确的步长大小往往需要经验和实验。

**input_fn** 参数定义了一个不断输入训练数据的函数，**trian** 方法的调用暗示着 **train_input_fn**输入数据，以下是这个函数的定义：
```python
def train_input_fn(features, labels, batch_size):
```
我们传递以下几个参数给 **train_input_fn**：
* train_feature 是一个 Python 数据字典：
    * 键(key)为属性的名称
    * 值(value)为一个例如训练数据集中数据格式的数组
* train_label 为训练数据集中包含每个样例标签值的数组
* args.batch_size 是一个定义了数据维度的整型数
**train_input_fn**函数依赖于**Dataset API**(上面提到的api之一),它是一个能够读取数据并将数据转化为训练所需的格式的高级 TensorFlow API ，下面这个函数的调用把属性和标签转化为 **tf.data.Dataset** 对象，Dataset API 的一个基类 。
```python
 dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
```
tf.dataset 类提供了许多有用的函数来处理训练所需的样例，下面这一行调用了其中三个函数：
```python
  dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
```
如果训练的样例排序越随机，那么训练的结果会更好。为了给我们的样例随机排序，需要调用 **tf.data.Dataset.shuffle** 这个方法。将缓存大小设置为比样例数据总和（120）更大的数字，确保样例数据能够打乱顺序。

在训练的过程中，训练方法会多次处理样例。在不设置任何参数的情况下调用 **tf.data.Dataset.repeat** 方法来确保 **train** 方法能够无尽的（随机排序后的）训练样例。

训练方法一次能够处理[一批](https://developers.google.com/machine-learning/glossary/#batch)样例数据, **tf.data.Dataset.batch** 方法能够创造一个连接多批次样例的bath（批处理）流，这个程序将默认的[**bath size**](https://developers.google.com/machine-learning/glossary/#batch_size)设置为100，意味着 **batch** 方法会将100个样例串联起来。最理想的批处理大小往往和问题本身有关。就以往的经验来说，越小的批处理维度会使训练方法更快的训练模型，但是有时会使结果的精确度降低。

Evaluate the model(评估我们的模型)
---------------------------------
**Evaluating**(评估)意味着我们需要确定我们的模型是否能够高效准确的做出预测。为了确定鸢尾花分类模型的效率，我们传递鸢尾花花萼和花瓣的数据给模型，并让其做出鸢尾花种类的预测，然后将模型给出的预测值和真实值做比。例如，模型在一半的样例输入的情况下能够产生正确预测概率为0.5的预测结果。下图的模型更为高效

|花萼长度(测试数据)|花萼宽度(测试数据)|花朵长度(测试数据)|花朵宽度(测试数据)|花种代号(测试数据)|预测值|
|------------------|------------------|------------------|------------------|------------------|------|
|5.9               |3.0                |4.3              |1.5               |1                 |1
|6.9               |3.1                |5.4              |2.1               |2                 |2
|5.1               |3.3                |1.7              |0.5               |0                 |0
|6.4               |2.8                |1.7              |2.2               |1                 |2(错误预测)
|6.4               |2.8                |1.7              |2.2               |2                 |1


**产生了一个精准度为80%的模型**

为了评估模型的效率，每个评估器都提供了一个 **evaluate**的方法，在 **premade_estimator.py** 程序中像下面这样调用了 **evaluate** 方法
```python
# Evaluate the model.
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_x, test_y, args.batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
```
**classifier.evaluate** 的调用和我们使用 **classifier.train** 的时候非常类似，最大的区别在于 **classifier.evaluate** 必须从册数数据源而不是训练数据源获取数据。换句话说，为了能够正确的得到模型的效率，评估模型所使用的数据必须与训练所使用的数据不同。**eval_input_fn** 函数能够从测试数据源中获取数据：
```python
def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
```
简短而言， **eval_input_fn** 在被 **classifier.evaluate** 调用的时候做了下面的几件事：
1. 从训练数据集中提取属性和标签数据并转化为 **tf.dataset**对象。
2. 创建测试数据集的一个批处理(对于测试数据集无需随机排序或重复代入)
3. 给 **classifier.evaluate** 返回测试数据的批处理。
运行代码会不断的生成类似下面这样的输出:
````python
Test set accuracy: 0.967
````

Predicting(预测)
----------------
我们训练出了一个模型，并且“认为”它能够很好的处理鸢尾花分类问题，那么接下来让我们使用训练好的模型做几个未标记的数据样例([unlabeled examples](https://developers.google.com/machine-learning/glossary/#unlabeled_example)),仅有属性，但是却没有类别标签。

在现实生活中，我们可能会从不同的 CSV 文件，数据收集应用中获取未标记的样例，然后提供给模型进行预测。现在我们仅仅手动将提供需要预测的数据样例提供给我们的模型：
```python
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
```
每个评估器(Estimator)都提供了一个 **predict**(预测)方法，在 **premade_estimator.py** 程序中如下方式调用:
```python
predictions = classifier.predict(
    input_fn=lambda:eval_input_fn(predict_x, batch_size=args.batch_size))
```
像 **evaluate** 那样，**predict** 方法也能够从 **eval_input_fn** 方法中获取样例数据。
当执行预测过程的时候，我们不把标签(label)传递给 **eval_input_fn**，因此  **eval_input_fn** 做了如下的事情：

1. 将原有的3元属性手动设置为我们创建的数据
2. 从手动提供的样例中创建一个批处理
3. 返回一个可供 **classifier.predict** 调用的批处理(batch)

**predict** 预测方法能够产生一个 python 迭代器 对象，不断生成结果样例的字典，字典中具有几个键，**probabilities**(概率)键的值为三个浮点数，每一个值代表了输入数据是特定鸢尾花花种的概率。例如，考虑如下的 **probabilities**列表：
```python
'probabilities': array([  1.19127117e-08,   3.97069454e-02,   9.60292995e-01])
```
上面的列表意味着：
* 输入的样例属于 Setosa(0号花种) 的可能性忽略不计
* 3.97%的概率属于 Versicolor(1号花种)
* 96.0%的概率属于 Virginica(2号花种)

**class_ids**键是一个仅有一个元素的数组，其值代表着最有可能的种类。例如：
```python
'class_ids': array([2])
```
数字2代表着 Virginica(鸢尾花花种) ，下面的代码根据预测值进行预测结果的汇报：
```python
for pred_dict, expec in zip(predictions, expected):
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]
    print(template.format(SPECIES[class_id], 100 * probability, expec))
```
运行代码会生成下面的输出：
```python
...
Prediction is "Setosa" (99.6%), expected "Setosa"

Prediction is "Versicolor" (99.8%), expected "Versicolor"

Prediction is "Virginica" (97.9%), expected "Virginica"
```

Summary(总结)
-------------
这篇文章提供了一个机器学习简短的介绍

因为 **premade_estimators.py** 中的程序依赖于 TensorFlow 高级 API,许多机器学习中的复杂数学问题都被隐藏，如果你想成为一个机器学习专家，我们最终建议学习有关 [**gradient descent**](https://developers.google.com/machine-learning/glossary/#gradient_descent)(梯度下降)，batching(批处理)，neural networks(神经网络) 的知识。

接下来我们建议阅读 [Feature Documents](https://www.tensorflow.org/get_started/feature_columns)(讲解了上面算法的具体流程)文档.