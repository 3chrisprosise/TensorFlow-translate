Feature Columns
---------------
这篇文档描述了feature columns(特性列)的细节,将特性列看作是原始数据和评估器间的中介。特性列有很多方法，使你能够将各种各样的原始数据转化成为评估其可以使用的数据，并且转化的过程非常简便。

在 [**Premade Estimators**](https://www.tensorflow.org/get_started/premade_estimators)(预先估计模型) 中，我们通过对 **premade Estimator**(预先估计量)，[DNNClassifier](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier)(基于深度神经网络的分类器)的使用来训练模型、通过输入的属性列来预测不同鸢尾花种类。这个样例仅仅创建了数值化的属性列([tf.feature_column.numeric_columnv](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column)),尽管数值列能够高效的表示出花萼和花瓣的长度数据，现实世界中的数据仍包含多种属性，而且许多是非数值的。
![自然界非数值化数据](https://www.tensorflow.org/images/feature_columns/feature_cloud.jpg)
自然界的一些属性(例如长度)是数值化的，然而更多是非数值的。

Input to a Deep Neural Network(输入深度神经网络)
------------------------------------------------
深度神经网络可以处理的数据格式是什么样子的？答案当然是数值型(例如：tf.float32)，毕竟每个网络中的神经元利用乘法和加法对权重和输入数据进行操作。然而，现实生活中的数据往往是非数值型(分类)的数据,例如，如果一个 **product_class**属性列包含下面几个非数值型的数据：
* kitchenware(厨房用具)
* electronics(电子产品)
* sports(体育运动)
机器学习模型将分类问题看作是一个简单的向量，0代表不属于，1代表属于，例如，当我们将 **product_class** 看作是体育运动的时候，一个机器学习模型通常会生成一个 **product_class** 向量，像这样 **[0, 0, 1]**，意味着：
* 0：不属于厨房用品
* 0：不属于电子产品
* 1：属于体育运动
所以无论原始数据是数值型还是分类型，一个机器学习算法都会将其属性转化为数值型处理。

Feature Columns(属性列)
-----------------------
如下图所示的那样，你通过评估器(鸢尾花问题中我们使用的是 DNNClassifier)中的 **feature_columns** 参数将数据输入到模型。特征列(Feature Columns)为模型和输入数据间创造了桥梁。
![数据的转化过程](https://www.tensorflow.org/images/feature_columns/inputs_to_model_bridge.jpg)

可以通过调用 tf.feature_column 模快创造一个特征列(Feature Columns)，这篇文章将会讲述其中9个函数。如下图所示，这九个函数除了 **bucketized_column** 都返回一个 **Categorical-Column**(分类列) 或 **Dense-Column**(密集数据列)对象, 同时继承了两个类。 
![tf.feature_column](https://www.tensorflow.org/images/feature_columns/some_constructors.jpg)

让我们更加细致的研究一下这些函数。

####Numeric column (数值型数据列)
鸢尾花问题的分类器调用了 **tf.feature_column.numeric_column** 函数处理所有的输入属性：
* SepalLength(花萼长度)
* SepalWidth
* PetalLength(花瓣长度)
* PetalWidth
尽管 **tf.numeric_column** 提供了可选的参数，不改变任何参数的情况仍旧是一个针对数值型输入模型的较好的处理方法。
```python
# Defaults to a tf.float32 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength")
```
不适用默认数据类型，可以为 **dtype** 参数赋值：
```python
# Represent a tf.float64 scalar.
numeric_feature_column = tf.feature_column.numeric_column(key="SepalLength",
                                                          dtype=tf.float64)
```
默认情况下,数值型列会创建一个单一值(标量)，使用 **shape** 参数指定另一个维度，例如：
```python
# Represent a 10-element vector in which each cell contains a tf.float32.
vector_feature_column = tf.feature_column.numeric_column(key="Bowling",
                                                         shape=10)

# Represent a 10x5 matrix in which each cell contains a tf.float32.
matrix_feature_column = tf.feature_column.numeric_column(key="MyMatrix",
                                                         shape=[10,5])
```

####Bucketized column
通常情况下，我们并不想直接将数据输入到模型中，而是根据数据的范围将数据分成几类，那么可以通过创建一个 **[bucketized column](https://www.tensorflow.org/api_docs/python/tf/feature_column/bucketized_column)** 解决这个问题。例如，考虑房屋建造时间的原始数据，我们将年份划分成几个范围而不是直接罗列出数据的具体值：
![房屋建造年份](https://www.tensorflow.org/images/feature_columns/bucketized_column.jpg)
模型会将数据表示成下面这样

|   Date Range        |     Represented as...  |
|---------------------|------------------------|
|  < 1960             | [1, 0, 0, 0]          |
|  >= 1960 but < 1980 | [0, 1, 0, 0]          |
|  >= 1980 but < 2000 | [0, 0, 1, 0]          |
|  > 2000             | [0, 0, 0, 1]          |

为什么我们会把完美的数值型数据分割为分类数据带入到模型中？分类方法将单独的输入数据分成了四维向量，这样我们的模型就可以学习生成四个独立的权重，而不是生成单一权重；这样的结果是产生更好的模型。更重要的是，bucketizing(这里我译作划分)，划分使得模型能够更加清晰的区分出不同年份，因为其中一个元素的值为(1)而其他三个为(0),当我们使用单一年份作为输入的时候，模型仅能学习线性的关系，所以 bucketing(划分)对于模型的灵活适应性而言更有利。

下面的代码示例展示了如何创建一个 bucketized feature：
```python
# First, convert the raw input to a numeric column.
numeric_feature_column = tf.feature_column.numeric_column("Year")

# Then, bucketize the numeric column on the years 1960, 1980, and 2000.
bucketized_feature_column = tf.feature_column.bucketized_column(
    source_column = numeric_feature_column,
    boundaries = [1960, 1980, 2000])
```
注意！指定的三维的向量最终创建出四维 bucketized(划分后的) 向量

Categorical identity column (分类识别列)
----------------------------------------
**Categorical identity column**可以看作是 **bucketized columns** 的一种特殊情况，在传统的 **bucketized columns** 中，每个 bucket 代表着数据的范围，然而在 **Categorical identity column** (分类识别列)中, 每个 bucket 代表一个特定的数值 0，1，2或3，这种情况下，一个分类识别关系映射图看起来像这样：
![分类识别关系映射](https://www.tensorflow.org/images/feature_columns/categorical_column_with_identity.jpg)
一个 categorical identity column 的映射关系。注意这里编码的方式是[一位有效编码](http://blog.csdn.net/google19890102/article/details/44039761),而不是二进制数值码。

与 **bucketized columns** 类似，一个模型可以从 **categorical identity column** 中得到几个独立的权重。例如，使用不同的整形数来代替字符描述产品种类的 **product_class**:
* 0="kitchenware"
* 1="electronics"
* 2="sport"
通过对 **[tf.feature_column.categorical_column_with_identity](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_identity)** 的调用可以实现一个 **categorical identity column** ，例如：
```python
# Create categorical output for an integer feature named "my_feature_b",
# The values of my_feature_b must be >= 0 and < num_buckets
identity_feature_column = tf.feature_column.categorical_column_with_identity(
    key='my_feature_b',
    num_buckets=4) # Values [0, 4)

# In order for the preceding call to work, the input_fn() must return
# a dictionary containing 'my_feature_b' as a key. Furthermore, the values
# assigned to 'my_feature_b' must belong to the set [0, 4).
def input_fn():
    ...
    return ({ 'my_feature_a':[7, 9, 5, 2], 'my_feature_b':[3, 1, 2, 2] },
            [Label_values])
```

###Categorical vocabulary column
我们不能够将字符串直接输入到模型当中，我们必须先将字符映射到数值分类的值。**Categorical vocabulary columns** 提供了一个非常好的将字符映射为 **一位有效编码**(前文提到) 向量，例如：
![映射关系图](https://www.tensorflow.org/images/feature_columns/categorical_column_with_vocabulary.jpg)

如你所见，可以把 **Categorical vocabulary column** 看作是一种枚举型的 **categorical identity columns**，TensorFlow 提供了两种不同的函数来创造 **create categorical vocabulary columns**：
* [tf.feature_column.categorical_column_with_vocabulary_list](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file)
* [tf.feature_column.categorical_column_with_vocabulary_file](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_file)

**categorical_column_with_vocabulary_list** 能够讲字符串根据词汇表映射到数字。例如：
```python
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature by mapping the input to one of
# the elements in the vocabulary list.
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_list(
        key="a feature returned by input_fn()",
        vocabulary_list=["kitchenware", "electronics", "sports"])
```
上面这个函数直接了当，但是仍有缺点。当词汇过长的时候，就需要很大规模的输入。这种情况下可以通过对 **tf.feature_column.categorical_column_with_vocabulary_file** 的调用能够从文件中读取词汇，例如：
```python
# Given input "feature_name_from_input_fn" which is a string,
# create a categorical feature to our model by mapping the input to one of
# the elements in the vocabulary file
vocabulary_feature_column =
    tf.feature_column.categorical_column_with_vocabulary_file(
        key="a feature returned by input_fn()",
        vocabulary_file="product_class.txt",
        vocabulary_size=3)
```

**product_class.txt** 文件中的每一行是一个词汇，例如：
```python
kitchenware
electronics
sports
```

###Hashed Column(哈希列)
到目前为止，我们仅仅处理过少量类别的分类问题。例如，我们的输出样例只含3类。大多数情况下，输入数据的类别过大导致不能将每个词汇都对应到一个数值，因为太耗费内存了(网络展开后带入数据进行运算的时候内存需求往往和输入呈几何型增长)。这种情况下，我们通常会将问题转化为“我们究竟需要从输入的数据中归纳出多少种类别？”，事实上 [**tf.feature_column.categorical_column_with_hash_bucket**](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket) 函数可以手动指定输入数据类别，当模型使用这种输入列的时候，会自动计算输入数据的哈希值，然后使用模块操作，带入到 **hash_bucket_size** 当中。如下：
```python
# pseudocode
feature_id = hash(raw_feature) % hash_buckets_size
```
创建**feature_column** (特性列)的代码很像下面的代码：
```python
hashed_feature_column =
    tf.feature_column.categorical_column_with_hash_bucket(
        key = "some_feature",
        hash_buckets_size = 100) # The number of categories
```
这样开来，你会觉得一切很诡异，毕竟我们将不同的特征划分到同一个值上去，这就意味着两个肯能毫无关系的特征会被映射到同一个类别上去，对于神经网络而言可能这两个特征会变成同一个。下图说明了这一问题，两个不同的类别经过处理后变为了同一个类别。
![哈希列](https://www.tensorflow.org/images/feature_columns/hashed_column.jpg)

类似于机器学习中的许多反直觉现象，在实践过程中，哈希的效果反而不错。这是因为哈希分类给予模型适当的数据划分(这里不是很懂)，使得模型可以利用额外的属性将体育用品(sports)中的kickenware(厨具)划分出来。

###Crossed column
将多个特性组合成为一个特性[feature crosses](https://developers.google.com/machine-learning/glossary/#feature_cross)(特性交叉),使得模型能够得到各个属性的独立权重。

更加具体而言，假设我们需要设计模型来计算 **Atlanta, GA**(一个地点) 的房地产价格。然而这个区域的房地产价格根据地理位置的不同差异巨大。根据经纬度划分房地产地理位置在这个问题上并不是很适用（特征列太多，且难以总结规律）。将精度和维度进行属性交叉(feature crosses),合并为一个属性，反而可以指明地点。假设我们将当前区域分为100 x 100 的区域,将10000个区域中的每个看作是一个经纬度交叉的结果。特性交叉使得模型能够根据相关的独立区域计算价格情况，并且能够更加强有力的反映出价格和区域的关系。

下图展示了经纬度关系：
![经纬度划分图](https://www.tensorflow.org/images/feature_columns/Atlanta.jpg)

我们按照上面的解决方案，我们通过调用[tf.feature_column.crossed_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/crossed_column)函数并组合之前提到的 **bucketized_column** 列来实现
```python
def make_dataset(latitude, longitude, labels):
    assert latitude.shape == longitude.shape == labels.shape

    features = {'latitude': latitude.flatten(),
                'longitude': longitude.flatten()}
    labels=labels.flatten()

    return tf.data.Dataset.from_tensor_slices((features, labels))

# Bucketize the latitude and longitude usig the `edges`
latitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('latitude'),
    list(atlanta.latitude.edges))

longitude_bucket_fc = tf.feature_column.bucketized_column(
    tf.feature_column.numeric_column('longitude'),
    list(atlanta.longitude.edges))

# Cross the bucketized columns, using 5000 hash bins.
crossed_lat_lon_fc = tf.feature_column.crossed_column(
    [latitude_bucket_fc, longitude_bucket_fc], 5000)

fc = [
    latitude_bucket_fc,
    longitude_bucket_fc,
    crossed_lat_lon_fc]

# Build and train the Estimator.
est = tf.estimator.LinearRegressor(fc, ...)
```

你可以通过下面的方法来创建交叉属性(**feature cross**)
* 属性名称(Feature names), 通过从 **input_fn** 返回的字典中得到。
* 任何分类过的列，前提是根据 **categorical_column_with_hash_bucket** （经过哈希处理的交叉属性列（**crossed_column**））
当特性列 **latitude_bucket_fc** 和 **longitude_bucket_fc** 交叉过后，TensorFlow 会根据样例的数据创建成对的数据列, 会创建出类似下列格式的网格：
```python
 (0,0),  (0,1)...  (0,99)
 (1,0),  (1,1)...  (1,99)
   ...     ...       ...
(99,0), (99,1)...(99, 99)
```

网格只有和对应的词汇表同时输入才具有意义。除了手动创建这个数据量巨大的输入数据表格 **crossed_column**(交叉列)会根据 **hash_bucket_size** 参数来创建数值。 特性列(feature column)会根据哈希函数和输入数据创建样例及其相应的索引表。接下来根据 **hash_bucket_size** 参数进行相应的模块操作。

正如上面讨论的那样，使用哈希及相应的模块方法确实可以使得分类的数量减少，但是会导致类别冲突；意味着多个属性交叉会产生相同的哈希桶(hash bucket)，但是实践证明，属性交叉(**feature crosses**)会提高模型的适应性(具体怎么提高没有说。。)

与直觉相反，创建属性交叉(feature crosses)的时候,仍然需要将原始数据输入到模型中(在前面简化示例代码中有提到)。独立的经纬度数据可以在交叉属性的哈希特性冲突的时候帮助模型辅助判断。

###Indicator and embedding columns(指示器和嵌入列)
指示器列和嵌入列并不直接作用于属性，而是将分类列(categorical columns)作为输入值。

当使用指示器列(indicator column),我们使 TensorFlow 直接处理我们分类后的样例(前面提到的 **product_class**)。指示器列(indicator column)将每个分类当作一个一位有效编码对待，当匹配的类别具有值 1 而其他分类为 0:
![指示器列](https://www.tensorflow.org/images/feature_columns/categorical_column_with_identity.jpg)

你可以通过调用 [**tf.feature_column.indicator_column**](https://www.tensorflow.org/api_docs/python/tf/feature_column/indicator_column)来创建一个指示器列
```python
categorical_column = ... # Create any type of categorical column.

# Represent the categorical column as an indicator column.
indicator_column = tf.feature_column.indicator_column(categorical_column)
```

现在假设我们有一百万或者一亿个分类。使用带有指示器列的神经网络进行训练的可能性几乎为0。

我们可以使用嵌入列(embedding column)来突破这一限制。嵌入列使用低维度的常量向量(每个单元可以是任意值，而不再是0或者1),来代替一位有效编码来表示数据。通过允许每个单元表示任意值，嵌入列能够比指示器列含有更少的单元数量。

让我们来看一个指示器列和嵌入列比较的例子。假设我们的输入样例包含从81个词语中选出的任意词。再假设我们的数据集中有四个不同的示例：
* "dog"
* "spoon"
* "scissors"
* "guitar"

这种情况下，下图展示出了指示器列和嵌入列相应的数据表示情况：
![指示器列和嵌入列](https://www.tensorflow.org/images/feature_columns/embedding_vs_indicator.jpg)
嵌入列相比于指示器列使用更低维度的空间存储了分类的数值

当样例通过分类列函数(**categorical_column_with...**)进行处理后，得到了一个字符和数值的映射关系。例如，某个方法将 "spoon" 映射到了[32] 这里的32只是一个示例, 接着你可以通过下面两种方式来表示数值分类：
* 指示器列。有一个函数可以将数值化的分类值转化为81个维度的向量(我们之前说的类别共有81种),其中有一个向量中的某个值为1其余值为0的时候表示一个具体的分类代号(0, 32, 79, 80)

* 嵌入列。有一个函数能够将数值分类代号(0,32,79,80)转化为查询表中的编号。每个查询表中的值包含一个3维向量。（这边还是有点不懂）