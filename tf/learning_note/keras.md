# keras

Keras是一个用于构建和培训深度学习模型的高级API。 它用于快速原型设计，高级研究和生产，具有三个主要优势：
- 用户友好
    - Keras具有针对常见用例优化的简单，一致的界面。 它为用户错误提供清晰且可操作的反馈。
- 模块化和可组合
    - Keras模型是通过将可配置的构建块连接在一起而制定的，几乎没有限制。
- 易于扩展
    - 编写自定义构建块以表达研究的新想法。 创建新图层，损失函数并开发最先进的模型。

`tf.keras` 是TensorFlow实现的Keras API规范。

`tf.keras` 可以运行一个兼容Keras的代码，但是：
1. 在最新的TensorFlow版本中的`tf.keras`版本可能和最新的PyPI的Keras版本不一致。

2. 当保存模型的权重时，`tf.keras`默认使用`checkpoint format` 格式。通过传`save_format='h5'` 来使用HDF5

## 构建一个简单模型

### Sequential 模型

在Keras，通过装配layers来构建模型。一个模型通常是一个layers图。最常见类型是一堆的layers：`tf.keras.Sequential`模型。

为了构建一个简单的，全连接的网络(比如多层感知机)：

```python
model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(layers.Dense(64, activation='relu'))
# Add another:
model.add(layers.Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(layers.Dense(10, activation='softmax'))
```

### 配置layers

有许多tf.keras.layers可用于一些常见的构造函数参数：

- activation: 为layer设置激活函数。此参数由内置函数的名称或可调用对象指定。 默认情况下，不应用任何激活。

- kernel_initializer 和 bias_initializer: 创建图层权重（内核和偏差）的初始化方案。 此参数是名称或可调用对象。 这默认为“Glorot uniform”初始化程序。

- kernel_regularizer and bias_regularizer: 应用层权重（内核和偏差）的正则化方案，例如L1或L2正则化。 默认情况下，不应用正则化。

下面使用构造函数参数实例化tf.keras.layers.Dense图层：

```python
# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
```

## 训练和评估

### 设置训练

构建模型后，通过调用compile方法配置其学习过程：

```python
model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(64, activation='relu'),
# Add another:
layers.Dense(64, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

tf.keras.Model.compile 传入3个重要的参数：

- optimizer: 此对象指定训练过程。 从tf.train模块传递优化器实例，例如tf.train.AdamOptimizer，tf.train.RMSPropOptimizer或tf.train.GradientDescentOptimizer。

- loss: 在优化期间最小化的功能。 常见的选择包括均方误差（mse），categorical_crossentropy和binary_crossentropy。 损失函数由名称或通过从tf.keras.losses模块传递可调用对象来指定。

- metrics: 用于监控训练。 这些是来自tf.keras.metrics模块的字符串名称或callables。

以下显示了配置训练模型的几个示例：

```python
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])
```

### 输入Numpy数据

对于小型数据集，请使用内存中的NumPy阵列来训练和评估模型。 使用拟合方法将模型“拟合”到训练数据：

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)
```

tf.keras.Model.fit有三个重要参数：

- epochs: 训练以纪元为结构。 一个纪元是对整个输入数据的一次迭代（这是以较小的批次完成的）。

- batch_size：当传递NumPy数据时，模型将数据分成较小的批次，并在训练期间迭代这些批次。 此整数指定每个批次的大小。 请注意，如果样本总数不能被批量大小整除，则最后一批可能会更小。

- validation_data：在对模型进行原型设计时，您希望轻松监控其在某些验证数据上的性能。 传递这个参数 - 输入和标签的元组 - 允许模型在每个时期结束时以推断数据的推理模式显示损失和度量。

```python
import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))
```

### Input tf.data datasets

使用数据集API可扩展到大型数据集或多设备训练。 将tf.data.Dataset实例传递给fit方法：

```python

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
```

这里，fit方法使用steps_per_epoch参数 - 这是模型在移动到下一个纪元之前运行的训练步数。 由于数据集生成批量数据，因此此代码段不需要batch_size。

数据集也可用于验证：

```python
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
```

### 评估和预测

tf.keras.Model.evaluate和tf.keras.Model.predict方法可以使用NumPy数据和tf.data.Dataset。
要评估所提供数据的推理模式损失和指标：

```python
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)
```

并且作为NumPy数组，预测所提供数据的推断中最后一层的输出：

```python

result = model.predict(data, batch_size=32)
print(result.shape)
```

## 构建高级模型

### API

tf.keras.Sequential模型是一个简单的图层堆栈，不能代表任意模型。 使用Keras功能API构建复杂的模型拓扑，例如：

- Multi-input models,
- Multi-output models,
- Models with shared layers (the same layer called several times),
- Models with non-sequential data flows (e.g. residual connections).

使用功能API构建模型的工作方式如下：

1. A layer instance is callable and returns a tensor.
2. Input tensors and output tensors are used to define a tf.keras.Model instance.
3. This model is trained just like the Sequential model.

以下示例使用功能API构建一个简单，完全连接的网络：

```python
inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)
```

实例化给定输入和输出的模型。

```python
model = tf.keras.Model(inputs=inputs, outputs=predictions)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)
```

### 模型子类化

通过继承tf.keras.Model并定义自己的前向传递来构建完全可自定义的模型。 在__init__方法中创建图层并将它们设置为类实例的属性。 在call方法中定义正向传递。

当启用eager执行时，模型子类化特别有用，因为可以强制写入正向传递。

以下示例显示了使用自定义正向传递的子类tf.keras.Model：

```python
class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes
    return tf.TensorShape(shape)
```

实例化新的模型类：

```python
model = MyModel(num_classes=10)

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

### 自定义layers

通过继承tf.keras.layers.Layer并实现以下方法来创建自定义层：

- build: Create the weights of the layer. Add weights with the add_weight method.
- call: Define the forward pass.
- compute_output_shape: Specify how to compute the output shape of the layer given the input shape.
- Optionally, a layer can be serialized by implementing the get_config method and the from_config class method.

这是一个自定义层的示例，它使用内核矩阵实现输入的matmul：

```python
class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    # Be sure to call this at the end
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)
```

Create a model using your custom layer:

```python
model = tf.keras.Sequential([
    MyLayer(10),
    layers.Activation('softmax')])

# The compile step specifies the training configuration
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)
```

### Callbacks

回调是传递给模型的对象，用于在训练期间自定义和扩展其行为。 您可以编写自己的自定义回调，或使用包含以下内置的tf.keras.callbacks：

- tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
- tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.
- tf.keras.callbacks.EarlyStopping: Interrupt training when validation performance has stopped improving.
- tf.keras.callbacks.TensorBoard: Monitor the model's behavior using TensorBoard.

To use a tf.keras.callbacks.Callback, pass it to the model's fit method:

```python
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))
```

### Save and restore

#### Weights only

Save and load the weights of a model using tf.keras.Model.save_weights:

```python
model = tf.keras.Sequential([
layers.Dense(64, activation='relu'),
layers.Dense(10, activation='softmax')])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')
```

默认情况下，这会以TensorFlow检查点文件格式保存模型的权重。 权重也可以保存为Keras HDF5格式（Keras的多后端实现的默认值）：

```python
# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')
```

#### Configuration only

可以保存模型的配置 - 这可以在没有任何权重的情况下序列化模型体系结构。 即使没有定义原始模型的代码，保存的配置也可以重新创建和初始化相同的模型。 Keras支持JSON和YAML序列化格式：

```python
# Serialize a model to JSON format
json_string = model.to_json()
json_string


import json
import pprint
pprint.pprint(json.loads(json_string))
```

Recreate the model (freshly initialized), from the json.
`fresh_model = tf.keras.models.model_from_json(json_string)`

Serialize a model to YAML format

`yaml_string = model.to_yaml()
print(yaml_string)`

Recreate the model from the yaml

`
fresh_model = tf.keras.models.model_from_yaml(yaml_string)`

#### Entire model

整个模型可以保存到包含权重值，模型配置甚至优化器配置的文件中。 这允许您检查模型并稍后从完全相同的状态恢复训练 - 无需访问原始代码。

```python
# Create a trivial model
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')
```

### Eager execution

急切执行是一个必要的编程环境，可以立即评估操作。 这对于Keras不是必需的，但是由tf.keras支持，对于检查程序和调试很有用。

所有tf.keras模型构建API都与急切执行兼容。 虽然可以使用顺序和功能API，但是热切执行尤其有利于模型子类化和构建自定义层 - 需要您将正向传递作为代码编写的API（而不是通过组合现有层来创建模型的API）。

有关使用具有自定义训练循环和tf.GradientTape的Keras模型的示例，请参阅急切的执行指南。

### Distribution

#### Estimators

The Estimators API is used for training models for distributed environments. This targets industry use cases such as distributed training on large datasets that can export a model for production.

A tf.keras.Model can be trained with the tf.estimator API by converting the model to an tf.estimator.Estimator object with tf.keras.estimator.model_to_estimator. See Creating Estimators from Keras models.

```python
model = tf.keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model)
```

### Multiple GPUs

tf.keras models can run on multiple GPUs using tf.contrib.distribute.DistributionStrategy. This API provides distributed training on multiple GPUs with almost no changes to existing code.

Currently, tf.contrib.distribute.MirroredStrategy is the only supported distribution strategy. MirroredStrategy does in-graph replication with synchronous training using all-reduce on a single machine. To use DistributionStrategy with Keras, convert the tf.keras.Model to a tf.estimator.Estimator with tf.keras.estimator.model_to_estimator, then train the estimator

The following example distributes a tf.keras.Model across multiple GPUs on a single machine.

First, define a simple model:

```python
model = tf.keras.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10,)))
model.add(layers.Dense(1, activation='sigmoid'))

optimizer = tf.train.GradientDescentOptimizer(0.2)

model.compile(loss='binary_crossentropy', optimizer=optimizer)
model.summary()
```

Define an input pipeline. The input_fn returns a tf.data.Dataset object used to distribute the data across multiple devices—with each device processing a slice of the input batch.

```python
def input_fn():
  x = np.random.random((1024, 10))
  y = np.random.randint(2, size=(1024, 1))
  x = tf.cast(x, tf.float32)
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset = dataset.repeat(10)
  dataset = dataset.batch(32)
  return dataset
```

Next, create a tf.estimator.RunConfig and set the train_distribute argument to the tf.contrib.distribute.MirroredStrategy instance. When creating MirroredStrategy, you can specify a list of devices or set the num_gpus argument. The default uses all available GPUs, like the following:

```python
strategy = tf.contrib.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(train_distribute=strategy)
```

Convert the Keras model to a tf.estimator.Estimator instance:

```python

keras_estimator = tf.keras.estimator.model_to_estimator(
  keras_model=model,
  config=config,
  model_dir='/tmp/model_dir')
```

Finally, train the Estimator instance by providing the input_fn and steps arguments:

`keras_estimator.train(input_fn=input_fn, steps=10)`