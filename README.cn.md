
# 图像分类

## 背景介绍

图像相比文字能够提供更加生动、容易理解及更具艺术感的信息，是人们转递与交换信息的重要来源。我们的本次研究对象为图像识别的一个重要哦分类图像分类。

图像分类是根据图像的语义信息将不同类别图像区分开来，是计算机视觉中重要的基本问题，也是图像检测、图像分割、物体跟踪、行为分析等其他高层视觉任务的基础。图像分类在很多领域有广泛应用，包括安防领域的人脸识别和智能视频分析等，交通领域的交通场景识别，互联网领域基于内容的图像检索和相册自动归类，医学领域的图像识别等。


一般来说，图像分类通过手工提取特征或特征学习方法对整个图像进行全部描述，然后使用分类器判别物体类别，因此如何提取图像的特征至关重要。在深度学习算法之前使用较多的是基于词袋(Bag of Words)模型的物体分类方法。词袋方法从自然语言处理中引入，即一句话可以用一个装了词的袋子表示其特征，袋子中的词为句子中的单词、短语或字。对于图像而言，词袋方法需要构建字典。最简单的词袋模型框架可以设计为**底层特征抽取**、**特征编码**、**分类器设计**三个过程。

而基于深度学习的图像分类方法，可以通过有监督或无监督的方式**学习**层次化的特征描述，从而取代了手工设计或选择图像特征的工作。深度学习模型中的卷积神经网络(Convolution Neural Network, CNN)近年来在图像领域取得了惊人的成绩，CNN直接利用图像像素信息作为输入，最大程度上保留了输入图像的所有信息，通过卷积操作进行特征的提取和高层抽象，模型输出直接是图像识别的结果。这种基于"输入-输出"直接端到端的学习方法取得了非常好的效果，得到了广泛的应用。

## 效果展示

图像分类包括通用图像分类、细粒度图像分类等。图1展示了通用图像分类效果，即模型可以正确识别图像上的主要物体。

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/dog_cat.png?raw=true"  width="350" ><br/>
图1. 通用图像分类展示
</p>


图2展示了细粒度图像分类-花卉识别的效果，要求模型可以正确识别花的类别。


<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/flowers.png?raw=true" width="400" ><br/>
图2. 细粒度图像分类展示
</p>


一个好的模型既要对不同类别识别正确，同时也应该能够对不同视角、光照、背景、变形或部分遮挡的图像正确识别(这里我们统一称作图像扰动)。图3展示了一些图像的扰动，较好的模型会像聪明的人类一样能够正确识别。

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/variations.png?raw=true" width="550" ><br/>
图3. 扰动图片展示[22]
</p>

## 数据准备
我们使用[CIFAR10](<https://www.cs.toronto.edu/~kriz/cifar.html>)数据集。CIFAR10数据集包含60,000张32x32的彩色图片，10个类别，每个类包含6,000张。其中50,000张图片作为训练集，10000张作为测试集。图11从每个类别中随机抽取了10张图片，展示了所有的类别。

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/cifar.png?raw=true" width="350"><br/>
图11. CIFAR10数据集[21]
</p>

Paddle API提供了自动加载cifar数据集模块 `paddle.dataset.cifar`。

### 模型结构

#### Paddle 初始化

导入 Paddle Fluid API 和辅助模块。

```python
import paddle
import paddle.fluid as fluid
import numpy
import sys
from __future__ import print_function

```


### VGG

首先介绍VGG模型结构，由于CIFAR10图片大小和数量相比ImageNet数据小很多，因此这里的模型针对CIFAR10数据做了一定的适配。卷积部分引入了BN和Dropout操作。
VGG核心模块的输入是数据层，`vgg_bn_drop` 定义了16层VGG结构，每层卷积后面引入BN层和Dropout层，详细的定义如下：

```python
def vgg_bn_drop(input):
    def conv_block(ipt, num_filter, groups, dropouts):
        return fluid.nets.img_conv_group(
            input=ipt,
            pool_size=2,
            pool_stride=2,
            conv_num_filter=[num_filter] * groups,
            conv_filter_size=3,
            conv_act='relu',
            conv_with_batchnorm=True,
            conv_batchnorm_drop_rate=dropouts,
            pool_type='max')

    conv1 = conv_block(input, 64, 2, [0.3, 0])
    conv2 = conv_block(conv1, 128, 2, [0.4, 0])
    conv3 = conv_block(conv2, 256, 3, [0.4, 0.4, 0])
    conv4 = conv_block(conv3, 512, 3, [0.4, 0.4, 0])
    conv5 = conv_block(conv4, 512, 3, [0.4, 0.4, 0])

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(input=drop, size=512, act=None)
    bn = fluid.layers.batch_norm(input=fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.5)
    fc2 = fluid.layers.fc(input=drop2, size=512, act=None)
    predict = fluid.layers.fc(input=fc2, size=10, act='softmax')
    return predict
```


1. 首先定义了一组卷积网络，即conv_block。卷积核大小为3x3，池化窗口大小为2x2，窗口滑动大小为2，groups决定每组VGG模块是几次连续的卷积操作，dropouts指定Dropout操作的概率。所使用的`img_conv_group`是在`paddle.networks`中预定义的模块，由若干组 Conv->BN->ReLu->Dropout 和 一组 Pooling 组成。

2. 五组卷积操作，即 5个conv_block。 第一、二组采用两次连续的卷积操作。第三、四、五组采用三次连续的卷积操作。每组最后一个卷积后面Dropout概率为0，即不使用Dropout操作。

3. 最后接两层512维的全连接。

4. 在这里，VGG网络首先提取高层特征，随后在全连接层中将其映射到和类别维度大小一致的向量上，最后通过Softmax方法计算图片划为每个类别的概率。

### ResNet

CIFAR10数据集上ResNet核心模块。

`resnet_cifar10`中的一些基本函数，再介绍网络连接过程。

  - `conv_bn_layer` : 带BN的卷积层。
  - `shortcut` : 残差模块的"直连"路径，"直连"实际分两种形式：残差模块输入和输出特征通道数不等时，采用1x1卷积的升维操作；残差模块输入和输出通道相等时，采用直连操作。
  - `basicblock` : 一个基础残差模块，即图9左边所示，由两组3x3卷积组成的路径和一条"直连"路径组成。
  - `layer_warp` : 一组残差模块，由若干个残差模块堆积而成。每组中第一个残差模块滑动窗口大小与其他可以不同，以用来减少特征图在垂直和水平方向的大小。

```python
def conv_bn_layer(input,
                  ch_out,
                  filter_size,
                  stride,
                  padding,
                  act='relu',
                  bias_attr=False):
    tmp = fluid.layers.conv2d(
        input=input,
        filter_size=filter_size,
        num_filters=ch_out,
        stride=stride,
        padding=padding,
        act=None,
        bias_attr=bias_attr)
    return fluid.layers.batch_norm(input=tmp, act=act)


def shortcut(input, ch_in, ch_out, stride):
    if ch_in != ch_out:
        return conv_bn_layer(input, ch_out, 1, stride, 0, None)
    else:
        return input


def basicblock(input, ch_in, ch_out, stride):
    tmp = conv_bn_layer(input, ch_out, 3, stride, 1)
    tmp = conv_bn_layer(tmp, ch_out, 3, 1, 1, act=None, bias_attr=True)
    short = shortcut(input, ch_in, ch_out, stride)
    return fluid.layers.elementwise_add(x=tmp, y=short, act='relu')


def layer_warp(block_func, input, ch_in, ch_out, count, stride):
    tmp = block_func(input, ch_in, ch_out, stride)
    for i in range(1, count):
        tmp = block_func(tmp, ch_out, ch_out, 1)
    return tmp
```

`resnet_cifar10` 的连接结构主要有以下几个过程。

1. 底层输入连接一层 `conv_bn_layer`，即带BN的卷积层。

2. 然后连接3组残差模块即下面配置3组 `layer_warp` ，每组采用图 10 左边残差模块组成。

3. 最后对网络做均值池化并返回该层。

注意：除第一层卷积层和最后一层全连接层之外，要求三组 `layer_warp` 总的含参层数能够被6整除，即 `resnet_cifar10` 的 depth 要满足 $(depth - 2) % 6 = 0$ 。

```python
def resnet_cifar10(ipt, depth=32):
    # depth should be one of 20, 32, 44, 56, 110, 1202
    assert (depth - 2) % 6 == 0
    n = (depth - 2) // 6
    nStages = {16, 64, 128}
    conv1 = conv_bn_layer(ipt, ch_out=16, filter_size=3, stride=1, padding=1)
    res1 = layer_warp(basicblock, conv1, 16, 16, n, 1)
    res2 = layer_warp(basicblock, res1, 16, 32, n, 2)
    res3 = layer_warp(basicblock, res2, 32, 64, n, 2)
    pool = fluid.layers.pool2d(
        input=res3, pool_size=8, pool_type='avg', pool_stride=1)
    predict = fluid.layers.fc(input=pool, size=10, act='softmax')
    return predict
```

## Infererence Program 配置

网络输入定义为 `data_layer` (数据层)，在图像分类中即为图像像素信息。CIFRAR10是RGB 3通道32x32大小的彩色图，因此输入数据大小为3072(3x32x32)。

```python
def inference_program():
    # The image is 32 * 32 with RGB representation.
    data_shape = [3, 32, 32]
    images = fluid.layers.data(name='pixel', shape=data_shape, dtype='float32')

    predict = resnet_cifar10(images, 32)
    # predict = vgg_bn_drop(images) # un-comment to use vgg net
    return predict
```

## Train Program 配置

然后我们需要设置训练程序 `train_program`。它首先从推理程序中进行预测。
在训练期间，它将从预测中计算 `avg_cost`。
在有监督训练中需要输入图像对应的类别信息，同样通过`fluid.layers.data`来定义。训练中采用多类交叉熵作为损失函数，并作为网络的输出，预测阶段定义网络的输出为分类器得到的概率信息。

**注意:** 训练程序应该返回一个数组，第一个返回参数必须是 `avg_cost`。训练器使用它来计算梯度。

```python
def train_program():
    predict = inference_program()

    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(cost)
    accuracy = fluid.layers.accuracy(input=predict, label=label)
    return [avg_cost, accuracy]
```

## Optimizer Function 配置

在下面的 `Adam optimizer`，`learning_rate` 是学习率，与网络的训练收敛速度有关系。

```python
def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.001)
```

## 训练模型

### Data Feeders 配置

`cifar.train10()` 每次产生一条样本，在完成shuffle和batch之后，作为训练的输入。

```python
# Each batch will yield 128 images
BATCH_SIZE = 128

# Reader for training
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.cifar.train10(), buf_size=50000),
    batch_size=BATCH_SIZE)

# Reader for testing. A separated data set for testing.
test_reader = paddle.batch(
    paddle.dataset.cifar.test10(), batch_size=BATCH_SIZE)
```

### Trainer 程序的实现
我们需要为训练过程制定一个main_program, 同样的，还需要为测试程序配置一个test_program。定义训练的 `place` ，并使用先前定义的优化器 `optimizer_func`。


```python
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

feed_order = ['pixel', 'label']

main_program = fluid.default_main_program()
star_program = fluid.default_startup_program()

avg_cost, acc = train_program()

# Test program
test_program = main_program.clone(for_test=True)

optimizer = optimizer_program()
optimizer.minimize(avg_cost)

exe = fluid.Executor(place)

EPOCH_NUM = 2

# For training test cost
def train_test(program, reader):
    count = 0
    feed_var_list = [
        program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder_test = fluid.DataFeeder(
        feed_list=feed_var_list, place=place)
    test_exe = fluid.Executor(place)
    accumulated = len([avg_cost, acc]) * [0]
    for tid, test_data in enumerate(reader()):
        avg_cost_np = test_exe.run(program=program,
                                   feed=feeder_test.feed(test_data),
                                   fetch_list=[avg_cost, acc])
        accumulated = [x[0] + x[1][0] for x in zip(accumulated, avg_cost_np)]
        count += 1
    return [x / count for x in accumulated]
```

### 训练主循环以及过程输出

在接下来的主训练循环中，我们将通过输出来来观察训练过程，或进行测试等。

也可以使用`plot`, 利用回调数据来打点画图:

```python
params_dirname = "image_classification_resnet.inference.model"

from paddle.utils.plot import Ploter

train_prompt = "Train cost"
test_prompt = "Test cost"
plot_cost = Ploter(test_prompt,train_prompt)

# main train loop.
def train_loop():
    feed_var_list_loop = [
        main_program.global_block().var(var_name) for var_name in feed_order
    ]
    feeder = fluid.DataFeeder(
        feed_list=feed_var_list_loop, place=place)
    exe.run(star_program)

    step = 0
    for pass_id in range(EPOCH_NUM):
        for step_id, data_train in enumerate(train_reader()):
            avg_loss_value = exe.run(main_program,
                                     feed=feeder.feed(data_train),
                                     fetch_list=[avg_cost, acc])
            if step % 1 == 0:
                plot_cost.append(train_prompt, step, avg_loss_value[0])
                plot_cost.plot()
            step += 1

        avg_cost_test, accuracy_test = train_test(test_program,
                                                  reader=test_reader)
        plot_cost.append(test_prompt, step, avg_cost_test)

        # save parameters
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ["pixel"],
                                          [predict], exe)
```

### 训练

通过`trainer_loop`函数训练, 这里我们只进行了2个Epoch, 一般我们在实际应用上会执行上百个以上Epoch

**注意:** CPU，每个 Epoch 将花费大约15～20分钟。这部分可能需要一段时间。请随意修改代码，在GPU上运行测试，以提高训练速度。

```python
train_loop()
```

一轮训练log示例如下所示，经过1个pass， 训练集上平均 Accuracy 为0.59 ，测试集上平均  Accuracy 为0.6 。

```text
Pass 0, Batch 0, Cost 3.869598, Acc 0.164062
...................................................................................................
Pass 100, Batch 0, Cost 1.481038, Acc 0.460938
...................................................................................................
Pass 200, Batch 0, Cost 1.340323, Acc 0.523438
...................................................................................................
Pass 300, Batch 0, Cost 1.223424, Acc 0.593750
..........................................................................................
Test with Pass 0, Loss 1.1, Acc 0.6
```

图13是训练的分类错误率曲线图，运行到第200个pass后基本收敛，最终得到测试集上分类错误率为8.54%。

<p align="center">
<img src="https://github.com/PaddlePaddle/book/blob/develop/03.image_classification/image/plot.png?raw=true" width="400" ><br/>
图13. CIFAR10数据集上VGG模型的分类错误率
</p>

## 应用模型

可以使用训练好的模型对图片进行分类，下面程序展示了如何加载已经训练好的网络和参数进行推断。

### 生成预测输入数据

`dog.png` 是一张小狗的图片. 我们将它转换成 `numpy` 数组以满足`feeder`的格式.

```python
# Prepare testing data.
from PIL import Image
import os

def load_image(file):
    im = Image.open(file)
    im = im.resize((32, 32), Image.ANTIALIAS)

    im = numpy.array(im).astype(numpy.float32)
    # The storage order of the loaded image is W(width),
    # H(height), C(channel). PaddlePaddle requires
    # the CHW order, so transpose them.
    im = im.transpose((2, 0, 1))  # CHW
    im = im / 255.0

    # Add one dimension to mimic the list format.
    im = numpy.expand_dims(im, axis=0)
    return im

cur_dir = os.getcwd()
img = load_image(cur_dir + '/image/dog.png')
```

### Inferencer 配置和预测

inferencer需要构建相应的过程。我们从`params_dirname` 加载网络和经过训练的参数。
我们可以简单地插入前面定义的推理程序。
现在我们准备做预测。

```python
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
inference_scope = fluid.core.Scope()

with fluid.scope_guard(inference_scope):

    [inference_program, feed_target_names,
     fetch_targets] = fluid.io.load_inference_model(params_dirname, exe)

        # The input's dimension of conv should be 4-D or 5-D.
        # Use inference_transpiler to speedup
    inference_transpiler_program = inference_program.clone()
    t = fluid.transpiler.InferenceTranspiler()
    t.transpile(inference_transpiler_program, place)

        # Construct feed as a dictionary of {feed_target_name: feed_target_data}
        # and results will contain a list of data corresponding to fetch_targets.
    results = exe.run(inference_program,
                      feed={feed_target_names[0]: img},
                      fetch_list=fetch_targets)

    transpiler_results = exe.run(inference_transpiler_program,
                                 feed={feed_target_names[0]: img},
                                 fetch_list=fetch_targets)

    assert len(results[0]) == len(transpiler_results[0])
    for i in range(len(results[0])):
        numpy.testing.assert_almost_equal(
            results[0][i], transpiler_results[0][i], decimal=5)

    # infer label
    label_list = [
        "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
        "ship", "truck"
    ]

    print("infer results: %s" % label_list[numpy.argmax(results[0])])
```

## 总结

传统图像分类方法由多个阶段构成，框架较为复杂，而端到端的CNN模型结构可一步到位，而且大幅度提升了分类准确率。
