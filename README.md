# tensorflow-speech-recognition-pai
Speech recognition using tensorflow in aliyun pai.

阿里云深度学习pai下用tensorflow实现的语音识别。

# Tutorial

## 数据来源
LibriSpeech ASR corpus http://www.openslr.org/12

## 预处理数据
我使用的是dev-clean,test-clean,train-clean-100.

下载下来的数据音频文件是flac格式。还有label是单个txt格式。

三个脚本预处理一下数据。

##### 首先转flac到wav 需要安装ffmpeg

###### ./flactowav.sh dev-clean

##### 接着把单个的label文件处理成对应多个

###### python processAllTxt.py dev-clean

###### 最后用pickle把数据处理对应直接读取的对象

##### 因为pai不能直接用各种原生的读取文件函数，比如scipy里读取音频之类的。

###### python wavtopickles.py dev-clean dev-data  #第一个参数是文件的目录，第二个是生成的文件目录

建议在阿里云华东2内网操作，因为生成出来有几十个g，内网传上去比较快。

我是买那种按小时收费的机子处理数据，装一下cpu版的tensorflow，处理完挂载ossfs传上oss。

## 训练模型

###### python speech_train.py #本地

pai上就选好文件个输入输出目录就可以了。

## 查看TensorBoard
输出目录的 model/nn/debug_models/summary 是TensorBoard的目录，pai上选择该目录可以查看TensorBoard
