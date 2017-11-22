# 课后作业：使用 LSTM 编写一个国际姓氏生成模型
<pre>
# 第一步当然是引入PyTorch及相关包
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
import numpy as np
</pre>
## 一 准备数据
<pre>
import glob
import unicodedata
import string
</pre>
<pre>
# all_letters 即课支持打印的字符+标点符号
all_letters = string.ascii_letters + " .,;'-"
# Plus EOS marker
n_letters = len(all_letters) + 1 
EOS = n_letters - 1

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicode_to_ascii("O'Néàl"))
O'Neal
# 姓氏中所有的可视字符
print('all_letters: ', all_letters)
# 所有字符的长度 +1 EOS结束符
print('n_letters: ', n_letters)
# 结束符，没有实质内容,索引从0开始，所以这是字符表长度减1
print('EOS: ', EOS)
</pre>
## 二 读取数据
#我们建立一个列表 all_categories 用于存储所有的国家名字。
#建立一个字典 category_lines，以读取的国名作为字典的索引，国名下存储对应国别的名字。
#按行读取出文件中的名字，并返回包含所有名字的列表
<pre>
def read_lines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]
</pre>
<pre>
# category_lines是一个字典
# 其中索引是国家名字，内容是从文件读取出的这个国家的所有名字
category_lines = {}
# all_categories是一个列表
# 其中包含了所有的国家名字
all_categories = []
# 循环所有文件
for filename in glob.glob('./names/*.txt'):
    # 从文件名中切割出国家名字
    category = filename.split('/')[-1].split('.')[0]
    # 将国家名字添加到列表中
    all_categories.append(category)
    # 读取对应国别文件中所有的名字
    lines = read_lines(filename)
    # 将所有名字存储在字典中对应的国别下
    category_lines[category] = lines

# 共有的国别数
n_categories = len(all_categories)

print('# categories: ', n_categories, all_categories)
print()
print('# Russian names: ', category_lines['Russian'][:10])
# categories:  18 ['Arabic', 'Italian', 'Irish', 'Greek', 'Vietnamese', 'Spanish', 'Russian', 'Polish', 'Portuguese', 'Korean', 'German', 'Dutch', 'Chinese', 'Czech', 'Japanese', 'Scottish', 'French', 'English']

# Russian names:  ['Ababko', 'Abaev', 'Abagyan', 'Abaidulin', 'Abaidullin', 'Abaimoff', 'Abaimov', 'Abakeliya', 'Abakovsky', 'Abakshin']
</pre>
#再统计下手头共有多少条训练数据
<pre>
all_line_num = 0
for key in category_lines:
    all_line_num += len(category_lines[key])
print(all_line_num)
20074
</pre>
## 三.准备训练
#首先建立一个可以随机选择数据对 (category, line) 的方法，以方便训练时调用。
<pre>
import random
def random_training_pair():
    #随机选择一个国别名
    category = random.choice(all_categories)
    #读取这个国别名下的所有人名
    line = random.choice(category_lines[category])
    return category, line
print(random_training_pair())
</pre>
#将名字所属的国家名转化为“独热向量”
<pre>
def make_category_input(category):
    li = all_categories.index(category)
    return  li
print(make_category_input('Chinese'))
</pre>
#对于训练过程中的每一步，或者说对于训练数据中每个名字的每个字符来说，神经网络的输入是 (category, current letter, hidden state)，输出是 (next letter, next hidden state)。

#与在课程中讲的一样，神经网络还是依据“当前的字符”预测“下一个字符”。比如对于“Kasparov”这个名字，创建的（input, target）数据对是 ("K", "a"), ("a", "s"), ("s", "p"), ("p", "a"), ("a", "r"), ("r", "o"), ("o", "v"), ("v", "EOS")。
<pre>
def make_chars_input(nameStr):
    name_char_list = list(map(lambda x: all_letters.find(x), nameStr))
    return name_char_list


def make_target(nameStr):
    target_char_list = list(map(lambda x: all_letters.find(x), nameStr[1:]))
    target_char_list.append(n_letters - 1)# EOS
    return target_char_list
</pre> 
#同样为了训练时方便使用，我们建立一个 random_training_set 函数，以随机选择出数据集 (category, line) 并转化成训练需要的 Tensor： (category, input, target)。
<pre>
def random_training_set():
    #随机选择数据集
    category, line = random_training_pair()
    #print(category, line)
    #转化成对应 Tensor
    category_input = make_category_input(category)
    line_input = make_chars_input(line)
    
    line_target = make_target(line)
    return category_input, line_input, line_target
    
</pre> 
## 四 搭建神经网络

#一个手动实现的LSTM模型，
<pre>
class LSTMNetwork(nn.Module):
    def __init__(self, category_size, name_size, hidden_size, output_size, num_layers = 1):
        super(LSTMNetwork, self).__init__()
        self.category_size=category_size
        self.hidden_size = hidden_size
        self.num_layers =num_layers
        self.name_size=name_size
        #进行嵌入
        self.embedding=nn.Embedding(category_size+name_size,hidden_size)
        
        #隐含层内部的相互链接
        self.lstm=nn.LSTM(hidden_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
        #输出层
        self.softmax=nn.LogSoftmax()

    def forward(self, input, hidden):
        
        #先分别进行embedding层的计算
        embedded=self.embedding(input)
        #embedded=embedded.view(input.data.size()[0],1, self.hidden_size)
        #从输入到隐含层的计算
        output, hidden = self.lstm(embedded, hidden)
        
        #output的尺寸：batch_size, len_seq, hidden_size
        output = output[:,-1,:]
        #全连接层
        output=self.fc(output)

        #output的尺寸：batch_size, output_size
        #softmax函数
        output=self.softmax(output)
        
        return output, hidden
 
    def initHidden(self):
        #对隐含单元的初始化
        #注意尺寸是： layer_size, batch_size, hidden_size
        #对隐单元的初始化
        #对引单元输出的初始化，全0.
        #注意hidden和cell的维度都是layers,batch_size,hidden_size
        hidden = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        #对隐单元内部的状态cell的初始化，全0
        cell = Variable(torch.zeros(self.num_layers, 1, self.hidden_size))
        return (hidden, cell)
</pre> 
#开始训练
<pre> 
#定义训练函数，在这个函数里，我们可以随机选择一条训练数据，遍历每个字符进行训练,指的是每一个姓氏的损失
def train_LSTM():
    #初始化 隐藏层、梯度清零、损失清零
    hidden=lstm.initHidden()
    optimizer.zero_grad()
    loss=0
    
    #随机选取一条训练数据
    category_input, line_input, line_target = random_training_set()
    #处理国别数据
    category_variable =Variable(torch.LongTensor(np.array([category_input])))
    
    
    #循环字符
    for t in range(len(line_input)):
        #姓氏
        name_variable = Variable(torch.LongTensor([line_input[t]]).unsqueeze(0))
        x= torch.cat((category_variable,name_variable), )
        #目标
        y_target=Variable(torch.LongTensor([line_target[t]]))
        y=torch.cat((category_variable,y_target),)
        #传入模型
        output,hidden=lstm(x,hidden)
        #累加损失
        loss+= cost(output, y)
        #计算平均损失
    loss = 1.0 * loss / len(line_input)
    #反向传播、更新梯度
    loss.backward(retain_variables = True)
    optimizer.step()
    
    return loss
</pre>
<pre>
import time
import math

def time_since(t):
    now = time.time()
    s = now - t
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
</pre>
<pre>
hidden_size = 10
num_epoch = 3
learning_rate = 0.001

#实例化模型
lstm = LSTMNetwork(n_categories, n_letters, hidden_size, n_letters, num_layers = 1)
#定义损失函数与优化方法
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)
cost = nn.NLLLoss()
</pre>
#训练开始
<pre>
start = time.time()

records = []
# 开始训练循环
for epoch in range(num_epoch):
    train_loss=0
    # 按所有数据的行数随机循环
    for i in range(all_line_num):
        loss = train_LSTM()
        train_loss+=loss
        
        #每隔3000步，跑一次校验集，并打印结果
        if i % 3000 == 0:#还可以采用if i>0
            training_process = (all_line_num * epoch + i) / (all_line_num * num_epoch) * 100
            training_process = '%.2f' % training_process
            print('第{}轮，训练损失：{:.2f}，训练进度：{}%，（{}）'\
                .format(epoch, train_loss.data.numpy()[0]/(i+1), float(training_process), time_since(start)))
            records.append([train_loss.data.numpy()[0]/(i+1)])
第0轮，训练损失：3.91，训练进度：0.0%，（0m 0s）
第0轮，训练损失：2.36，训练进度：4.98%，（0m 39s）
第0轮，训练损失：2.00，训练进度：9.96%，（1m 17s）
第0轮，训练损失：1.82，训练进度：14.94%，（1m 58s）
第0轮，训练损失：1.71，训练进度：19.93%，（2m 37s）
第0轮，训练损失：1.64，训练进度：24.91%，（3m 23s）
第0轮，训练损失：1.59，训练进度：29.89%，（5m 10s）
第1轮，训练损失：1.63，训练进度：33.33%，（7m 33s）
第1轮，训练损失：1.28，训练进度：38.31%，（8m 14s）
第1轮，训练损失：1.28，训练进度：43.3%，（8m 53s）
第1轮，训练损失：1.27，训练进度：48.28%，（9m 35s）
第1轮，训练损失：1.27，训练进度：53.26%，（10m 35s）
第1轮，训练损失：1.26，训练进度：58.24%，（11m 46s）
第1轮，训练损失：1.26，训练进度：63.22%，（13m 35s）
第2轮，训练损失：1.24，训练进度：66.67%，（16m 25s）
第2轮，训练损失：1.23，训练进度：71.65%，（17m 5s）
第2轮，训练损失：1.23，训练进度：76.63%，（17m 46s）
第2轮，训练损失：1.23，训练进度：81.61%，（18m 32s）
第2轮，训练损失：1.23，训练进度：86.59%，（19m 30s）
第2轮，训练损失：1.23，训练进度：91.57%，（20m 12s）
第2轮，训练损失：1.23，训练进度：96.56%，（21m 14s）
</pre>
<pre>
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline
a = [i[0] for i in records]
plt.plot(a, label = 'Train Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
</pre>
![](http://upload-images.jianshu.io/upload_images/7539367-0cf974eaa53166e7.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
<pre>
start = time.time()

records = []
train_losses=[]
# 开始训练循环
for epoch in range(num_epoch):
    # 按所有数据的行数随机循环
    for i in range(all_line_num):
        loss = train_LSTM()
        train_losses.append(loss.data.numpy()[0])
        
        #每隔3000步，跑一次校验集，并打印结果
        if i % 3000 == 0:
            training_process = (all_line_num * epoch + i) / (all_line_num * num_epoch) * 100
            training_process = '%.2f' % training_process
            print('第{}轮，训练损失：{:.2f}，训练进度：{}%，（{}）'\
                .format(epoch, np.mean(train_losses), float(training_process), time_since(start)))
            records.append([np.mean(train_losses)])
第0轮，训练损失：0.57，训练进度：0.0%，（0m 1s）
第0轮，训练损失：1.20，训练进度：4.98%，（0m 40s）
第0轮，训练损失：1.20，训练进度：9.96%，（1m 19s）
第0轮，训练损失：1.19，训练进度：14.94%，（1m 59s）
第0轮，训练损失：1.19，训练进度：19.93%，（2m 38s）
第0轮，训练损失：1.19，训练进度：24.91%，（3m 16s）
第0轮，训练损失：1.19，训练进度：29.89%，（3m 54s）
第1轮，训练损失：1.19，训练进度：33.33%，（4m 18s）
第1轮，训练损失：1.19，训练进度：38.31%，（4m 56s）
第1轮，训练损失：1.19，训练进度：43.3%，（5m 33s）
第1轮，训练损失：1.19，训练进度：48.28%，（6m 11s）
第1轮，训练损失：1.19，训练进度：53.26%，（6m 51s）
第1轮，训练损失：1.19，训练进度：58.24%，（7m 31s）
第1轮，训练损失：1.19，训练进度：63.22%，（8m 11s）
第2轮，训练损失：1.19，训练进度：66.67%，（8m 38s）
第2轮，训练损失：1.19，训练进度：71.65%，（9m 18s）
第2轮，训练损失：1.19，训练进度：76.63%，（9m 58s）
第2轮，训练损失：1.19，训练进度：81.61%，（10m 37s）
第2轮，训练损失：1.19，训练进度：86.59%，（11m 14s）
第2轮，训练损失：1.19，训练进度：91.57%，（11m 52s）
第2轮，训练损失：1.19，训练进度：96.56%，（12m 29s）
</pre>
<pre>
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
%matplotlib inline

a = [i[0] for i in records]
plt.plot(a, label = 'Train Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
</pre>
![](http://upload-images.jianshu.io/upload_images/7539367-fb4ab28a18addcee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
<pre>
category_input, line_input, line_target = random_training_set()
print(category_input,line_input,line_target)

category_variable =Variable(torch.LongTensor(np.array([category_input])))
print(category_variable)
12 [33, 14, 13, 6] [14, 13, 6, 58]
Variable containing:
 12
[torch.LongTensor of size 1]
</pre>
<pre>
hidden=lstm.initHidden()
print(line_input)
for t in range(len(line_input)):
        # 姓氏
        name_variable = Variable(torch.LongTensor([line_input[t]]).unsqueeze(0))
        x= torch.cat((category_variable,name_variable), )
        print("x:",x)
        # 目标
        y_target=Variable(torch.LongTensor([line_target[t]]))
        y=torch.cat((category_variable,y_target),)
        print("y:",y)
        output,hidden=lstm(x,hidden)
        print("output:",output)
        #将输出转化成一个多项式分布
        output_dist = output.data[1].view(-1).div(0.2).exp()
        print("\n output_dist:",output_dist)
        # 从而可以根据混乱度 temperature 来选择下一个字符
        # 混乱度低，则趋向于选择网络预测最大概率的那个字符
        # 混乱度高，则趋向于随机选择字符
        top_i = torch.multinomial(output_dist,1)[0]
        print("top_i:",top_i)
        # 继续下一个字符
        char = all_letters[top_i]
        print(char)
        print("\n\n\n")
[33, 14, 13, 6]
x: Variable containing:
 12
 33
[torch.LongTensor of size 2x1]

y: Variable containing:
 12
 14
[torch.LongTensor of size 2]

output: Variable containing:

Columns 0 to 7 
 -7.6223 -16.3712  -5.2575  -7.4323 -14.1191 -12.7495  -6.0463  -7.9841
 -0.9720  -6.2958  -5.9078  -6.7603  -2.3781  -8.6197  -8.6854  -5.8696

Columns 8 to 15 
-13.1167  -9.3242 -11.9493  -7.2719  -0.0173  -8.6945  -5.8931 -15.2512
 -2.5721  -9.0082  -9.6383  -5.5788  -7.2735  -7.4884  -1.3644  -5.2803

Columns 16 to 23 
-13.7072  -5.5850  -9.6204 -10.0998  -9.7830 -11.5185  -9.6959 -11.9997
 -8.0350  -4.4532  -5.7749  -6.0160  -1.9541  -8.4552  -5.7726 -10.7585

Columns 24 to 31 
-12.6510 -10.7926 -14.5638  -9.1270 -10.7655 -10.3632 -22.4923 -16.2821
 -4.4812  -6.8534 -14.5945 -14.0463 -15.9115 -15.9477 -24.3491 -15.9836

Columns 32 to 39 
-11.7907 -10.3027 -21.7551 -22.4805 -10.0079 -11.2759 -10.5400  -9.9822
-15.4044 -15.8451 -23.5043 -24.3488 -14.6396 -15.2765 -15.8740 -14.5124

Columns 40 to 47 
-18.7455 -16.4313 -22.4932 -11.0103 -11.3857 -11.4640 -22.0709 -22.5495
-20.0916 -15.8277 -24.4041 -15.4113 -14.8090 -13.9283 -23.9275 -24.4049

Columns 48 to 55 
-22.5649 -22.5017 -11.5057 -22.4812 -11.7662 -22.5529 -21.8164 -22.5007
-24.4157 -24.4717 -13.1533 -24.4116  -8.0476 -24.4036 -23.5169 -24.4438

Columns 56 to 58 
-17.8170 -11.1635 -14.3949
 -6.0342 -11.3352  -8.6593
[torch.FloatTensor of size 2x59]


 output_dist: 
1.00000e-03 *
  7.7509
  0.0000
  0.0000
  0.0000
  0.0069
  0.0000
  0.0000
  0.0000
  0.0026
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  1.0893
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0571
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
[torch.FloatTensor of size 59]

top_i: 0
a




x: Variable containing:
 12
 14
[torch.LongTensor of size 2x1]

y: Variable containing:
 12
 13
[torch.LongTensor of size 2]

output: Variable containing:

Columns 0 to 7 
 -9.8484 -21.6479  -6.7175  -9.2568 -17.9819 -16.4458  -8.1086 -10.9628
 -3.7190  -3.8813  -3.5973  -3.4396  -3.1877  -4.8013  -4.0024  -5.2023

Columns 8 to 15 
-17.4344 -11.2084 -15.4250  -9.1383  -0.0035 -11.3049  -7.2729 -19.1732
 -3.3735  -5.2561  -4.3942  -2.7652  -2.9937  -1.8777  -2.7858  -3.8607

Columns 16 to 23 
-17.4324  -7.1356 -12.1753 -13.1811 -12.7822 -14.5527 -12.3614 -14.3258
 -6.6375  -2.2872  -2.6011  -3.4582  -2.7333  -3.6812  -4.0732  -6.9907

Columns 24 to 31 
-16.0475 -13.7465 -16.4628 -10.1410 -11.8363 -11.4294 -23.9009 -18.2947
 -3.7885  -4.6344 -14.8312 -13.5196 -16.4278 -16.1359 -29.2842 -17.0814

Columns 32 to 39 
-13.2224 -11.3683 -23.1856 -23.8724 -11.2256 -12.5969 -11.5495 -11.1336
-16.0236 -15.8784 -28.3114 -29.4837 -14.0891 -15.5775 -16.3510 -13.9354

Columns 40 to 47 
-20.3654 -18.5305 -23.8893 -12.2066 -12.7318 -13.0461 -23.4626 -23.9377
-22.6497 -16.8527 -29.4361 -15.5311 -15.4309 -14.4771 -28.9107 -29.5442

Columns 48 to 55 
-23.9510 -23.8879 -12.8537 -23.8778 -14.2513 -23.9275 -23.2716 -23.8964
-29.5162 -29.5469 -15.7299 -29.4876  -6.4270 -29.6345 -27.7614 -29.4312

Columns 56 to 58 
-21.6193 -12.6709 -17.5237
-11.1401 -11.6405  -2.6431
[torch.FloatTensor of size 2x59]


 output_dist: 
1.00000e-05 *
  0.0008
  0.0004
  0.0015
  0.0034
  0.0120
  0.0000
  0.0002
  0.0000
  0.0047
  0.0000
  0.0000
  0.0989
  0.0316
  8.3679
  0.0893
  0.0004
  0.0000
  1.0799
  0.2248
  0.0031
  0.1161
  0.0010
  0.0001
  0.0000
  0.0006
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.1823
[torch.FloatTensor of size 59]

top_i: 13
n




x: Variable containing:
 12
 13
[torch.LongTensor of size 2x1]

y: Variable containing:
 12
  6
[torch.LongTensor of size 2]

output: Variable containing:

Columns 0 to 7 
-10.1132 -22.5432  -6.9783  -9.3914 -18.6288 -16.8539  -8.3305 -11.2952
 -3.2818  -5.7059  -4.0053  -2.5401  -2.8539  -4.3284  -1.4368  -4.4980

Columns 8 to 15 
-18.1749 -11.3362 -16.0763  -9.4403  -0.0028 -11.8083  -7.4399 -19.7961
 -3.9043  -5.2345  -3.8715  -4.7489  -3.7370  -2.9453  -3.3979  -6.1179

Columns 16 to 23 
-17.7172  -7.2461 -12.4389 -13.6253 -13.3423 -14.9741 -12.7798 -14.5385
 -8.3525  -6.1223  -3.4481  -2.6818  -6.0232  -6.1723  -7.1689  -7.6865

Columns 24 to 31 
-16.5781 -14.1671 -16.6672 -10.2388 -11.9071 -11.4836 -24.0957 -18.5318
 -5.3841  -5.2731 -14.8088 -13.0912 -16.1658 -15.7097 -28.5508 -16.6106

Columns 32 to 39 
-13.3625 -11.4262 -23.3826 -24.0657 -11.3387 -12.7116 -11.6070 -11.2504
-15.7894 -15.4293 -27.6152 -28.6834 -14.0546 -15.5056 -16.0907 -13.5235

Columns 40 to 47 
-20.5725 -18.7841 -24.0835 -12.2997 -12.8803 -13.2517 -23.6553 -24.1282
-22.2584 -16.4106 -28.6622 -15.4788 -15.4669 -14.3619 -28.1727 -28.7496

Columns 48 to 55 
-24.1373 -24.0791 -13.0862 -24.0728 -14.5257 -24.1138 -23.4648 -24.0879
-28.6869 -28.7054 -17.3247 -28.6793  -5.8040 -28.7881 -27.0809 -28.6097

Columns 56 to 58 
-22.0021 -12.8033 -17.7347
-11.4931 -12.1935  -1.3695
[torch.FloatTensor of size 2x59]


 output_dist: 
1.00000e-03 *
  0.0001
  0.0000
  0.0000
  0.0030
  0.0006
  0.0000
  0.7586
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0004
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0015
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  0.0000
  1.0619
[torch.FloatTensor of size 59]

top_i: 58
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
<ipython-input-139-7355492bf383> in <module>()
     21         print("top_i:",top_i)
     22         # 继续下一个字符
---> 23         char = all_letters[top_i]
     24         print(char)
     25         print("\n\n\n")

IndexError: string index out of range
</pre>
## 五 测试神经网络
<pre>
max_length = 20

# 通过指定国别名 category
# 以及开始字符 start_char
# 还有混乱度 temperature 来生成一个名字
def generate_one(category, start_char='A', temperature=0.2):
    # 初始化输入数据，国别 以及 输入的第一个字符
    # 国别
    category_index=make_category_input(category)#将国别索引成序号
    category_variable =Variable(torch.LongTensor(np.array([category_index])))#将编码转换成variable形式
    # 第一个字符
    chars_input=make_chars_input(start_char)#将字符进行编码
    #将编码转换成variable形式
    name_variable=Variable(torch.LongTensor([chars_input]))
    #将国别和第一个字符的variable形式合并，并以input形式喂给lstm forward
    input=torch.cat((category_variable,name_variable), )
    # 初始化隐藏层
    hidden=lstm.initHidden()

    output_str = start_char
    
    for i in range(max_length):
        
        # 调用模型
        output, hidden = lstm(input,hidden)
        # 这里是将输出转化为一个多项式分布
        #!!!!!!!output是2×59形式的张量，第一行代表国别，第二行才代表字符，如果默认取第一行的话，
        #output_dist的维度是2×59=118,很大部分是国别的编码，这样生成的字符的编码很大概率是国别的编码
        output_dist = output.data[1].view(-1).div(temperature).exp()
        # 从而可以根据混乱度 temperature 来选择下一个字符
        # 混乱度低，则趋向于选择网络预测最大概率的那个字符
        # 混乱度高，则趋向于随机选择字符
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # 生成字符是 EOS，则生成结束
        if top_i == EOS:
            break
        else:
            # 继续下一个字符
            char = all_letters[top_i]
            output_str += char
            chars_input = make_chars_input(char)
            name_variable = Variable(torch.LongTensor([chars_input]))
            input=torch.cat((category_variable,name_variable), )# 为了循环，输出的生成字符作为下一个循环的输入
            
    return output_str

# 再定义一个函数，方便每次生成多个名字
def generate(category, start_chars='a'):
    for start_char in start_chars:
        print(generate_one(category, start_char))
</pre>
<pre>
generate('Spanish', 'SPN')
San
Para
Nas
</pre>
<pre>
generate('Russian', 'RUs')
Rig
Ung
san
</pre>
<pre>
generate('Spanish', 'spS')
san
ppppppppppppppppppppp
Sara
</pre>
<pre>
generate('Spanish', 'SPA')
Sau
Pang
Alan
</pre>
<pre>
generate('Chinese', 'Chi')
Chin
hhhhhhhhhhhhhhhhhhhhh
iiiiiiiiiiiiiiiiiiiii
</pre>
<pre>
generate('Chinese','CHI')
Cher
Han
Irar
</pre>
<pre>
generate_one('Chinese','M')
'Mara'
</pre>
<pre>
generate_one('Korean','M')
'Macha'
</pre>
