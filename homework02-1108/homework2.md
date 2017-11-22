# 一.处理训练数据
<pre>
import glob
all_filenames = glob.glob('./data/names/*.txt')
print(all_filenames)
['./data/names/Arabic.txt', './data/names/Italian.txt', './data/names/Irish.txt', './data/names/Greek.txt', './data/names/Vietnamese.txt', './data/names/Spanish.txt', './data/names/Russian.txt', './data/names/Polish.txt', './data/names/Portuguese.txt', './data/names/Korean.txt', './data/names/German.txt', './data/names/Dutch.txt', './data/names/Chinese.txt', './data/names/Czech.txt', './data/names/Japanese.txt', './data/names/Scottish.txt', './data/names/French.txt', './data/names/English.txt']
</pre>
<pre>
import unicodedata
import string

# 使用26个英文字母大小写再加上.,;这三个字符
# 建立字母表，并取其长度
all_letters = string.ascii_letters #+ " .,;'"
n_letters = len(all_letters)


# 将Unicode字符串转换为纯ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

print(unicode_to_ascii('Ślusàrski'))
print('all_letters:', all_letters)
print('all_letters:', len(all_letters))
Slusarski
all_letters: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
all_letters: 52
</pre>
#以18种语言为索引，将读取出的姓氏各自存储在名为 category_lines 的字典中。
<pre>
# 构建category_lines字典，名字和每种语言对应的列表
category_lines = {}
all_categories = []

# 按行读取出名字并转换成纯ASCII
def readLines(filename):
    lines = open(filename).read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

for filename in all_filenames:
    # 取出每个文件的文件名（语言名）
    category = filename.split('/')[-1].split('.')[0]
    # 将语言名加入到all_categories列表
    all_categories.append(category)
    # 取出所有的姓氏lines
    lines = readLines(filename)
    # 将所有姓氏以语言为索引，加入到字典中
    category_lines[category] = lines

n_categories = len(all_categories)

print('all_categories:', all_categories)
print('n_categories =', n_categories)
all_categories: ['Arabic', 'Italian', 'Irish', 'Greek', 'Vietnamese', 'Spanish', 'Russian', 'Polish', 'Portuguese', 'Korean', 'German', 'Dutch', 'Chinese', 'Czech', 'Japanese', 'Scottish', 'French', 'English']
n_categories = 18
</pre>
<pre>
all_line_num = 0
for key in category_lines:
    all_line_num += len(category_lines[key])
print(all_line_num)
</pre>
# 二.准备训练
<pre>
# 首先导入程序所需要的程序包

#PyTorch用的包
import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable


#绘图、计算用的程序包
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

%matplotlib inline
</pre>
<pre>
line='abc'
result=[]
for item in line:
    
    list=['a','b','c','d','e','f','g','h','i']
    result.append(list.index(item))
print(result)
[0, 1, 2]
</pre>
#用line_index来保存姓氏的索引
<pre>
import random

def random_training_pair():   
    #随机选择一种语言
    category = random.choice(all_categories)
    #从语言中随机选择一个姓氏
    line = random.choice(category_lines[category])
    #我们将姓氏和语言都转化为索引
    category_index = all_categories.index(category)
    
    line_index = []
    #你需要把 line 中字母的索引加入到line_index 中
    #Todo:
    for item in line:
        list=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
              'W','X','Y','Z']#," ",".",",",";","'"
        line_index.append(list.index(item))
    return category, line, category_index, line_index

#测试一下上面的函数方法
for i in range(5):
    category, line, category_index, line_index = random_training_pair()
    print('category =', category, '/ line =', line)
    print('category =', category_index, '/ line =', line_index)
category = Czech / line = Mojjis
category = 13 / line = [38, 14, 9, 9, 8, 18]
category = Korean / line = Lim
category = 9 / line = [37, 8, 12]
category = Korean / line = Yoo
category = 9 / line = [50, 14, 14]
category = Dutch / line = Romijnders
category = 11 / line = [43, 14, 12, 8, 9, 13, 3, 4, 17, 18]
category = Chinese / line = Cao
category = 12 / line = [28, 0, 14]
</pre>
#我们再建立一个用户转化模型输出的辅助函数。
#它可以把网络的输出（1 x 18的张量）转化成“最可能的语言类别”，这就需要找到18列数据中哪个概率值最大。
#我们可以使用 Tensor.topk 方法来得到数据中最大值位置的索引。
<pre>
def category_from_output(output):
    # 1 代表在‘列’间找到最大
    # top_n 是具体的值
    # top_i 是位置索引
    # 注意这里 top_n 和 top_i 都是1x1的张量
    # output.data 取出张量数据
    top_n, top_i = output.data.topk(1) # Tensor out of Variable with .data
    # 从张量中取出索引值
    category_i = top_i[0][0]
    # 返回语言类别名和位置索引
    return all_categories[category_i], category_i
</pre>
# 三.编写LSTM
<pre>
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2,batch_size,output_size, n_layers):
        super(LSTMNetwork, self).__init__()
        self.n_layers = n_layers
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.batch_size = batch_size
        # LSTM的构造如下：
        # 一个embedding层，将输入的任意一个单词（list）映射为一个向量（向量的维度与隐含层有关系？）
        self.embedding = nn.Embedding(input_size,hidden_size1)
        # 然后是一个LSTM隐含层，共有hidden_size个LSTM神经元，并且它可以根据n_layers设置层数
        self.lstm = nn.LSTM(hidden_size1,hidden_size2,n_layers)
        # 接着是一个全链接层，外接一个softmax输出
        self.fc = nn.Linear(hidden_size2,output_size)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, input, hidden=None):
        #首先根据输入input，进行词向量嵌入
        embedded = self.embedding(input)
        
        # 这里需要注意！
        # PyTorch设计的LSTM层有一个特别别扭的地方是，输入张量的第一个维度需要是时间步，
        # 第二个维度才是batch_size，所以需要对embedded变形
        # 因为此次没有采用batch，所以batch_size为1
        # 变形的维度应该是（input_list_size, batch_size, hidden_size）
        embedded = embedded.view(input.data.size()[0],self.batch_size, self.hidden_size1)
    
        # 调用PyTorch自带的LSTM层函数，注意有两个输入，一个是输入层的输入，另一个是隐含层自身的输入
        # 输出output是所有步的隐含神经元的输出结果，hidden是隐含层在最后一个时间步的状态。
        # 注意hidden是一个tuple，包含了最后时间步的隐含层神经元的输出，以及每一个隐含层神经元的cell的状态
        
        output, hidden = self.lstm(embedded, hidden)
        
        #我们要把最后一个时间步的隐含神经元输出结果拿出来，送给全连接层
        output = output[-1,...]

        #全链接层
        out = self.fc(output)
        # softmax
        out = self.logsoftmax(out)
        return out

    def initHidden(self):
        # 对隐单元的初始化
        # 对引单元输出的初始化，全0.
        # 注意hidden和cell的维度都是layers,batch_size,hidden_size
        hidden = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size2))
        # 对隐单元内部的状态cell的初始化，全0
        cell = Variable(torch.zeros(self.n_layers, self.batch_size, self.hidden_size2))
        return (hidden, cell)
</pre>
# 四.训练网络
<pre>
import time
import math

# 开始训练LSTM网络
n_epochs = 5

# 构造一个LSTM网络的实例
lstm = LSTMNetwork(n_letters, 50,9,1, n_categories, 2)

#定义损失函数
cost = torch.nn.NLLLoss()

#定义优化器,
optimizer = torch.optim.Adam(lstm.parameters(), lr = 0.001)
records = []

# 用于计算训练时间的函数
def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()

# 开始训练，一共5个epoch，否则容易过拟合
losses = []
for epoch in range(n_epochs):
    
    #每次随机选择数据进行训练，每个 EPOCH 训练“所有名字个数”次。y代表语言的编号index，x代表姓氏的编号index
    for i in range(all_line_num):
        category, line, y, x = random_training_pair()
        x = Variable(torch.LongTensor(x))
        y = Variable(torch.LongTensor(np.array([y])))
        optimizer.zero_grad()
        
        # Step1:初始化LSTM隐含层单元的状态
        hidden = lstm.initHidden()
        
        # Step2:让LSTM开始做运算，注意，不需要手工编写对时间步的循环，而是直接交给PyTorch的LSTM层。
        # 它自动会根据数据的维度计算若干时间步
        output = lstm(x,hidden)
        
        # Step3:计算损失
        loss = cost(output,y)
        losses.append(loss.data.numpy()[0])
        
        #反向传播
        loss.backward()
        optimizer.step()
        
        #每隔3000步，跑一次校验集，并打印结果
        if i % 3000 == 0:
            # 判断模型的预测是否正确
            guess, guess_i = category_from_output(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 计算训练进度
            training_process = (all_line_num * epoch + i) / (all_line_num * 5) * 100
            training_process = '%.2f' % training_process
            print('第{}轮，训练损失：{:.2f}，训练进度：{}%，（{}），名字：{}，预测国家：{}，正确？{}'\
                .format(epoch, np.mean(losses), float(training_process), time_since(start), line, guess, correct))
            records.append([np.mean(losses)])
 </pre>
 <pre>
 第0轮，训练损失：3.08，训练进度：0.0%，（0m 0s），名字：Mitchell，预测国家：French，正确？✗ (Scottish)
第0轮，训练损失：2.61，训练进度：2.99%，（0m 40s），名字：Sarto，预测国家：Scottish，正确？✗ (Italian)
第0轮，训练损失：2.41，训练进度：5.98%，（1m 18s），名字：Hot，预测国家：Chinese，正确？✗ (Russian)
第0轮，训练损失：2.29，训练进度：8.97%，（1m 56s），名字：Nishio，预测国家：Russian，正确？✗ (Japanese)
第0轮，训练损失：2.19，训练进度：11.96%，（2m 35s），名字：Han，预测国家：Vietnamese，正确？✓
第0轮，训练损失：2.12，训练进度：14.94%，（3m 13s），名字：Penzig，预测国家：French，正确？✗ (German)
第0轮，训练损失：2.05，训练进度：17.93%，（3m 51s），名字：Holeva，预测国家：Czech，正确？✗ (Russian)
第1轮，训练损失：2.01，训练进度：20.0%，（4m 18s），名字：Samson，预测国家：Scottish，正确？✗ (Dutch)
第1轮，训练损失：1.97，训练进度：22.99%，（4m 58s），名字：Degarmo，预测国家：Italian，正确？✗ (French)
第1轮，训练损失：1.93，训练进度：25.98%，（5m 34s），名字：Allegri，预测国家：Italian，正确？✓
第1轮，训练损失：1.90，训练进度：28.97%，（6m 11s），名字：Hung，预测国家：Korean，正确？✓
第1轮，训练损失：1.87，训练进度：31.96%，（6m 50s），名字：Liberzon，预测国家：Scottish，正确？✗ (Russian)
第1轮，训练损失：1.84，训练进度：34.94%，（7m 29s），名字：Huynh，预测国家：Korean，正确？✗ (Vietnamese)
第1轮，训练损失：1.82，训练进度：37.93%，（8m 8s），名字：Degarmo，预测国家：Portuguese，正确？✗ (French)
第2轮，训练损失：1.80，训练进度：40.0%，（8m 34s），名字：Ho，预测国家：Korean，正确？✓
第2轮，训练损失：1.78，训练进度：42.99%，（9m 14s），名字：Zdunowski，预测国家：Polish，正确？✓
第2轮，训练损失：1.76，训练进度：45.98%，（9m 51s），名字：Murchadh，预测国家：Irish，正确？✓
第2轮，训练损失：1.74，训练进度：48.97%，（10m 28s），名字：Koury，预测国家：Arabic，正确？✓
第2轮，训练损失：1.72，训练进度：51.96%，（11m 7s），名字：Starek，预测国家：Polish，正确？✓
第2轮，训练损失：1.71，训练进度：54.94%，（11m 46s），名字：Mitsuwa，预测国家：Japanese，正确？✓
第2轮，训练损失：1.69，训练进度：57.93%，（12m 24s），名字：Langlois，预测国家：Greek，正确？✗ (French)
第3轮，训练损失：1.68，训练进度：60.0%，（12m 50s），名字：Stewart，预测国家：German，正确？✗ (Scottish)
第3轮，训练损失：1.67，训练进度：62.99%，（13m 27s），名字：Takabe，预测国家：Arabic，正确？✗ (Japanese)
第3轮，训练损失：1.66，训练进度：65.98%，（14m 5s），名字：Simmon，预测国家：Irish，正确？✗ (German)
第3轮，训练损失：1.64，训练进度：68.97%，（14m 43s），名字：Rodham，预测国家：Irish，正确？✗ (English)
第3轮，训练损失：1.63，训练进度：71.96%，（15m 20s），名字：Sleiman，预测国家：Italian，正确？✗ (Arabic)
第3轮，训练损失：1.62，训练进度：74.94%，（15m 59s），名字：Wan，预测国家：Chinese，正确？✓
第3轮，训练损失：1.61，训练进度：77.93%，（16m 37s），名字：Duarte，预测国家：French，正确？✗ (Portuguese)
第4轮，训练损失：1.60，训练进度：80.0%，（17m 4s），名字：Giese，预测国家：Arabic，正确？✗ (German)
第4轮，训练损失：1.59，训练进度：82.99%，（17m 44s），名字：Swann，预测国家：Irish，正确？✗ (English)
第4轮，训练损失：1.58，训练进度：85.98%，（18m 24s），名字：Yoo，预测国家：Korean，正确？✓
第4轮，训练损失：1.57，训练进度：88.97%，（19m 5s），名字：Panoulias，预测国家：Greek，正确？✓
第4轮，训练损失：1.56，训练进度：91.96%，（19m 42s），名字：Watt，预测国家：Scottish，正确？✓
第4轮，训练损失：1.55，训练进度：94.94%，（20m 20s），名字：Freitas，预测国家：Greek，正确？✗ (Portuguese)
第4轮，训练损失：1.54，训练进度：97.93%，（20m 56s），名字：Riagain，预测国家：Irish，正确？✓
</pre>
<pre>
a = [i[0] for i in records]
plt.plot(a, label = 'Train Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
</pre>
![](http://upload-images.jianshu.io/upload_images/7539367-2e10688bc693347a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 五.通过姓氏分析语言的相近性
<pre>
import matplotlib.pyplot as plt


# 建立一个（18 x 18）的方阵张量
# 用于保存神经网络做出的预测结果
confusion = torch.zeros(n_categories, n_categories)
# 用于评估的模型的测试次数
n_confusion = 10000


# 评估用方法 传进去一个名字，给出预测结果
# 可以观察到这个方法的实现与 train 方法前半部分类似
# 其实它就是去掉反向传播的 train 方法
def evaluate(line_list):
    # 调用模型前应该先初始化模型的隐含层
    hidden = lstm.initHidden()
    # 别忘了将输入的list转化为torch.Variable
    line_variable = Variable(torch.LongTensor(line_list))
    # 调用模型
    output = lstm(line_variable, hidden)
    
    return output

# 循环一万次
for i in range(n_confusion):
    # 随机选择测试数据，包括姓氏以及所属语言
    category, line, category_index, line_list = random_training_pair()
    # 取得预测结果
    output = evaluate(line_list)
    
    # 取得预测结果的语言和索引
    guess, guess_i = category_from_output(output)
    
    # 以姓氏实际的所属语言为行
    # 以模型预测的所属语言为列
    # 在方阵的特定位置增加1

    confusion[category_index][guess_i] += 1

# 数据归一化
for i in range(n_categories):
    confusion[i] = confusion[i] / confusion[i].sum()

# 设置一个图表
fig = plt.figure()
ax = fig.add_subplot(111)
# 将 confusion 方阵数据传入
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# 设置图表两边的语言类别名称
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.show()
</pre>
![](http://upload-images.jianshu.io/upload_images/7539367-ec88058aaf9aff85.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# 六.封装模型
<pre>
# predict函数
# 第一个参数为要进行预测的姓氏
# 第二个参数为预测最大可能所属语言的数量
def predict(input_line, n_predictions=3):
    # 首先将用户输入的名字打印出来
    print('\n> %s' % input_line)
    # 将用户输入的字符串转化为索引列表
    line_list=[]
    for item in input_line:
        list=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V',
              'W','X','Y','Z']
        line_list.append(list.index(item))
    # 将用户输入的名字传入模型中进行预测
    output = evaluate(line_list)

    # 获得概率最大的n_predictions个语言类别
    topv, topi = output.data.topk(n_predictions, 1, True)
    # topv中保存着概率值
    # topi中保存着位置索引

    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        # 将预测概率最大的三种语言类别格式化后打印出来
        print('(%.2f) %s' % (value, all_categories[category_index]))
        # 将它们存储到 predictions 中
        predictions.append([value, all_categories[category_index]])
</pre>
<pre>


> Xi
(-0.42) Chinese
(-1.41) Japanese
(-3.62) Korean
</pre>
<pre>


> Moon
(-0.97) Chinese
(-1.30) Korean
(-2.41) Scottish
</pre>

<pre>
> Park
(-1.57) German
(-1.78) Scottish
(-1.99) Czech
</pre>
<pre>
> Kim
(-0.80) Chinese
(-0.81) Korean
(-2.70) Vietnamese
</pre>
<pre>
> Bryant
(-1.26) Irish
(-1.57) German
(-1.77) English
</pre>
<pre>
> Xu
(-0.78) Chinese
(-1.09) Japanese
(-2.53) Greek
</pre>
