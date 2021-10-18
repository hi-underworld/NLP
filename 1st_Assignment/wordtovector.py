import jieba
from gensim.models import word2vec
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rc("font",family='PingFang HK')

SEGMENT = False
TRAIN = False

#segment
if SEGMENT:
    with open('实验一数据集.txt',encoding='utf-8') as f:
        document = f.read()
        document_cut = jieba.cut(document)
        result = ' '.join(document_cut)
        with open('实验一数据集分词.txt', 'w',encoding="utf-8") as f2:
            f2.write(result)
#train
if TRAIN:
    sentences = word2vec.LineSentence('实验一数据集分词.txt') 
    model = word2vec.Word2Vec(sentences, window=5, min_count=1, workers=4)
    model.save('./w2v.model')

#load_model    
else:
    model = word2vec.Word2Vec.load('./w2v.model')

#test
model.wv.similarity('中国','中华')
model.wv.most_similar(positive=['武汉'], topn=5)
model.wv.most_similar(positive=['湖北','成都'], negative = ['武汉'], topn = 5)

province_list = ['江苏', '南京', '成都', '四川', '湖北', '武汉', '河南', '郑州', '甘肃', '兰州', '湖南', '长沙', '陕西', '西安', '吉林', '长春', '广东', '广州', '浙江', '杭州']
province_vec = model.wv[province_list]

pca = PCA(n_components=2)
results = pca.fit_transform(province_vec)

plt.scatter(x=results[:, 0], y=results[:, 1])
for i in range(len(province_list)):
    plt.annotate(province_list[i], xy=(results[i, 0], results[i, 1]),
        xytext=(results[i, 0]+0.1, results[i, 1]+0.1))
plt.show()

