import jieba
str_text = "分词就是将句子、段落、文章这种长文本，分解为以字词为单位的数据结构，方便后续的处理分析工作。词向量，顾名思义就是⽤来表⽰词的向量，也可被认为是词的特征向量或表征。把词映射为实数域向量的技术也叫词嵌入。我们通常在训练语言模型的同时得到词向量。"
seg_list = jieba.cut(str_text, cut_all=True)
print("全模式:","/".join(seg_list))
seg_list = jieba.cut(str_text, cut_all=False)
print("精准模式:","/".join(seg_list))
seg_list = jieba.cut_for_search(str_text)
print("搜索引擎模式:","/".join(seg_list))

# encoding=utf-8
import jieba

seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("Full Mode: " + "/ ".join(seg_list))  # 全模式
# 【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("Default Mode: " + "/ ".join(seg_list))  # 精确模式
# 【精确模式】: 我/ 来到/ 北京/ 清华大学

seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
print(", ".join(seg_list))
# 【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print(", ".join(seg_list))
# 【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造