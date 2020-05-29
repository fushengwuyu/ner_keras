NER任务的几种实现

#### 1. 数据
* data/ : 人民日报实体抽取数据  

#### 2. 实现
* bert+crf  
    序列标注方法, test F1=0.94      
    ner_basic.py  
* bert+mrc+sigmoid  
  阅读理解方式，构造query，指针标出实体在text中的索引。test F1=0.92, 未细调，待优化。      
  ner_mrc_sigmoid.py  
  
* bert+mrc+seq2seq  
  阅读理解方式，构造query，直接生成实体。  
  ner_mrc_seq2seq.py  
  
  