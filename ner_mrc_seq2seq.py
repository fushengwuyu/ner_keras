# @Author:sunshine
# @Time  : 2020/5/12 下午3:46

"""
基础的序列标注的方式
"""

from bert4keras.tokenizers import load_vocab, Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding, AutoRegressiveDecoder
from keras.optimizers import Adam
import keras.backend as K
import keras
import json
from collections import defaultdict
import numpy as np

config_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/vocab.txt'

batch_size = 16
max_len = 256

model_path = 'best_ner_mrc.h5'

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split(' ')
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])
                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 加载数据
train_data = load_data('data/china-people-daily-ner-corpus/example.train')[:200]
valid_data = load_data('data/china-people-daily-ner-corpus/example.dev')[:100]
test_data = load_data('data/china-people-daily-ner-corpus/example.test')

# query 映射
query_mapping = json.load(open('data/mrc_query_mapping.json', 'r', encoding='utf-8'))

# 加载并精简词表,建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

tokenizer = Tokenizer(token_dict)


# 数据迭代器
class MyDataGenerator(DataGenerator):
    def __init__(self, data, batch_size=32, buffer_size=None):
        super(MyDataGenerator, self).__init__(data, batch_size, buffer_size)

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []

        for is_end, item in self.sample(random):
            """
            单条样本: [[cls], query, [sep], text, [sep], entity1#entity2#...[sep]]
            """

            for k, v in query_mapping.items():

                query_token_ids, query_segment_ids = tokenizer.encode(v)
                token_ids = query_token_ids.copy()
                entity_ids = []
                for w, l in item:
                    w_token_ids = tokenizer.encode(w)[0][1:-1]
                    if l == k:
                        entity_ids.extend(w_token_ids + [tokenizer.token_to_id('#')])
                    if len(token_ids) + len(w_token_ids) + len(entity_ids) < max_len:
                        token_ids += w_token_ids
                    else:
                        break

                token_ids += [tokenizer._token_end_id]
                segment_ids = [0] * len(token_ids) + [1] * (len(entity_ids) + 1)
                token_ids += entity_ids
                token_ids += [tokenizer._token_end_id]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)

                if len(batch_token_ids) == self.batch_size or is_end:
                    # padding
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    yield [batch_token_ids, batch_segment_ids], None
                    batch_token_ids, batch_segment_ids = [], []


# 构建模型
model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens
)

y_true = model.input[0][:, 1:]
y_mask = model.input[1][:, 1:]
y_pred = model.output[:, :-1]

cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)

model.add_loss(cross_entropy)
model.compile(optimizer=Adam(learning_rate=1e-5))
model.summary()


class Evaluator(keras.callbacks.Callback):
    """评估和保存模型"""

    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.lowest:
            self.lowest = logs['loss']
            model.save(model_path)


class AutoAnswer(AutoRegressiveDecoder):
    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)

        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        result = set()
        for k, v in query_mapping.items():
            token_ids, segment_ids = tokenizer.encode(first_text=v, second_text=text)
            output_ids = self.beam_search([token_ids, segment_ids], topk)
            entity_str = tokenizer.decode(output_ids)
            entity = entity_str.split('#') if entity_str else []
            for item in entity:
                result.add((item, k))
        return result


aotu_answer = AutoAnswer(start_id=None, end_id=tokenizer._token_end_id, maxlen=64)

if __name__ == '__main__':
    # 训练
    # train_D = MyDataGenerator(train_data, batch_size)
    # evalutor = Evaluator()
    #
    # model.fit_generator(train_D.forfit(), epochs=10, steps_per_epoch=len(train_D))
    # model.save_weights(model_path)

    # predict
    model.load_weights('best_ner_mrc.h5')
    text = '海钓地点在金门与厦门之间的海域。'
    result = aotu_answer.generate(text)
    # print(result)
