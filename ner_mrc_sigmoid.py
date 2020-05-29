# @Author:sunshine
# @Time  : 2020/5/27 上午9:51

from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import numpy as np
import json
import tensorflow as tf
from keras.layers import Input, Lambda
from keras.utils import to_categorical

maxlen = 256
epochs = 10
batch_size = 32
bert_layers = 12
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

# bert配置
config_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/vocab.txt'

model_path = 'models/mrc_sigmoid.weights'
# query 映射
query_mapping = json.load(open('data/mrc_query_mapping.json', 'r', encoding='utf-8'))


# def load_data(filename):
#     data = []
#     with codecs.open(filename, 'r', encoding='utf-8') as rd:
#         for line in rd:
#             text, label = json.loads(line.strip('\n'))
#             label_ont_hot = []
#             for k, v in query_mapping.items():
#                 for i in range(len(label)):
#                     if label[i] == 'O':
#                         label_ont_hot.append([1, 0])
#                     elif label[i] == 'B-' + v:
#                         label_ont_hot.append([0, 1])
#                     elif i + 1 < len(label):
#                         if label[i + 1] == 'I-' + v:
#                             label_ont_hot.append([1, 0])
#                         else:
#                             label_ont_hot.append([0, 1])
#                     else:
#                         label_ont_hot.append([0, 1])
#             data.append([text, label_ont_hot])
#
#     return data

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


# 标注数据
train_data = load_data('data/china-people-daily-ner-corpus/example.train')
valid_data = load_data('data/china-people-daily-ner-corpus/example.dev')
test_data = load_data('data/china-people-daily-ner-corpus/example.test')

# 建立分词器
tokenizer = Tokenizer(vocab_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_start, batch_end = [], [], [], []
        for is_end, item in self.sample(random):
            for k, v in query_mapping.items():

                query_token_ids, query_segment_ids = tokenizer.encode(v)
                token_ids = query_token_ids.copy()
                start = query_segment_ids.copy()
                end = query_segment_ids.copy()
                for w, l in item:
                    w_token_ids = tokenizer.encode(w)[0][1:-1]
                    if len(token_ids) + len(w_token_ids) < maxlen:
                        token_ids += w_token_ids
                        start_tmp = [0] * len(w_token_ids)
                        end_tmp = [0] * len(w_token_ids)
                        if l == k:
                            start_tmp[0] = end_tmp[-1] = 1
                        start += (start_tmp)
                        end += (end_tmp)
                    else:
                        break

                token_ids += [tokenizer._token_end_id]
                segment_ids = query_segment_ids + [1] * (len(token_ids) - len(query_token_ids))
                start += [0]
                end += [0]
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_start.append(to_categorical(start, 2))
                batch_end.append(to_categorical(end, 2))

                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_start = sequence_padding(batch_start)
                    batch_end = sequence_padding(batch_end)
                    yield [batch_token_ids, batch_segment_ids, batch_start, batch_end], None
                    batch_token_ids, batch_segment_ids, batch_start, batch_end = [], [], [], []


bert_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
)

mask = bert_model.input[1]

start_labels = Input(shape=(None, 2), name='Start-Labels')
end_labels = Input(shape=(None, 2), name='End-Labels')

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
x = bert_model.get_layer(output_layer).output

start_output = Dense(2, activation='sigmoid', name='start')(x)
end_output = Dense(2, activation='sigmoid', name='end')(x)
start_output = Lambda(lambda x: x ** 2)(start_output)
end_output = Lambda(lambda x: x ** 2)(end_output)

start_model = Model(bert_model.input, start_output)
end_model = Model(bert_model.input, end_output)

model = Model(bert_model.input + [start_labels, end_labels], [start_output, end_output])
model.summary()


def focal_loss(logits, labels, mask, lambda_param=1.5):
    probs = K.softmax(logits, axis=-1)
    pos_probs = probs[:, :, 1]
    prob_label_pos = tf.where(K.equal(labels, 1), pos_probs, K.ones_like(pos_probs))
    prob_label_neg = tf.where(K.equal(labels, 0), pos_probs, K.zeros_like(pos_probs))
    loss = K.pow(1. - prob_label_pos, lambda_param) * K.log(prob_label_pos + 1e-7) + \
           K.pow(prob_label_neg, lambda_param) * K.log(1. - prob_label_neg + 1e-7)
    loss = -loss * K.cast(mask, 'float32')
    loss = K.sum(loss, axis=-1, keepdims=True)
    loss = K.mean(loss)
    return loss


# start loss
start_loss = K.binary_crossentropy(start_labels, start_output)
start_loss = K.mean(start_loss, 2)
start_loss = K.sum(start_loss * mask) / K.sum(mask)
# end部分loss
end_loss = K.binary_crossentropy(end_labels, end_output)
end_loss = K.mean(end_loss, 2)
end_loss = K.sum(end_loss * mask) / K.sum(mask)

# start_loss = focal_loss(start_output, start_labels, mask)
# end_loss = focal_loss(end_output, end_labels, mask)

loss = start_loss + end_loss
model.add_loss(loss)
model.compile(optimizer=Adam(1e-5))


def extract(text):
    result = set()
    for k, v in query_mapping.items():
        text_tokens = tokenizer.tokenize(text)[1:]
        query_tokens = tokenizer.tokenize(v)
        while len(text_tokens) + len(query_tokens) > 512:
            text_tokens.pop(-2)

        token_ids = tokenizer.tokens_to_ids(query_tokens)
        token_ids += tokenizer.tokens_to_ids(text_tokens)
        segment_ids = [0] * len(query_tokens) + [1] * len(text_tokens)

        start_out = start_model.predict([[token_ids], [segment_ids]])[0][len(query_tokens):]
        end_out = end_model.predict([[token_ids], [segment_ids]])[0][len(query_tokens):]

        start = np.where(start_out > 0.6)
        end = np.where(end_out > 0.5)
        start = [i for i, j in zip(*start) if j == 1]
        end = [i for i, j in zip(*end) if j == 1]
        for s, e in zip(start, end):
            if e >= s:
                tmp_str = ''.join(text_tokens[s: e + 1])
                result.add((tmp_str, k))
    return result


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = ''.join([i[0] for i in d])
        R = extract(text)
        T = set([tuple(i) for i in d if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights(model_path)
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )


if __name__ == '__main__':

    # evaluator = Evaluate()
    # train_generator = data_generator(train_data, batch_size)
    #
    # model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

    model.load_weights(model_path)
    p,r,f1 = evaluate(test_data)
    print(p, r, f1)