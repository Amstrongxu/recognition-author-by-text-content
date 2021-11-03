# _*_ coding:utf-8 _*_
# @Time : 2021/11/2 20:50
# @Author : xupeng
# @File : dataset_preprocess.py
# @software : PyCharm

"""
本教程为你提供了一个如何使用 tf.data.TextLineDataset 来加载文本文件的示例。
TextLineDataset 通常被用来以文本文件构建数据集（原文件中的一行为一个样本) 。
这适用于大多数的基于行的文本数据（例如，诗歌或错误日志) 。
下面我们将使用相同作品（荷马的伊利亚特）三个不同版本的英文翻译，然后训练一个模型来通过单行文本确定译者。
三个版本的翻译分别来自于:

William Cowper — text

Edward, Earl of Derby — text

Samuel Butler — text
"""


import tensorflow as tf

import tensorflow_datasets as tfds
import os

DIRECTORY_URL = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
FILE_NAMES = ['cowper.txt', 'derby.txt', 'butler.txt']

BUFFER_SIZE = 50000
BATCH_SIZE = 64
TAKE_SIZE = 5000

#下载文本文件到本地
def get_data():
    for name in FILE_NAMES:
        #get_file()该函数的作用主要是直接从URL下载资源
        text_dir = tf.keras.utils.get_file(name, origin=DIRECTORY_URL + name)
    parent_dir = os.path.dirname(text_dir)
    print(parent_dir)
    return parent_dir

parent_dir= r'C:\Users\28954\.keras\datasets'

# 将文本加载到数据集中
def labeler(example, index):
  return example, tf.cast(index, tf.int64)

def load_data_to_dataset(parent_dir):
    labeled_data_sets = []
    for i, file_name in enumerate(FILE_NAMES):
        # tf.data.TextLineDataset
        # 接口提供了一种方法从数据文件中读取。我们提供只需要提供文件名（1
        # 个或者多个）。这个接口会自动构造一个dataset，类中保存的元素：文中一行，就是一个元素，是string类型的tensor。
        lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))

        labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
        labeled_data_sets.append(labeled_dataset)

    return labeled_data_sets

labeled_data_sets = load_data_to_dataset(parent_dir)



# 将这些标记的数据集合并到一个数据集中，然后对其进行随机化操作。
def combine_to_one_dataset(labeled_data_sets):
    all_labeled_data = labeled_data_sets[0]
    for labeled_dataset in labeled_data_sets[1:]:
        all_labeled_data = all_labeled_data.concatenate(labeled_dataset)
    all_labeled_data = all_labeled_data.shuffle(
        BUFFER_SIZE, reshuffle_each_iteration=False)
    return all_labeled_data

all_labeled_data = combine_to_one_dataset(labeled_data_sets)

# 将文本编码成数字,构建文本与整数的一一映射
# 建立词汇表
def get_vocab(all_labeled_data):
    tokenizer = tfds.deprecated.text.Tokenizer()
    vocabulary_set = set()
    for text_tensor, _ in all_labeled_data:
        some_tokens = tokenizer.tokenize(text_tensor.numpy())
        vocabulary_set.update(some_tokens)
    vocab_size = len(vocabulary_set)
    return vocabulary_set, vocab_size

vocabulary_set, vocab_size = get_vocab(all_labeled_data)

# 样本编码
# 通过传递 vocabulary_set 到 tfds.features.text.TokenTextEncoder
# 来构建一个编码器。编码器的 encode 方法传入一行文本，返回一个整数列表。
def sample_encoder(vocabulary_set):
    encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)
    return encoder

encoder = sample_encoder(vocabulary_set)

#对数据集进行编码
# 在数据集上运行编码器（通过将编码器打包到 tf.py_function
# 并且传参至数据集的 map 方法的方式来运行）。
def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label

def encode_map_fn(text, label):
    # py_func doesn't set the shape of the returned tensors.
    encoded_text, label = tf.py_function(encode,
                                         inp=[text, label],
                                         Tout=(tf.int64, tf.int64))
    # `tf.data.Datasets` work best if all components have a shape set
    #  so set the shapes manually:
    encoded_text.set_shape([None])
    label.set_shape([])
    return encoded_text, label

def encode_all_labeled_data(all_labeled_data):
    all_encoded_data = all_labeled_data.map(encode_map_fn)
    return all_encoded_data

all_encoded_data = encode_all_labeled_data(all_labeled_data)

# 将数据集分割为测试集和训练集且进行分支
# 使用 tf.data.Dataset.take 和 tf.data.Dataset.skip
# 来建立一个小一些的测试数据集和稍大一些的训练数据集
# 在数据集被传入模型之前，数据集需要被分批。最典型的是，每个分支中的样本大小与格式需要一致。
# 但是数据集中样本并不全是相同大小的（每行文本字数并不相同）。
# 因此，使用 tf.data.Dataset.padded_batch（而不是 batch ）将样本填充到相同的大小。
def gen_train_test_dataset(all_encoded_data):
    train_data = all_encoded_data.skip(TAKE_SIZE).shuffle(BUFFER_SIZE)
    train_data = train_data.padded_batch(BATCH_SIZE)

    test_data = all_encoded_data.take(TAKE_SIZE)
    test_data = test_data.padded_batch(BATCH_SIZE)
    # 现在，test_data 和 train_data 不是（ example, label ）对的集合，
    # 而是批次的集合。每个批次都是一对（多样本, 多标签 ），表示为数组。
    return train_data, test_data

train_data, test_data = gen_train_test_dataset(all_encoded_data)



# if __name__ == "__main__":
#     # get_data() #下载了3个文本文件到     C:\Users\28954\.keras\datasets
#     parent_dir= r'C:\Users\28954\.keras\datasets'
#     labeled_data_sets = load_data_to_dataset(parent_dir)
#     print(len(labeled_data_sets))
#     print(labeled_data_sets[0])
#     # for index,(str, label) in enumerate(labeled_data_sets[0]):
#     #     print(str)
#     #     print(label)
#     #     break
#     all_labeled_data = combine_to_one_dataset(labeled_data_sets)
#     # for ex in all_labeled_data.take(5):
#     #     print(ex)
#     vocabulary_set, vocab_size = get_vocab(all_labeled_data)
#     print(vocab_size)
#     encoder = sample_encoder(vocabulary_set)
#     # #测试编码器
#     # example_text = next(iter(all_labeled_data))[0].numpy()
#     # print(example_text)
#     # encoded_example = encoder.encode(example_text)
#     # print(encoded_example)
#     all_encoded_data = encode_all_labeled_data(all_labeled_data)
#     # for ex in all_encoded_data.take(5):
#     #     print(ex)
#     train_data, test_data = gen_train_test_dataset(all_encoded_data)
#     sample_text, sample_labels = next(iter(test_data))
#
#     print(sample_text[0])
#     print(sample_labels[0])




