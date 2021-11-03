# _*_ coding:utf-8 _*_
# @Time : 2021/11/3 10:40
# @Author : xupeng
# @File : build_model.py
# @software : PyCharm

import tensorflow as tf
import dataset_preprocess

vocab_size = dataset_preprocess.vocab_size + 1  #padding 后 词汇表加一

# 建立模型
def build_model():
    model = tf.keras.Sequential()
    # 第一层将整数表示转换为密集矢量嵌入
    model.add(tf.keras.layers.Embedding(vocab_size, 64))
    # 下一层是LSTM层，它允许模型利用上下文中理解单词含义
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)))
    #Dense层
    for units in [64, 64]:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    # 输出层。第一个参数是标签个数。
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    return model

model = build_model()
# 最后，编译这个模型。对于一个 softmax 分类模型来说，
# 通常使用 sparse_categorical_crossentropy 作为其损失函数。
# 你可以尝试其他的优化器，但是 adam 是最常用的。
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 训练模型
model.fit(dataset_preprocess.train_data,
          epochs=3,
          validation_data=dataset_preprocess.test_data
          )
eval_loss, eval_acc = model.evaluate(dataset_preprocess.test_data)

print('\nEval loss: {}, Eval accuracy: {}'.format(eval_loss, eval_acc))
