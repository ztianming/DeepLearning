import tensorflow as tf
from sklearn.preprocessing import scale
import pandas as pd
import numpy as np

all_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 分割为训练集、验证集 比例：3：1
all_data = all_data.sample(frac=1.0)  # 全部打乱
cut_idx = int(round(0.25 * all_data.shape[0]))
# print(cut_idx)
validation_data, train_data = all_data.iloc[:cut_idx], all_data.iloc[cut_idx:]
# print(validation_data)
# print(train_data)

# print(train_data.head())
# print(train_data.ix[:, 1:-1])

# 最大最小归一化
train_x = scale(np.asarray(train_data.ix[:, 1:-1]))
train_y = np.asarray(train_data.ix[:, -1])
# print(train_x, train_y)

# onehot编码
train_y = np.asarray(pd.get_dummies(train_y))

validation_x = scale(np.asarray(validation_data.ix[:, 1:-1]))
validation_y = np.asarray(validation_data.ix[:, -1])
# print(validation_x, validation_y)
validation_y = np.asarray(pd.get_dummies(validation_y))

test_x = scale(np.asarray(test_data.ix[:, 1:-1]))

# 类别
output = train_y.shape[1]
# 特征数量
feature_num = 20

# 输入
X = tf.placeholder(tf.float32, shape=[None, feature_num])
# 输出
Y = tf.placeholder(tf.float32, [None, output])

# 定义神经网络
def neural_networks():
    # --------------------- Encoder -------------------- #
    e_w_1 = tf.Variable(tf.truncated_normal([feature_num, 256], stddev=0.1))
    e_b_1 = tf.Variable(tf.constant(0.0, shape=[256]))

    e_w_2 = tf.Variable(tf.truncated_normal([256, 128], stddev=0.1))
    e_b_2 = tf.Variable(tf.constant(0.0, shape=[128]))

    e_w_3 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.1))
    e_b_3 = tf.Variable(tf.constant(0.0, shape=[64]))

    # --------------------- Decoder  ------------------- #
    d_w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    d_b_1 = tf.Variable(tf.constant(0.0, shape=[128]))

    d_w_2 = tf.Variable(tf.truncated_normal([128, 256], stddev=0.1))
    d_b_2 = tf.Variable(tf.constant(0.0, shape=[256]))

    d_w_3 = tf.Variable(tf.truncated_normal([256, feature_num], stddev=0.1))
    d_b_3 = tf.Variable(tf.constant(0.0, shape=[feature_num]))

    # --------------------- DNN  ------------------- #
    w_1 = tf.Variable(tf.truncated_normal([64, 128], stddev=0.1))
    b_1 = tf.Variable(tf.constant(0.0, shape=[128]))

    w_2 = tf.Variable(tf.truncated_normal([128, 128], stddev=0.1))
    b_2 = tf.Variable(tf.constant(0.0, shape=[128]))

    w_3 = tf.Variable(tf.truncated_normal([128, output], stddev=0.1))
    b_3 = tf.Variable(tf.constant(0.0, shape=[output]))

    #########################################################
    layer_1 = tf.nn.tanh(tf.add(tf.matmul(X, e_w_1), e_b_1))
    layer_2 = tf.nn.tanh(tf.add(tf.matmul(layer_1, e_w_2), e_b_2))
    encoded = tf.nn.tanh(tf.add(tf.matmul(layer_2, e_w_3), e_b_3))

    layer_4 = tf.nn.tanh(tf.add(tf.matmul(encoded, d_w_1), d_b_1))
    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, d_w_2), d_b_2))
    decoded = tf.nn.tanh(tf.add(tf.matmul(layer_5, d_w_3), d_b_3))

    layer_7 = tf.nn.tanh(tf.add(tf.matmul(encoded, w_1), b_1))
    # layer_8 = tf.nn.sigmoid(tf.add(tf.matmul(layer_7, w_2), b_2))
    layer_8 = tf.nn.tanh(tf.add(tf.matmul(layer_7, w_2), b_2))
    out = tf.nn.softmax(tf.add(tf.matmul(layer_8, w_3), b_3))
    return (decoded, out)

# 训练神经网络
def train_neural_networks():
    decoded, predict_output = neural_networks()

    us_cost_function = tf.reduce_mean(tf.pow(X - decoded, 2))
    # 交叉熵
    s_cost_function = -tf.reduce_sum(Y * tf.log(predict_output))
    # 最速下降法
    # us_optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(us_cost_function)
    # s_optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(s_cost_function)
    us_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(us_cost_function)
    s_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(s_cost_function)

    correct_prediction = tf.equal(tf.argmax(predict_output, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 训练周期
    training_epochs = 40
    batch_size = 10
    total_batches = train_data.shape[0]

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # ------------ Training Autoencoders - Unsupervised Learning ----------- #
        # autoencoder是一种非监督学习算法，他利用反向传播算法，让目标值等于输入值
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                _, c = sess.run([us_optimizer, us_cost_function], feed_dict={X: batch_x})
                epoch_costs = np.append(epoch_costs, c)
            print("Epoch: ", epoch, " Loss: ", np.mean(epoch_costs))
        print("------------------------------------------------------------------")
        # log("correct_prediction:", sess.run(correct_prediction))
        # ---------------- Training NN - Supervised Learning ------------------ #
        for epoch in range(training_epochs):
            epoch_costs = np.empty(0)
            for b in range(total_batches):
                offset = (b * batch_size) % (train_x.shape[0] - batch_size)
                batch_x = train_x[offset:(offset + batch_size), :]
                batch_y = train_y[offset:(offset + batch_size), :]
                _, c = sess.run([s_optimizer, s_cost_function], feed_dict={X: batch_x, Y: batch_y})
                epoch_costs = np.append(epoch_costs, c)

            accuracy_in_train_set = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
            # validation data
            accuracy_in_test_set = sess.run(accuracy, feed_dict={X: validation_x, Y: validation_y})
            np.set_printoptions(threshold=np.inf)


            test_output = sess.run(predict_output, feed_dict={X: test_x})
            print("Epoch: ", epoch, " Loss: ", np.mean(epoch_costs),
                  " Accuracy: ", accuracy_in_train_set, ' ',
                  accuracy_in_test_set)
            if epoch == training_epochs-1:
                pre_y = sess.run(tf.argmax(test_output, 1))
                print(len(pre_y))
                print(pre_y)
            if accuracy_in_test_set > 0.92:
                print("save net")
                save_path = saver.save(sess, 'my_net/save_net.ckpt')
# 运行
train_neural_networks()



