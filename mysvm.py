import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.function_base import average
from scipy.io import loadmat 
import math
import sys
batchsz = 146

class Logger(object):   #终端输出保存到log.txt
    def __init__(self,fileN ="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN,"a")
 
    def write(self,message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 
sys.stdout = Logger("./log.txt") 

def obtain_w_via_gradient_descent(x, c, y, penalty_c, threshold = 1e-19, learn_rate = 1e-3):
    """ 利用梯度下降法求解如下的SVM问题：min 1/2 * w^T * w + C * Σ_i=1:n（max(0, 1 - y_i * (w^T * x_i + b))）
    :param x: 训练样本 x = [x_1, x_2, ..., x_i]
    :param c: 类别数
    :param y: 样本标签 y = [y_1, y_2, ..., y_c]
    :param threshold: 梯度下降停止阈值
    """
    data_num = np.shape(x)[1]
    feature_dim = np.shape(x)[0]
    w = np.ones([feature_dim, c], dtype=np.float32)
    b = np.ones([c, 1], dtype=np.float32)
    dl_dw = np.zeros([feature_dim, c], dtype=np.float)
    dl_db = np.zeros([c, 1], dtype=np.float)
    it = 1
    th = 0.1
    while it < 50000 and th > threshold:
        a = np.tile(b, [1, data_num])
        ksi = (np.transpose(w) @ x + np.tile(b, [1, data_num])) * y
        index_martix = ksi < 1

        for class_num in range(c):
            index_vector = index_martix[class_num, :]

            if True in index_vector:
                x_c = x[:, index_vector]

                data_num_c = np.shape(x_c)[1]
                e = np.ones([data_num_c, 1], dtype=np.float)
                y_c = np.reshape(y[class_num, index_vector], [data_num_c, 1])
                w_c = np.reshape(w[:, class_num], [feature_dim, 1])
                b_c = b[class_num]

                dl_dw[:, class_num] = (w_c + 2 * penalty_c * (x_c @ np.transpose(x_c) @ w_c +
                                                              x_c @ e * b_c -
                                                              x_c @ y_c))[:, 0]
                dl_db[class_num, 0] = 2 * penalty_c * (b_c * data_num_c +
                                                       np.transpose(w_c) @ x_c @ e -
                                                       np.transpose(y_c) @ e)
            else:
                w_c = np.reshape(w[:, class_num], [feature_dim, 1])
                dl_dw[:, class_num] = w_c[:, 0]
                dl_db[class_num, 0] = 0

        w_ = w - learn_rate * (dl_dw / np.linalg.norm(dl_dw, ord=2))
        b_ = b - learn_rate * dl_db

        th = np.sum(np.square(w_ - w)) + np.sum(np.square(b_ - b))
        it = it + 1

        w = w_
        b = b_

        if it % 5000 == 0:
            y_predict = np.transpose(w) @ x + np.tile(b, [1, data_num])
            correct_prediction = np.equal(np.argmax(y_predict, 0), np.argmax(y, 0))
            accuracy = np.mean(correct_prediction.astype(np.float))
            print("epoch:", it, "acc:", accuracy)
    return w,b



def signal_type(y):
    it = {1:0,2:1,3:2,5:3,6:4,7:5,8:6,11:7,14:8,15:9}    # 统一label ，10种信号编号0-9
    len=np.shape(y)[0]
    y_type=np.zeros(len)
    for i in range(len):
        y_type[i]=it[y[i]]
    return y_type


def convert_to_one_hot(y, C):    #将label 转换为onhot向量
    return np.eye(C)[y.reshape(-1)]


def getFeature(signal):     #提取特征，RMS，每个通道提取一个特征
    signal_feature=np.zeros([147,64])
    for signal_index in range(147):
        signal_matrix = signal[0,signal_index]
        for channel_index in range(64):
            signal_feature[signal_index,channel_index]=math.sqrt(np.sum(signal_matrix[:,channel_index]**2))  #RMS
    return signal_feature

def predict(w,b,x,y):        #训练结束后得到w,b ，进行预测推理验证，计算模型准确性
    data_num = np.shape(x)[1]
    y_predict = np.transpose(w) @ x + np.tile(b, [1, data_num])
    #print(np.shape(y_predict))
    correct_prediction = np.equal(np.argmax(y_predict, 0), np.argmax(y, 0))
    accuracy = np.mean(correct_prediction.astype(np.float))
    #print("test data acc:", accuracy)
    return accuracy

def main():
    data=loadmat('./data.mat')                     #数据读取
    label=loadmat('./label.mat')
    signal_data=data['preprocessed_dyn']
    #print(np.shape(signal_data[0,0]))
    #print(label['label'])


    feature_matrix=getFeature(signal_data)  #提取特征
    #print(feature_matrix)
    #print(np.shape(feature_matrix)) # (147,64)
    x=feature_matrix    #x为输入训练数据矩阵
    x = np.transpose(x)      #(64*147)   

  
    y=np.array(label['label'])
    y=y[0]
    y = y.astype(np.int)
    y_type=signal_type(y)   #label 处理编号0-9
    #print(y_type) #(147)
    y_type = y_type.astype(np.int)
    y_onehot = convert_to_one_hot(y_type, 10)   #转换为onhot向量矩阵
    y_onehot[y_onehot == 0] = -1                               
    y_onehot = np.transpose(y_onehot)   #(10*147)         

    test_result=np.zeros(147)    #用于保存每次训练后测试集样本是否分类正确（正确acc:1.0,错误acc:0.0)

    for iteration in range(147):  #训练循环147次，iteration index 样本为测试集
        print()
        print('Test data index =',iteration)  
        train_data=np.delete(x,iteration,axis=1)  #X矩阵删除测试集数据   (64*146)
        train_y_onehot=np.delete(y_onehot,iteration,axis=1)  #y_onehot矩阵删除测试集label (10*146)

        fw,fb=(obtain_w_via_gradient_descent(train_data, 10, train_y_onehot, 0.5)) #final_w ,final_b
        
        test_accuary = predict(fw,fb,x[:,iteration:iteration+1],y_onehot[:,iteration:iteration+1])  #预测test数据，返回是否分类正确 1/0
        print(iteration,' test data acc:',test_accuary)
        
        test_result[iteration]=test_accuary
    
    average_accuary=np.mean(test_result)

    print('Test result accuary vector')
    print(test_result)
    print()
    print('Final training model evalutaion:')
    print('Avarage accuary in 147 trainings:',average_accuary)

if __name__ == '__main__':
    main()