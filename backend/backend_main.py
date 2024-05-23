from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
import random
import os
import pdb
import json
import pandas as pd
import torch.utils.data as data
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
from xgboost import XGBClassifier
import joblib
from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)  # 创建了一个新的Flask web应用实例


CORS(app)  # 启用跨域资源共享


# 定义目标目录
target_dir = '../save_weights'

# 检查目录是否存在，如果不存在则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    print(f"目录 '{target_dir}' 不存在，已创建")
elif not os.path.isdir(target_dir):
    raise NotADirectoryError(f"路径 '{target_dir}' 已存在，但不是一个目录")
else:
    print(f"目录 '{target_dir}' 已存在")

pretrained_torch_weights = 'RandomForestRegressor.pkl'
pretrained_sklearn_weights = 'XGBClassifier.json'

pretrained_sklearn_weights_path = os.path.join(target_dir, pretrained_sklearn_weights)
pretrained_torch_weights_path = os.path.join(target_dir, pretrained_torch_weights)

def process_train_data(kernel_log_data_path, failure_tag_data_path):
    # 将得到的数据按照时间进行取整
    agg_time = '5min'
    kernel_data = pd.read_csv(kernel_log_data_path)
    kernel_data['collect_time'] = pd.to_datetime(kernel_data['collect_time']).dt.ceil(agg_time)
    group_kernel_data = kernel_data.groupby(['serial_number','collect_time'],as_index=False).agg('sum')

    # 读取产生故障的数据的标签
    failure_data = pd.read_csv(failure_tag_data_path)
    failure_data['failure_time']= pd.to_datetime(failure_data['failure_time'])

    # merge the kernel and failure data
    merged_data = pd.merge(group_kernel_data,failure_data[['serial_number','failure_time']],how='left',on=['serial_number'])
    merged_data['failure_tag']=(merged_data['failure_time'].notnull()) & ((merged_data['failure_time']-merged_data['collect_time']).dt.seconds <= 5*60)
    # translate bool into 0/1
    merged_data['failure_tag']= merged_data['failure_tag'].astype(int)

    # delete duplicate value
    merged_data.duplicated().sum()

    # delete the null value 
    a = merged_data.drop(['failure_time'],axis=1)
    # a.isna().sum()   

    # compute the error time
    merged_data['error_time'] = pd.to_datetime(merged_data['failure_time']) - pd.to_datetime(merged_data['collect_time'])
    merged_data['error_time'] = merged_data['error_time'].dt.seconds/60


    # extract the feature
    feature_data = merged_data.drop(['serial_number', 'collect_time','manufacturer','vendor','failure_time'], axis=1)
    negetive_samples = feature_data[feature_data['failure_tag']==0]
    positive_samples = feature_data[feature_data['failure_tag']==1]
    negetive_samples.drop('error_time',axis=1,inplace=True)
    error_time = positive_samples['error_time']
    positive_samples.drop('error_time',axis=1,inplace=True)
    print("negetive_samples:",negetive_samples.shape)
    print("positive_samples:",positive_samples.shape)

    # balance positive and negative samples
    negetive_samples = negetive_samples.sample(frac=0.005)
    sample = negetive_samples.append(positive_samples)

    # create the train dataset
    X_train_class = torch.from_numpy(sample.iloc[:,:-1].values).type(torch.FloatTensor)
    y_train_class = torch.from_numpy(sample.iloc[:,-1].values).type(torch.LongTensor)

    # 创建回归的数据集
    X_train_regressor = positive_samples.iloc[:, :-1]
    y_train_regressor = error_time
    X_train_regressor = torch.from_numpy(X_train_regressor.values).type(torch.FloatTensor)
    y_train_regressor = torch.from_numpy(y_train_regressor.values).type(torch.FloatTensor)
    return X_train_class, y_train_class, X_train_regressor, y_train_regressor

def process_test_data(kernel_log_test_path):
    agg_time = '5min'
    kernel_test_data = pd.read_csv(kernel_log_test_path)
    kernel_test_data['collect_time'] = pd.to_datetime(kernel_test_data['collect_time']).dt.ceil(agg_time)
    group_kernel_test_data = kernel_test_data.groupby(['serial_number','collect_time'],as_index=False).agg('sum')
    test_ids = pd.DataFrame(group_kernel_test_data[['serial_number','collect_time']])
    test_features = group_kernel_test_data.drop(['serial_number', 'collect_time','manufacturer','vendor'], axis=1)
    X_test = torch.from_numpy(test_features.values).type(torch.FloatTensor)
    return X_test, test_ids

@app.route('/check_weights', methods=['GET'])
def check_weights():
    exists = os.path.isfile(pretrained_torch_weights_path)
    return jsonify({'exists': exists})


@app.route('/train', methods=['POST'])
def train():
    if 'train_data' not in request.files or 'label_data' not in request.files:
        return jsonify({'success': False, 'message': 'Training and label data files are required.'})

    train_data = request.files['train_data']
    label_data = request.files['label_data']

    X_train_cls, y_train_cls, X_train_Reg, y_train_Reg = process_train_data(train_data, label_data)

    # 先对结果进行分类
    clf = XGBClassifier()
    clf.fit(X_train_cls, y_train_cls)

    # 保存模型权重到文件
    clf.save_model(pretrained_sklearn_weights_path)

    # 再对结果进行回归
    regressor = RandomForestRegressor()
    regressor.fit(X_train_Reg, y_train_Reg)

    joblib.dump(regressor, pretrained_torch_weights_path)

    return jsonify({'success': True, 'message': 'Model trained and weights saved.'})


@app.route('/predict', methods=['POST'])
def predict():
    print("打印内容：", request.files)

    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'Test data file is required.'})
    
    file = request.files['file']
    X_test, test_ids = process_test_data(file)

    # 先进行分类
    clf = XGBClassifier()
    clf.load_model(pretrained_sklearn_weights_path)
    test_predict = clf.predict(X_test.cpu())
    print(test_predict)

    test_ids['predict'] = test_predict
    test_ids = test_ids[test_ids['predict'] == 1]
    test_ids = test_ids.drop('predict', axis=1)

    reg = RandomForestRegressor()
    reg = joblib.load(pretrained_torch_weights_path)

    pre_time = reg.predict(X_test.cpu()[test_predict == 1]).tolist()
    test_ids['pre_time'] = pre_time
    test_ids['pre_time'] = test_ids['pre_time'].astype(int)

    final_test = test_ids.drop_duplicates(subset=['serial_number'], keep='first', inplace=False)
    predictions_file = 'predictions.csv'
    final_test.to_csv(predictions_file, index=False)
    print("predict finished")
    return send_file(predictions_file, as_attachment=True)




if __name__ == "__main__":
    print('run 127.0.0.1:18000')
    app.run(host='127.0.0.1', port=18000)