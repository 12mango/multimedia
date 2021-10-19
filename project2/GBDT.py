# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 15:53:41 2021

@author: 14981
"""

"""
def GBDT(data,Class):  # LightGBM
    # X_train,X_test,y_train,y_test =train_test_split(X,Y,test_size=0.2)

    # lgb_train = lgb.Dataset(X_train, y_train) # 将数据保存到LightGBM二进制文件将使加载更快
    # lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)  # 创建验证数据

    # 将参数写成字典下形式

    Classes=max(Class)+1

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',  # 设置提升类型
        'objective': 'multiclass',  # 目标函数
        'metrics': {'multi_error'},  # 评估函数
        #    	'num_leaves': 31,   # 叶子节点数
        #    	'learning_rate': 0.05,  # 学习速率
        #    	'feature_fraction': 0.9, # 建树的特征选择比例
        #    	'bagging_fraction': 0.8, # 建树的样本采样比例
        #    	'bagging_freq': 5,  # k 意味着每 k 次迭代执行bagging
        'verbose': -1,  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        'num_class': Classes
    }
    
    X = data
    
    '''
    # one-hot
    Y = np.zeros((len(data),Classes))
    for i,per_class in enumerate(Class):
        Y[i][per_class]=1
    '''
    Y=Class

    # 切分训练集、测试集,切分比例7.5 : 2.5
    train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size=0.25)
    
    '''
    # 空值处理，默认方法：使用特征列的平均值进行填充
    my_imputer = Imputer()
    train_X = my_imputer.fit_transform(train_X)
    test_X = my_imputer.transform(test_X)
    '''

    # 转换为Dataset数据格式
    lgb_train = lgb.Dataset(train_X, train_y)
    lgb_eval = lgb.Dataset(test_X, test_y, reference=lgb_train)

    # 拟合、预测
    my_model = lgb.train(params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, early_stopping_rounds=5)
    predictions = my_model.predict(test_X, num_iteration=my_model.best_iteration)
    
    print(predictions)
    print(predictions.shape)
    print(test_y.shape)
    test_Y = np.zeros((len(test_y),Classes))
    for i,per_class in enumerate(test_y):
        test_Y[i][per_class]=1
    print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_Y)))

    '''
    data = lgb.Dataset(X, Y)
    result=lgb.cv(params, data, nfold=5).get("multi_error-mean")[-1]
    print(result)
    '''
"""