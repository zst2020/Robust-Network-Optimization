import pandas as pd
import numpy as np
import model_set
import matplotlib.pyplot as plt
from plot import plt_cdf,plt_pdf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import random
def predict_se(model, test_x, test_y, se_max, plot_on=True):
    print('Predicting SE...')
    pred_y = model.predict(test_x).flatten()
    test_y = test_y.flatten()
    # 计算绝对误差平均值，方差，以及百分比误差平均值，方差
    error = abs(pred_y - test_y) * se_max
    percen_err = abs(pred_y - test_y) / abs(test_y)
    print('error mean:{:.2f},std:{:.2f}'.format(np.mean(error), np.std(error)))
    print('percentage error mean:{:.3f},std:{:.3f}'.format(np.mean(percen_err), np.std(percen_err)))
    # 计算分段百分比误差
    percen_err_y = np.hstack((test_y[:, np.newaxis] * se_max, percen_err[:, np.newaxis]))
    sort_idxs = np.argsort(percen_err_y[:, 0])
    percen_err_y = percen_err_y[sort_idxs, :]
    mape_sections = np.array([1500, 2500, np.iinfo(int).max])
    mape_sections_mean = np.empty(len(mape_sections))
    section_idx = 0
    se_section_idx = 0
    for i in range(len(percen_err_y)):
        if percen_err_y[i, 0] > mape_sections[section_idx] or i == len(percen_err_y) - 1:
            mape_sections_mean[section_idx] = np.mean(percen_err_y[se_section_idx:i, 1])
            section_idx = section_idx + 1
            se_section_idx = i
    print('MAPE by sections:', np.around(mape_sections_mean, decimals=3))
    # choose how many bins you want here
    num_bins = 30
    # use the histogram function to bin the data
    counts, bin_edges = np.histogram(error, bins=num_bins)
    # now find the cdf
    cdf = np.cumsum(counts)
    # and finally plot the cdf
    if plot_on:
        plt.figure()
        # 随机抽取60个索引
        num_samples = 60
        x_range = np.arange(num_samples)
        sample_indices = random.sample(range(len(test_y)), num_samples)
        plt.plot(x_range, test_y[sample_indices] * se_max, label='Truth')
        plt.plot(x_range, pred_y[sample_indices] * se_max, label='Prediction')
        plt.xlabel('Sample ID ')
        plt.ylabel('Spectral Efficiency (bit/Number)')
        plt.legend()
        plt.show()
        axlim = test_y.mean() * se_max * 2

        plt.figure()
        plt.plot(pred_y * se_max, test_y * se_max, 'b*')
        plt.plot([0, axlim], [0, axlim])
        plt.axis([0, axlim, 0, axlim])
        plt.xlabel('Prediction (bit/Number)')
        plt.ylabel('Ground Truth (bit/Number)')
        plt.title('Spectral Efficiency')
        plt.show()
        
        plt.figure()
        plt.plot(bin_edges[1:], cdf)
        plt.xlabel('Absolute Error (bit/RB Number)')
        plt.ylabel('Count')
        plt.title('Error CDF')
        plt.show()

        plt_cdf(percen_err_y[:, 1], 'MAPE', '%')
    return np.mean(percen_err)

def train_se(n_epoch,train_x, train_y, val_x, val_y,predict_mode,patience = 15, sample_weight=None):
    print('Trianing ...')
    # 使用tensorboard保存结果
    # logdir = "logs/mlp/" + time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    # file_writer.set_as_default()
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(
    #     log_dir=logdir, update_freq='epoch', profile_batch=0)W
    
    # 定义神经网络模型
    
    model = predict_mode(dim=train_x.shape[1])

    # 控制学习率变化
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=10, min_lr=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto', restore_best_weights=True)

    # validation_split=0.2
    # 训练模型
    training_history = model.fit(x=train_x, y=train_y, validation_split=0.2,
                                 epochs=n_epoch, batch_size=64, sample_weight=sample_weight,
                                 callbacks=[reduce_lr, early_stop])
    return model

def predict_rank(model,x,y_ture):
    print('Predicting rank ...')
    y_pred = model.predict(x)
    y_pred = y_pred.flatten()
    mse = np.sum(np.abs(y_ture-y_pred))/len(y_ture)
    mape = np.mean(np.abs((y_ture-y_pred)/y_ture))
    print('mse: {}, mape: {}'.format(mse, mape))
    return y_pred

def train_predict_rank(n_epoch,train_x,train_y,val_x,val_y,test_x,test_y,predict_mode,random_seed = 0):
    print('Training rank ...')
    model = predict_mode(dim=train_x.shape[1])
    if random_seed != 0:
        np.random.seed(random_seed)
        np.random.shuffle(train_x)
        np.random.seed(random_seed)
        np.random.shuffle(train_y)

    # 控制学习率变化
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=10, verbose=1, mode='auto', min_delta=0.0001,
                                  cooldown=10, min_lr=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto', restore_best_weights=True)

    # validation_split=0.2
    # 训练模型
    training_history = model.fit(x=train_x, y=train_y, validation_data=(val_x,val_y),
                                 epochs=n_epoch, batch_size=64,
                                 callbacks=[reduce_lr, early_stop])
    train_y_pred = predict_rank(model,train_x,train_y)
    val_y_pred = predict_rank(model,val_x,val_y)
    test_y_pred = predict_rank(model,test_x,test_y)
    
    return train_y_pred,val_y_pred,test_y_pred

if __name__ == '__main__':


    pass