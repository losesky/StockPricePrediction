import streamlit as st
import akshare as ak
import numpy as np
# 导入 Keras 库和包
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler


def main():
    st.set_page_config(page_title="Stock Price Prediction using LSTM", page_icon=":shark:", layout="centered")
    st.title("Stock Price Prediction using LSTM")
    # 选择证券市场
    mac = st.radio("Select Market", ('上证', '深证', '北证'))
    if mac == '上证':
        mk = "sh"
    elif mac == '深证':
        mk = "sz"
    elif mac == '北证':
        mk = "bj"

    # 价格预测是基于过去多少天的价格
    TimeSteps = st.slider("基于过去多少天的价格来预测:", min_value=1, max_value=15, value=5)

    # 600000
    ticker = st.text_input("Enter stock ticker symbol:")
    if ticker and mk and TimeSteps:
        try:
            StockData = ak.stock_individual_fund_flow(stock=ticker, market=mk)

            st.dataframe(StockData)
            # st.write(StockData.head(10))
            # st.write(StockData.values)
            # 创建一个日期列
            # StockData['TradeDate'] = StockData.index
            # st.write(StockData['TradeDate'])
            # 提取每天的收盘价
            FullData = StockData[['收盘价']].values
            # st.write('### 标准化前的数据 ###')
            # st.write(FullData[0:10])

            # 用于神经网络快速训练的特征缩放
            # 在标准化或规范化之间进行选择
            sc = StandardScaler()
            # sc = MinMaxScaler()

            DataScaler = sc.fit(FullData)
            X = DataScaler.transform(FullData)
            # st.write('### 标准化后的数据 ###')
            # st.write(X[0:10])

            # split into samples
            X_samples = list()
            y_samples = list()

            NumerOfRows = len(X)

            # 遍历值以创建组合
            for i in range(TimeSteps, NumerOfRows, 1):
                x_sample = X[i - TimeSteps:i]
                y_sample = X[i]
                X_samples.append(x_sample)
                y_samples.append(y_sample)

            ################################################
            # 将输入重塑为 3D（样本数、时间步长、特征）
            X_data = np.array(X_samples)
            X_data = X_data.reshape(X_data.shape[0], X_data.shape[1], 1)
            # st.write('#### Input Data shape ####')
            # st.write(X_data.shape)

            # 我们不会将 y 重塑为 3D 数据，因为它应该只是一列
            y_data = np.array(y_samples)
            y_data = y_data.reshape(y_data.shape[0], 1)
            # st.write('\n#### Output Data shape ####')
            # st.write(y_data.shape)

            # 将数据拆分为训练和测试
            # 选择测试数据记录的数量
            TestingRecords = TimeSteps

            # 将数据拆分为训练和测试
            X_train = X_data[:-TestingRecords]
            X_test = X_data[-TestingRecords:]
            y_train = y_data[:-TestingRecords]
            y_test = y_data[-TestingRecords:]

            ############################################
            # 打印训练和测试的形状
            # st.write('\n#### Training Data shape ####')
            # st.write(X_train.shape)
            # st.write(y_train.shape)
            # st.write('\n#### Testing Data shape ####')
            # st.write(X_test.shape)
            # st.write(y_test.shape)
            # 可视化发送到 LSTM 模型的输入和输出
            # for inp, out in zip(X_train[0:2], y_train[0:2]):
            #    st.write(inp, '--', out)
            # 为 LSTM 定义输入形状
            TimeSteps = X_train.shape[1]
            TotalFeatures = X_train.shape[2]
            # st.write("Number of TimeSteps:", TimeSteps)
            # st.write("Number of Features:", TotalFeatures)
            # 初始化RNN
            regressor = Sequential()
            # 添加 first 隐藏层和 LSTM 层
            # return_sequences = True，表示每个时间步的输出要与隐藏的下一层共享
            regressor.add(
                LSTM(units=10, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))
            # 添加 Second 隐藏层和 LSTM 层
            regressor.add(
                LSTM(units=5, activation='relu', input_shape=(TimeSteps, TotalFeatures), return_sequences=True))
            # 添加 Third 隐藏层和 LSTM 层
            regressor.add(LSTM(units=5, activation='relu', return_sequences=False))
            # 添加输出层
            regressor.add(Dense(units=1))
            # 编译 RNN
            regressor.compile(optimizer='adam', loss='mean_squared_error')

            ##################################################

            import time
            # 测量模型训练所花费的时间
            StartTime = time.time()

            # 将 RNN 拟合到训练集
            regressor.fit(X_train, y_train, batch_size=5, epochs=100)

            ##################################################

            # 对测试数据进行预测
            predicted_Price = regressor.predict(X_test)
            predicted_Price = DataScaler.inverse_transform(predicted_Price)
            # 显示预测价格值
            st.success('Predicted_Price:' + str(predicted_Price[0][0]))
            # 获取测试数据的原始价格值
            orig = DataScaler.inverse_transform(y_test)
            dig = (100 - (100 * (abs(orig - predicted_Price) / orig)).mean()) / 100
            accuracy = "{:.2%}".format(dig)
            # 预测的准确性
            if dig > 0.995:
                st.write('Congratulations accuracy:', accuracy)
            else:
                st.write('Bad accuracy:', accuracy)

            # 计算时间
            EndTime = time.time()
            st.write("Total Time Taken: ", round(EndTime - StartTime), 'Second')

        except TypeError or ValueError:
            st.error("Invalid input.")


if __name__ == "__main__":
    main()
