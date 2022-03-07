import DataCenter as dc
from Bean import Parameter as pr
from tqdm import tqdm
import pandas as pd
from Factor.Stock.ParentStockFactor import ParentStockFactor
from sklearn.covariance import LedoitWolf
from Factor.Stock.Quote import Quote
from Bean.Tool import ToolFactor
from Factor.Stock.Scale import MarketValue
import numpy as np
import scipy
import warnings
import statsmodels.api as sm
from datetime import datetime
warnings.filterwarnings("ignore")
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse


warnings.filterwarnings("ignore")


class PCRiskMatrix(ParentStockFactor):
    '''股票收益率协方差矩阵'''
    def __init__(self):
        super().__init__()
        self.name = "风险矩阵"
        self.type = "Risk"

        # 模型的指数半衰期参数
        p1 = pr.Parameter("半衰期", 60)
        self.paraList[p1.name] = p1



    def init(self):

        q = Quote()
        self.addNeedFactor("行情", q)

    def loadData(self):
        '''
        直接把每一个调仓日的股票收益协方差矩阵存起来。为了节省存储空间，直接存协方差矩阵的所有特征值以及前K大特征值对应的特征向量。
        计算协方差矩阵时使用了Lediot-Wolf估计方法（参考 “协方差矩阵的估计和评价方法【天风金工因子选股系列之七】”）。
        之后又做了估计的调整，参考了barra模型中的volatility regime adjustment
        '''
        lag = self.paraList["半衰期"].value
        df = self.getNeedFactor("行情").getFactorDataSet()["DReturn"]
        # 收益率超过0.2的拉回至0.2
        df = df.map(lambda x: 0.2 if x > 0.2 else x)
        df = df.map(lambda x: -0.2 if x < -0.2 else x)
        LWShrinkCovDict = {}    # 协方差矩阵字典，key是日期
        Bt_list = []
        for tradingDate in tqdm(dc.DataCenter.dataSet.actionDates):
            idx = dc.DataCenter.dataSet.tradingDates.to_list().index(tradingDate)
            dateInterval = dc.DataCenter.dataSet.tradingDates.loc[idx - 120: idx - 1].values   # 获取过去120个交易日的日期
            df2 = df.loc[dateInterval]      # 获取过去120个交易日的收益率series
            df2 = pd.pivot_table(df2.reset_index(), index="tradingDate", columns="code", values="DReturn")   # 转换成矩阵形式，方便计算协方差矩阵
            df2 = df2.T.fillna(df2.T.median().to_dict(), axis=0).T    # 用截面中位数填充缺失值

            # 使用Lediot线性压缩估计量来计算原始协方差矩阵，而不是直接用dataframe.cov()函数。这是因为变量数大于样本数，直接计算协方差矩阵会导致（1）计算出来的矩阵不可逆，可能无法规划求解出结果 （2）估计不准确。 Lediot线性压缩估计量来估计协方差矩阵能够有效解决这一问题
            LWShrinkCov = pd.DataFrame(LedoitWolf().fit(df2).covariance_, index=df2.columns, columns=df2.columns)

            # region 估计偏差的调整。思路参考了barra模型中的volatility regime adjustment。（效果感觉不太明显）
            # 如果预测的很准确的话，那么每只股票的真实收益/预测的波动率，这个东西的方差应该等于1。Bt就是其方差。 如果Bt>1，说明当期的波动率预测偏小了，风险预测偏小。反之亦然。 于是可以用过去的Bt来动态调整当期的波动率预测。具体做法是，定义一个指数半衰的权重，对过去所有的Bt计算加权平方和再开根号，记lambdaF，LambdaF>1，说明过去的风险预测都偏小了，需要放大一下当期的风险。
            ft = df.loc[tradingDate].reindex(LWShrinkCov.index).fillna(df.loc[tradingDate].median())    # 当期的真实收益。这里虽然计算了未来信息，但是并没有用到计算之中。
            Bt = (np.square(ft) / np.diag(LWShrinkCov)).mean()    # 计算Bt

            if Bt_list.__len__() == 0:
                pass
            else:
                ewt = np.exp(np.log(.5) / (lag / 30)) ** np.arange(Bt_list.__len__(), 0, -1)
                ewt = ewt / ewt.sum()    # 指数半衰权重
                lambdaF = np.sqrt((np.square(Bt_list) * ewt).sum())    #
                LWShrinkCov = LWShrinkCov * lambdaF    # 这一步是调整过程。

            Bt_list.append(Bt)
            # endregion

            # 为了减小数据文件大小，存储主要特征值，特征向量，对应的股票列表
            S = self.Spectral_Decomposition(LWShrinkCov)

            LWShrinkCovDict[tradingDate] =  S  # 保存前k个特征值，特征向量，以及特质收益波动率


        self.addFactorDataSet(LWShrinkCovDict)


    def getCovMatrix(self, tradingDate, stockList):


        u, lambda1, D, totalStock = self.getFactorDataSet()[tradingDate] # u：前K大特征值对应特征向量，lambda1：前K大特征值。
        # D：股票特质收益波动率
        u = pd.DataFrame(u, index=totalStock).reindex(stockList)
        D = pd.DataFrame(D, index=totalStock).reindex(stockList)
        u = u.fillna(u.median().to_dict())    # 对于没有波动率估计的的股票，用中位数填充
        D = D.fillna(D.median().to_dict())

        PCEcov = u.dot(np.diag(lambda1)).dot(u.T)          # 用前K大特征值计算近似的协方差矩阵，这样做是为了加快规划求解的速度

        return PCEcov.values, D[0].values  # 返回近似协方差矩阵，以及特质收益的方差向量


    def Spectral_Decomposition(self, LWShrinkCov):
        '''
        :param  LWShrinkCov: 协方差矩阵
        :return:    [0]: 前k大特征值对应特征向量
                    [1]: 前k大特征值
                    [2]: 其余特征值
                    [3]: 股票列表
        '''
        lambda_, u = np.linalg.eigh(LWShrinkCov)    # 特征值分解
        indices = (-lambda_).argsort()
        lambda_ = lambda_[indices]    # 对特征值从大到小排序
        u = u[:, indices]    # 排序对应的特征向量

        K = min(((lambda_.cumsum() / lambda_.sum()) < 0.8).sum() + 1, 100)  # 保留K个特征值，这K个特征值能够解释80%的信息，最多保留100个主成分，这个100可以调整为其他数值，越小，规划求解越快。

        u1, lambda1 = u[:, 0:K], lambda_[0:K]  # 前K大特征值及对应的特征向量
        u2, lambda2 = u[:, K:], np.diag(lambda_[K:])  # 其余特征值

        return [u1, lambda1, np.diag(u2.dot(lambda2).dot(u2.T)) , LWShrinkCov.index ]   # 保存前k个特征值，特征向量，以及特质收益波动率,对应的股票code
