
from Factor.Event.ParentEventFactor import ParentEventFactor
from Factor.Public.ParentPublicFactor import ParentPublicFactor
import pandas as pd
import DataCenter as dc
from Factor.Stock.Quote import Quote
from Bean.Parameter import Parameter
from Factor.Stock.ParentStockFactor import ParentStockFactor
from Factor.Stock.FactorLib import FactorLib
from Bean.Tool import ToolFactor
from statsmodels import regression
import cvxpy as cp
import logging
from matplotlib import pyplot as plt
import numpy as ny

from scipy.optimize import minimize






class EventFactorModel(ParentStockFactor):
    def __init__(self):
        super().__init__()
        self.name = "事件因子"
        p1 = Parameter("训练周期", "自然年")
        p1.enumList.append("自然半年")
        p1.enumList.append("年")
        p1.enumList.append("月")
        p1.enumList.append("季")
        self.paraList[p1.name] = p1
        p1 = Parameter("目标收益", 90)
        self.paraList[p1.name] = p1

        p1 = Parameter("买入滞后", 1)
        self.paraList[p1.name] = p1

        p1 = Parameter("预测周期", 20)
        self.paraList[p1.name] = p1

        p1 = Parameter("衰减方式", "拟合衰减")
        p1.enumList.append("线性衰减")
        self.paraList[p1.name] = p1

        p1 = Parameter("拟合方式", "二次函数")
        p1.enumList.append("三次函数")
        self.paraList[p1.name] = p1

        p1 = Parameter("画出拟合图像", "是")
        p1.enumList.append("否")
        self.paraList[p1.name] = p1

        p1 = Parameter("回归模型", "提升树回归")
        p1.enumList.append("线性回归")
        self.paraList[p1.name] = p1

        p1 = Parameter("备选因子", 0)
        p1.type = "MultiSelected"
        self.paraList[p1.name] = p1

        p1 = Parameter("事件", 0)
        p1.type = "MultiFactor"
        self.paraList[p1.name] = p1
        self.eventFactor=None
        self.selectedFactorNames=[]
        self.modelDic={}

    def init(self):
        super().init()
        f=SpecialReturn()
        self.addNeedFactor("特质收益",f)

        multiFactors =self.paraList["事件"].multiSelected

        for f, fChars in multiFactors.items():
            f = self.addNeedFactor(f.getFactorName(), f)
            self.eventFactor=f
        self.selectedFactorNames=[]
        for fName, selected in self.paraList["备选因子"].multiSelected.items():
            if selected:
                f = FactorLib()
                f.paraList["因子名"].value = fName
                f.paraList["周期"].value = "月"
                f.paraList["类型"].value = "原始值"
                self.addNeedFactor(fName, f)
                self.selectedFactorNames.append(fName)
        self.eventFactorNames=[]
        self.dataSetName=self.eventFactor.name+"因子"
        self.modelDic={} #用于存放训练好的模型
        self.decayModelDic = {}  # 用于存放训练好事件衰减模型


    def loadData(self):
        eventList=self.getTrainData()
        holdPeriod = self.getParaValue("目标收益")
        tradingDates=dc.DataCenter.dataSet.tradingDates
        dateTable=dc.DataCenter.dataSet.tradingDateTable
        maxActionDate=dc.DataCenter.dataSet.actionDates.max()
        startDate=dc.DataCenter.dataSet.actionDates.min()
        periodActionDates=None
        if self.getParaValue("训练周期")=="自然年":
            lastPeriodDate=dateTable[
                (dateTable['TradingDate'] <=startDate) & (
                        dateTable['TradingDate'] <= maxActionDate) & (
                        dateTable['IfMonthEnd'] == 1) & ((dateTable['TradingDate'] // 100 % 100).isin([12]))][
                "TradingDate"].max()
            periodActionDates=dateTable[
                (dateTable['TradingDate'] >= lastPeriodDate) & (
                        dateTable['TradingDate'] <= maxActionDate) & (
                        dateTable['IfMonthEnd'] == 1) & ((dateTable['TradingDate'] // 100 % 100).isin([12]))][
                "TradingDate"]
        if self.getParaValue("训练周期")=="月":
            lastPeriodDate=dateTable[
                (dateTable['TradingDate'] <= startDate) & (
                        dateTable['TradingDate'] <= maxActionDate) & (
                        dateTable['IfMonthEnd'] == 1)][
                "TradingDate"].max()
            periodActionDates=dateTable[
                (dateTable['TradingDate'] >= lastPeriodDate) & (
                        dateTable['TradingDate'] <= maxActionDate) & (
                        dateTable['IfMonthEnd'] == 1  )][
                "TradingDate"]

        factorList=list(eventList.columns)
        for pRange in range(0,holdPeriod,1):
            factorList.remove(pRange)
        factorList.remove("value")
        factorList.remove("calEndDate")
        factorList.remove("calStartDate")
        self.eventFactorNames=factorList
        for actionDate in periodActionDates:

            trainData = eventList[eventList["calEndDate"] <= actionDate]
            Y = trainData[holdPeriod - 1].fillna(0)
            X = trainData[factorList].fillna(0)
            if self.paraList["回归模型"].value == "线性回归":
                regModel = regression.linear_model.OLS(Y, X).fit()
            elif self.paraList["回归模型"].value == "提升树回归":
                import lightgbm as lgb
                params = {
                    'max_depth': 100,  # 树的最大深度
                    'learning_rate': 0.01,  # 学习率
                    'num_iterations': 20,  # 迭代次数，等价于树的个数
                    'boosting_type': 'gbdt',  # 学习方法 gbdt 代表梯度提升树(Gradient Boosting Decision Tree)，
                    'objective': 'regression',  # 用来选择做提升树回归还是提升树分类
                    'metric': 'l2',  # 损失函数 l2 代表残差平方和， l1代表残差绝对值和
                    'num_leaves': 32,  # 一棵树的最大叶子个数
                    'verbosity': -1  # 是否打印训练过程，等于2时可以看到每棵树的信息
                }
                regModel = lgb.train(params=params, train_set=lgb.Dataset(X, Y))

                # lgb.plot_tree(regModel, tree_index=2)   # 可以画出第几棵树
                # lgb.plot_importance(regModel)       #  画出因子的重要性
            else:
                logging.error("错误的回归模型！")
            self.modelDic[actionDate]=regModel

            #region 训练时间衰减模型

            if self.paraList["衰减方式"].value == "拟合衰减":
                if str(type(regModel)) == "<class 'statsmodels.regression.linear_model.RegressionResultsWrapper'>":
                    y_bar = regModel.predict(X).values
                else:
                    y_bar = regModel.predict(X)
                    # 确保y_bar输出是array格式
                y_bar = y_bar.reshape(y_bar.shape[0], 1)    # y_bar是预测的90天收益

                # 利用预测的90天收益（y_bar），计算出1-90天内每一天的b_t，t天真实累计收益 = 90天累计收益*b_t
                b_t_df = self.get_bt(y_bar, holdPeriod, trainData)

                # b_t是关于t的函数，但是不够光滑，因此用二次函数去拟合b_t
                if self.paraList["拟合方式"].value == "二次函数":
                    para = self.get_curve(b_t_df, holdPeriod)   # para是二次函数的三个位置参数，array格式
                    print(actionDate, "二次函数为: " + str(para[0]), " * t^2 + " + str(para[1]) + " * t + " + str(para[0]))
                    def beta_curve(t, a, b, c):
                        return a * ny.power(t, 2) + b * t + c
                elif self.paraList["拟合方式"].value == "三次函数":
                    para = self.get_curve3(b_t_df, holdPeriod)  # para是三次函数的四个位置参数，array格式
                    def beta_curve(t, a, b, c, d):
                        return a * ny.power(t, 3) + b * ny.power(t, 2) + c * t + d

                self.decayModelDic[actionDate] = para
                if self.paraList["画出拟合图像"].value == "是":
                    if self.paraList["拟合方式"].value == "二次函数":
                        b_t_df["curve"] = b_t_df["t"].map(lambda t: beta_curve(t, para[0], para[1], para[2]))
                    elif self.paraList["拟合方式"].value == "三次函数":
                        b_t_df["curve"] = b_t_df["t"].map(lambda t: beta_curve(t, para[0], para[1], para[2], para[3]))

                    b_t_df[[0, "curve"]].plot()

            elif self.paraList["衰减方式"].value == "线性衰减":
                self.decayModelDic[actionDate] = None
            else:
                logging.error("错误的衰减方式！")
            #endregion

        self.storeDataSet(eventList)

    #得到针对每个事件所需要的训练数据
    def getTrainData(self):
        specialReturnsDF = self.getNeedFactor("特质收益").retrieveDataSet()
        specialReturnsDF=specialReturnsDF.fillna(0).sort_index()
        tradingDates=dc.DataCenter.dataSet.tradingDates
        eventStartDate=dc.DataCenter.dataSet.tradingDates.min()
        endDate=dc.DataCenter.dataSet.tradingDates.max()
        eventList = self.eventFactor.getEventList(eventStartDate,endDate)
        eventList = eventList.loc[eventStartDate:endDate]


        holdPeriod=self.getParaValue("目标收益")

        delayPeriod = self.paraList["买入滞后"].value




        # region 获取离事件发生日最近的因子库的因子值


        if len(self.selectedFactorNames) > 0:

            factorF = self.getNeedFactor(self.selectedFactorNames[0])
            factorDF = factorF.retrieveDataSet()
            tradingDateDic = factorF.tradingDateDic

            def getFactorValue(df, fName):
                date = df.index.get_level_values("announceDate").values[0]
                code = df.index.get_level_values("code").values[0]
                storeDate = tradingDateDic.at[date]

                if storeDate > 0:
                    result = factorDF.loc[storeDate][fName]
                    if code in result:
                        return result.loc[code]
                return ny.nan



            for fName in self.selectedFactorNames:
                eventList[fName] = eventList.groupby(
                    eventList.index).apply(lambda x: getFactorValue(x, fName))

        # endregion

        # region 得到预测周期内事件的特质收益累计值
        eventList=eventList.reset_index()
        eventList["calStartDate"]=ToolFactor.delayTradingDates(eventList["announceDate"],tradingDates,delayPeriod)
        eventList["calEndDate"] = ToolFactor.delayTradingDates(eventList["announceDate"],tradingDates, holdPeriod)
        codeList=list(specialReturnsDF.columns)
        eventList=eventList[eventList["code"].isin(codeList)]
        eventList=eventList.set_index(["announceDate","code"])
        def getSpecialReturn(x):
            code = x.index.get_level_values("code").values[0]
            calStartDate=x["calStartDate"].values[0]
            calEndDate = x["calEndDate"].values[0]
            sReturn=specialReturnsDF[code].loc[calStartDate:calEndDate]
            sReturn=(1+sReturn).cumprod()-1
            sReturn=sReturn.reset_index(drop=True)
            return sReturn

        eventReturns= eventList.groupby(["announceDate","code"]).apply(lambda x: getSpecialReturn(x))
        eventReturns=eventReturns.unstack()
        eventList=pd.concat([eventList,eventReturns],axis=1)

        # endregion

        return eventList
    def getColletionValues(self,codeList,date):
        holdPeriod = self.getParaValue("目标收益")
        modelDic = self.modelDic
        actionDates = list(modelDic.keys())
        actionDates = [n for n in actionDates if n <= date]
        lastActionDate = max(actionDates)
        model = modelDic[lastActionDate]
        eventList = self.retrieveDataSet()
        eventList = eventList.loc[:date]
        eventList = eventList[eventList["calEndDate"] >= date]
        X = eventList[self.eventFactorNames].fillna(0)
        Y = model.predict(X)
        Y = pd.Series(Y, index=X.index)
        Y.name = "y"
        Y = Y.sort_index().reset_index()
        Y = Y.drop_duplicates(subset=["code"], keep='last')
        # 获得当前日期距离事件发生时间的距离，根据距离和时间衰减模型计算当前未来20天的预期收益
        Y["decay"] = ToolFactor.calDateDistance(Y["announceDate"], dc.DataCenter.dataSet.tradingDateDic, date)

        if self.paraList["衰减方式"].value == "线性衰减":
            Y["y"]=(holdPeriod-Y["decay"])*1.0/holdPeriod*Y["y"]
        elif self.paraList["衰减方式"].value == "拟合衰减":
            decayModel = self.decayModelDic[lastActionDate]    # array格式，存放预期向前beta曲线的参数
            if self.paraList["拟合方式"].value == "二次函数":
                a, b, c = decayModel
                def beta_curve(t, a, b, c):
                    return a * ny.power(t, 2) + b * t + c
                Y["y"] = Y["y"] * (beta_curve(holdPeriod, a, b, c) - beta_curve(Y["decay"], a, b, c))
            elif self.paraList["拟合方式"].value == "三次函数":
                a, b, c, d = decayModel
                def beta_curve(t, a, b, c, d):
                    return a * ny.power(t, 3) + b * ny.power(t, 2) + c * t + d
                Y["y"] = Y["y"] * (beta_curve(holdPeriod, a,b,c,d) - beta_curve(Y["decay"], a,b,c,d))
        else:
            logging.error("错误的衰减方式!")
        Y = Y.set_index("code")["y"]
        result = Y.reindex(codeList)
        result.name = self.fullName
        return result



    def get_bt(self, y_bar, holdPeriod, trainData):
        '''利用预测的90天收益（y_bar），计算出1-90天内每一天的b_t，t天真实累计收益 = 90天累计收益*b_t
        y_bar： 预测的事件90天收益
        holdPeriod： 目标收益'''
        b_t_list = []  # 存放90天每天的b_t
        for t in range(0, holdPeriod):
            # x = sm.add_constant(y_bar)
            x = y_bar
            b_t = ny.linalg.inv(x.T.dot(x)).dot(x.T).dot(trainData[[t]].values)  # 计算出b_t
            b_t_list.append(b_t[0][0])
        b_t_df = pd.DataFrame(b_t_list)
        return b_t_df# 将b_t保存到dataframe中

    def get_curve(self, b_t_df, holdPeriod):
        '''b_t是关于t的函数，但是不够光滑，因此用二次函数去拟合b_t
        return: 二次函数的三个位置参数a,b,c，是array格式， a*t^2 +b*t + c'''
        b_t_df["t"] = b_t_df.index + 1
        b_t_df["t2"] = (b_t_df["t"]) ** 2
        b_t_df["1"] = 1

        t = b_t_df[["t2", "t", "1"]].values
        b = b_t_df[0].values

        x = cp.Variable(3)
        cost = cp.sum_squares(t @ x - b)
        constraints = [x[0] <= -1e-8, x[1] >= 1e-8, x[2] >= 1e-8, 240 * x[0] + x[1] >= 1e-8, x[0]+x[1]+x[2] >= 1e-8,
                       holdPeriod**2 * x[0] + holdPeriod * x[1] + x[2] - b[holdPeriod - 1] == 0]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        return x.value



    def get_curve3(self, b_t_df, holdPeriod):
        '''用三次函数去拟合b_t
        return: 三次函数的四个位置参数a,b,c,d，是array格式'''
        b_t_df["t"] = b_t_df.index + 1
        b_t_df["t2"] = (b_t_df["t"]) ** 2
        b_t_df["t3"] = (b_t_df["t"]) ** 3
        b_t_df["1"] = 1

        t = b_t_df[["t3", "t2", "t", "1"]].values
        b = b_t_df[0].values

        x = cp.Variable(4)
        cost = cp.sum_squares(t @ x - b)
        constraints = [x[0]+x[1]+x[2]+x[3] >= 1e-8,
                       holdPeriod**3 * x[0] + holdPeriod**2 * x[1] + holdPeriod*x[2] + x[3]- b[holdPeriod - 1] ==0]
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()
        return x.value




