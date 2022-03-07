import DataCenter as dc
from Bean import Parameter as pr
import pandas as pd
from Factor.Stock.ParentStockFactor import ParentStockFactor
from Factor.Stock.FactorLib import FactorLib
from Factor.Stock.Quote import Quote
from Factor.Stock.Sector import SectorGroup
from Factor.Stock.Scale import MarketValue
from Factor.Stock.Model.ParentAlphaModel import ParentAlphaModel
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model
import statsmodels.api as sm
import datetime
import scipy.stats
import numpy as ny
import math
from sklearn import preprocessing
import logging
from Factor.Stock.Sector import Sector
from Bean.Tool import ToolFactor

import torch
from torch import nn

import numpy as np

import pandas as pd


class NNETModel(ParentAlphaModel):

    def __init__(self):
        super().__init__()
        self.name="神经网络模型"
        p1 = pr.Parameter("周期", "月")
        p1.enumList.append("周")
        p1.enumList.append("日")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("训练周期", "月")
        p1.enumList.append("周")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("备选因子", 0)
        p1.type = "MultiSelected"
        self.paraList[p1.name] = p1

        self.selectedFactorNames = []  # 记录选中的因子列表


        p1 = pr.Parameter("风险调整因子", 0)
        p1.type = "MultiSelected"

        # p1.multiSelected["LEV>DR"] = True
        self.paraList[p1.name] = p1
        self.selectedRiskAdjustedFactorNames = []  # 记录选中的因子列表


        p1 = pr.Parameter("训练模型", "临近周期训练")
        p1.enumList.append("相似周期训练")
        p1.enumList.append("临近周期训练")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("最小训练周期", 12)
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("最大训练周期", 180)
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("目标因子数", 10)
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("加权方法", "打分加权")
        p1.enumList.append("市值加权")
        p1.enumList.append("无")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("y处理", "无")        # 对收益率序列做处理
        p1.enumList.append("rankNorm")
        p1.enumList.append("rank")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("coefficient", 0.2)    # 计算icir作为loss时，ic/(coef+vol)，控制这里的coef
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("网络模型", "弱因子ICIR")
        p1.enumList.append("弱因子IC")
        p1.enumList.append("直接预测")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("训练轮次", [20, 5])
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("行业分类", "无")
        p1.enumList.append("中信二级动态")
        p1.enumList.append("中信一级")
        self.paraList[p1.name] = p1
        #只有当
        p1 = pr.Parameter("分组调整", "无")
        p1.enumList.append("中信一级")
        p1.enumList.append("医药非医药")
        p1.enumList.append("沪深300")
        p1.enumList.append("中证800")
        p1.enumList.append("中证1800")
        self.paraList[p1.name] = p1

        #只有当
        p1 = pr.Parameter("分组训练", "无")
        p1.enumList.append("中信一级")
        p1.enumList.append("医药非医药")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("剔除停牌", "是")
        p1.enumList.append("否")
        p1.enumList.append("是")
        self.paraList[p1.name] = p1


        p1 = pr.Parameter("目标收益天数", 40)
        p1.type = "Float"
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("收益风险调整", "否")
        self.paraList[p1.name] = p1
        self.fitModel=None
        self.characDic={} #用于存放回归属性
        self.adjusted=False #因子值是否需要计算调整值

    def init(self):
        self.adjusted = False
        testPeriod=self.paraList["周期"].value
        trainedPeriod=self.paraList["训练周期"].value

        if self.getParaValue("收益风险调整")=="是":
            m = Quote()
            m.paraList["类型"].value = "CAPAF"
            self.addNeedFactor("市值", m)

            s = Sector()
            s.setParaValue("分类", "中信一级")
            self.addNeedFactor("行业", s)


        for fName,selected in self.paraList["风险调整因子"].multiSelected.items():
            if selected:
                f = FactorLib()
                f.paraList["类型"].value = "原始值"
                f.paraList["周期"].value = trainedPeriod
                f.paraList["因子名"].value = fName
                self.addNeedFactor(fName,f)
                self.selectedRiskAdjustedFactorNames.append(fName)
                if trainedPeriod != testPeriod:
                    f = FactorLib()
                    f.paraList["类型"].value = "原始值"
                    f.paraList["周期"].value = testPeriod
                    f.factorTradingDates=dc.DataCenter.dataSet.actionDates
                    f.paraList["因子名"].value = fName
                    self.addNeedFactor(fName+"tested", f)

        if (len(self.selectedRiskAdjustedFactorNames)>0) or (self.paraList["行业分类"].value!="无") :
            self.adjusted=True
        for fName,selected in self.paraList["备选因子"].multiSelected.items():
            if selected:
                f = FactorLib()
                f.paraList["因子名"].value =fName
                f.paraList["周期"].value = trainedPeriod
                if self.adjusted:
                    f.paraList["类型"].value = "原始值"
                self.addNeedFactor(fName,f)
                self.selectedFactorNames.append(fName)
                if trainedPeriod != testPeriod:
                    f = FactorLib()
                    f.paraList["因子名"].value = fName
                    f.paraList["周期"].value = testPeriod
                    f.factorTradingDates = dc.DataCenter.dataSet.actionDates

                    if self.adjusted:
                        f.paraList["类型"].value = "原始值"
                    self.addNeedFactor(fName + "tested", f)


        q=Quote()
        q.paraList["类型"].value="ClosePriceAdj"
        self.addNeedFactor("收盘价",q)

        m=Quote()
        m.paraList["类型"].value = "OpenPriceAdj"
        self.addNeedFactor("开盘价", m)
        m = Quote()
        m.paraList["类型"].value = "AvgPriceAdj"
        self.addNeedFactor("均价", m)
        m = Quote()
        m.paraList["类型"].value = "HighPriceAdj"
        self.addNeedFactor("最高价", m)
        m = Quote()
        m.paraList["类型"].value = "LowPriceAdj"
        self.addNeedFactor("最低价", m)
        m = Quote()
        m.paraList["类型"].value = "IntradayReturn"
        self.addNeedFactor("当天收益", m)
        m = Quote()
        m.paraList["类型"].value = "OvernightReturn"
        self.addNeedFactor("隔夜收益", m)
        m = Quote()
        m.paraList["类型"].value = "DReturn"
        self.addNeedFactor("总收益", m)
        m = Quote()
        m.paraList["类型"].value = "TurnoverValue"
        self.addNeedFactor("换手率", m)
        m = Quote()
        m.paraList["类型"].value = "TurnoverVolume"
        self.addNeedFactor("成交量", m)
        m = Quote()
        m.paraList["类型"].value = "TurnoverDeals"
        self.addNeedFactor("单数", m)
        m = Quote()
        m.paraList["类型"].value = "CAPAF"
        self.addNeedFactor("市值", m)

        if self.adjusted:
            s=Sector()
            s.setParaValue("分类",self.getParaValue("行业分类"))
            self.addNeedFactor("行业",s)


        if self.paraList["分组训练"].value!="无":
            sg=SectorGroup()
            sg.paraList["分类"].value=self.paraList["分组训练"].value
            self.addNeedFactor("训练分组", sg)


        if self.paraList["剔除停牌"].value=="是":
            quote = Quote()
            quote.paraList["类型"].value = "TurnoverValue"
            self.addNeedFactor("成交量", quote)



    def loadData(self):
        #region 基础数据准备

        testPeriod = self.paraList["周期"].value
        trainedPeriod = self.paraList["训练周期"].value
        selectedTrainedDatas=self.getTrainedData()
        selectedTrainedDatas.groupby("tradingDate").apply(lambda x: x.shape)
        selectedTestDatas=None
        if testPeriod!=trainedPeriod:
            selectedTestDatas=self.getTestData()

        #endregion

        predictions = None

        if testPeriod == trainedPeriod:
            selectedTestDatas = selectedTrainedDatas


        if self.paraList["分组训练"].value == "无":
           predictions=self.fit(selectedTrainedDatas,selectedTestDatas)
        else: #不同周期分组训练有问题，待修正
            selectedTrainedDatasGroup = selectedTrainedDatas.groupby("训练分组", as_index=False, group_keys=False)
            selectedTestDatasGroup = selectedTestDatas.groupby("训练分组", as_index=False, group_keys=False)
            for name, group in selectedTrainedDatasGroup:
                logging.info("分组" + str(name) + "训练开始")
                trainedGroupData = selectedTrainedDatasGroup.get_group(name)
                testGroupData = selectedTestDatasGroup.get_group(name)
                predictGroup = self.fit(trainedGroupData, testGroupData)
                predictions = predictGroup if predictions is None else predictions.append(predictGroup)
                logging.info("分组"+name+"训练已完成")
        predictions=predictions.sort_index()
        self.addFactorDataSet(predictions)

    def getColletionValues(self,codeList,date):
        pf3 = self.getFactorDataSet()
        pf3 = pf3.loc[date]["factorValue"]
        result=pf3.reindex(codeList)
        return result

    def Symmetric_Orthogonalize(self,DF):
        F = DF.values
        M = np.dot(F.T, F)
        # D是M的特征值的向量，U是对应的特征向量
        D, U = np.linalg.eig(M)
        D = D ** (-0.5)
        if np.sum(np.isnan(D)) != 0:
            logging.error(" F.T*F矩阵的特征值过小，D中出现nan值，日期："+str(DF.iloc[0].name[0]))

            # writer = pd.ExcelWriter('output.xlsx')
            # DF.to_excel(writer, 'Sheet1')
            # print(DF)
            return DF
        # 向量转成对称阵
        D = np.diag(D)
        # 变换矩阵S
        S = np.dot(np.dot(U, D), U.T)
        # Fnew = Fintital * S
        F = np.dot(F, S)
        # print(np.sum(F[:,2] * F[:,3]))
        DF = pd.DataFrame(F, index=DF.index, columns=DF.columns)
        return DF

    def fit(self,selectedTrainedDatas,selectedTestDatas):
        minActionDate=dc.DataCenter.dataSet.actionDates.min()
        maxPeriod = self.paraList["最大训练周期"].value
        selectedTrainedDatas = selectedTrainedDatas.reset_index().set_index(["endDate","code"]).sort_index()



        minDate = dc.DataCenter.getPeriodStartDate(maxPeriod * 22, minActionDate)
        evalStartDate = minActionDate
        Xtrain = selectedTrainedDatas.loc[minDate:evalStartDate][self.selectedFactorNames + ["cap"]]
        Ytrain = selectedTrainedDatas.loc[minDate:evalStartDate]["StockReturn"]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        torch.cuda.manual_seed(100)     # 设定随机数种子，确保每次运行模型初始参数一致
        torch.manual_seed(100)          # 设定随机数种子，确保每次运行模型初始参数一致
        factorNum=len(self.selectedFactorNames)

        # 创建模型后还要初始化模型参数
        if self.paraList["网络模型"].value == "直接预测":
            model = StockReturn_pred_Model(input_factor_number=factorNum, output_factor_number=1).to(device)
            init_model_param(model, param=(torch.ones(factorNum).reshape(1, factorNum) / factorNum).to(device))       # 初始化模型参数
        elif self.paraList["网络模型"].value == "弱因子ICIR":
            model = orthogonalWeakFactors_ICIRmax(input_factor_number=factorNum, output_factor_number=self.paraList["目标因子数"].value).to(device)
            init_model_param(model, param=(
                torch.nn.init.uniform_(torch.ones([self.paraList["目标因子数"].value, factorNum]) / factorNum, a=0, b=1)).to(
                device))       # 初始化模型参数
        elif self.paraList["网络模型"].value == "弱因子IC":
            model = orthogonalWeakFactors_ICmax(input_factor_number=factorNum, output_factor_number=self.paraList["目标因子数"].value).to(device)
            init_model_param(model, param=(
                torch.nn.init.uniform_(torch.ones([self.paraList["目标因子数"].value, factorNum]) / factorNum, a=0, b=1)).to(
                device))       # 初始化模型参数
        else:
            model = None
            logging.error("！！！错误的参数值：网络模型")

        model.device = device
        model.optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        model.weightMethod = self.paraList["加权方法"].value   # 指定加权方法
        model.coefficient = self.paraList["coefficient"].value

        print(model)
        epochs = self.paraList["训练轮次"].value[0]
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            print("开始时间"+str(minDate)+"到"+str(evalStartDate))
            model = model.trainModel(Xtrain, Ytrain)


        newXtrain = model.predict(Xtrain[self.selectedFactorNames])
        newXtrain = self.Symmetric_Orthogonalize(newXtrain)
        print("NewFactorICMean:"+str(newXtrain.mean(axis=1).corr(Ytrain)))
        newDataSet=pd.concat((newXtrain, Ytrain), axis=1)
        newFactorNameList=list(newXtrain.columns)
        selectedDataIC = newDataSet.groupby("endDate").apply(
            lambda x: x[newFactorNameList].apply(lambda y: y.corr(x["StockReturn"], method='pearson')))


        predictions = None

        lastTrainMaxDate=minActionDate
        for actiondate in dc.DataCenter.dataSet.actionDates:
            logging.info("ICIR Model Fitting:" + str(actiondate))
            #如果有新的数据需要训练就训练
            dateTrain=selectedTrainedDatas.loc[lastTrainMaxDate+1:actiondate]
            if dateTrain.shape[0]>2000*5:
                dateTrainX=dateTrain[self.selectedFactorNames + ["cap"]]
                dateTrainY=dateTrain["StockReturn"]
                Xtrain = Xtrain.append(dateTrainX)
                Ytrain = Ytrain.append(dateTrainY)
                epochs = self.paraList["训练轮次"].value[1]
                for t in range(epochs):
                    print(f"Epoch {t+1}\n-------------------------------")
                    model = model.trainModel(Xtrain, Ytrain)

                newXtrain = model.predict(Xtrain[self.selectedFactorNames])
                print("NewFactorICMean:" + str(newXtrain.mean(axis=1).corr(Ytrain)))
                newDataSet = pd.concat((newXtrain, Ytrain), axis=1)
                newFactorNameList = list(newXtrain.columns)
                selectedDataIC = newDataSet.groupby("endDate").apply(
                    lambda x: x[newFactorNameList].apply(lambda y: y.corr(x["StockReturn"], method='pearson')))
                lastTrainMaxDate=actiondate

            X_test = selectedTestDatas.loc[actiondate][self.selectedFactorNames].fillna(0)
            newXtest=model.predict(X_test[self.selectedFactorNames])
            newXtrain = self.Symmetric_Orthogonalize(newXtrain)
            minDate = dc.DataCenter.getPeriodStartDate(maxPeriod * 22, actiondate)
            selectedTrainedIC = selectedDataIC.loc[minDate:actiondate]
            selectedDataICIR = selectedTrainedIC.mean() / selectedTrainedIC.std()
            Y_pred = (newXtest * selectedDataICIR).sum(axis=1)
            Y_pred = (Y_pred-Y_pred.mean())/Y_pred.std()
            codes = newXtest.index.values
            prediction = pd.DataFrame(Y_pred.values, index=pd.MultiIndex.from_product([[actiondate], codes]),
                                      columns=["factorValue"])

            if predictions is None:
                predictions = prediction
            else:
                predictions = predictions.append(prediction)
        return predictions



    #准备模型所需要的数据
    def getTrainedData(self):
        objectPeriod = int(self.paraList["目标收益天数"].value)
        baseF=self.getNeedFactor(self.selectedFactorNames[0])
        self.getNeedFactor("均价").fullName
        df = baseF.getFactorDataSet()
        startDate = self.factorTradingDates.min()
        endDate = self.factorTradingDates.max()
        dateTable = dc.DataCenter.dataSet.tradingDateTable

        trainedPeriod = self.paraList["训练周期"].value
        storedDataMin=df.index.levels[0].min()
        if trainedPeriod == "月":
            tradingPeriodDates = dateTable[(dateTable['IfMonthEnd'] == 1) & (dateTable['TradingDate'] >= startDate)& (dateTable['TradingDate'] >= storedDataMin)& (dateTable['TradingDate'] <=endDate)][
                "TradingDate"]
        elif trainedPeriod == "周":
            tradingPeriodDates = dateTable[(dateTable['IfWeekEnd'] == 1) & (dateTable['TradingDate'] >= startDate)& (dateTable['TradingDate'] >= storedDataMin) & (dateTable['TradingDate'] <= endDate)][
                "TradingDate"]
        trainFactors = df.loc[tradingPeriodDates]
        tradingDates = dc.DataCenter.dataSet.tradingDates
        stockPool = dc.DataCenter.dataSet.containerStockPool
        trainedDatas = None

        factorStoredDateList = list(tradingPeriodDates)


        for i in range(len(factorStoredDateList)):  # 计算区间收益率
            date = factorStoredDateList[i]
            #如果因子库没有存相关数据则跳过
            if date not in trainFactors.index.levels[0].values:
                continue
            currentPos = dc.DataCenter.dataSet.tradingDateDic[date]
            if objectPeriod == 0:
                if i + 1 < len(factorStoredDateList):
                    nextStoreDate = factorStoredDateList[i + 1]
                    nextDatePos = dc.DataCenter.dataSet.tradingDateDic[nextStoreDate]
                else:
                    nextDatePos = dc.DataCenter.dataSet.tradingDateDic.max() + 5
            else:
                nextDatePos = currentPos + objectPeriod
            codes = stockPool.getStockList(date)
            storedCodes = trainFactors.loc[date].index.values
            codes = list(set(codes).intersection(set(storedCodes)))
            trainedData = trainFactors.loc[date].loc[codes][self.selectedFactorNames]

            if self.adjusted:
                print(str(datetime.datetime.now()) + ": TrainedData Normalize: " + str(date))
                from Bean.Tool import ToolFactor
                X_train = pd.DataFrame()
                if (self.paraList["行业分类"].value != "无"):
                    sectorValues = self.getNeedFactor("行业").getColletionValues(codes,
                                                                               date)
                    sectorRiskMatrix = pd.get_dummies(sectorValues)
                    X_train=sectorRiskMatrix

                if (len(self.selectedRiskAdjustedFactorNames) > 0):

                    adjustedFactorValues =  trainFactors.loc[date].loc[codes][self.selectedRiskAdjustedFactorNames]
                    X_train = pd.concat([X_train, adjustedFactorValues], axis=1, join='outer')
                trainedData = trainedData.apply(lambda x:ToolFactor.normalizeValues(x, X_train))


            trainedData["cap"] = self.getNeedFactor("市值").getColletionValues(codes, date)

            if nextDatePos >= dc.DataCenter.dataSet.tradingDateDic.max():
                trainedData["endDate"]=np.nan
                trainedData["StockReturn"] = None
            else:
                nextDate = dc.DataCenter.dataSet.tradingDates.iat[nextDatePos]
                closePrices1 =self.getNeedFactor("收盘价").getColletionValues(codes, date)
                closePrices2 =self.getNeedFactor("收盘价").getColletionValues(codes, nextDate)
                returns = closePrices2 / closePrices1 - 1
                # 对于收益数据进行预处理
                # if self.adjusted:
                #    print("收益风险调整："+str(date))
                #    returns = normalizer.normalizeColletionValuesByRiskValues(returns, riskAdjustedData, date)
                trainedData["StockReturn"] = returns
                trainedData["StockReturn"] = trainedData["StockReturn"].fillna(0)
                if self.paraList["y处理"].value == "rank":
                    trainedData["StockReturn"] = trainedData["StockReturn"].rank()
                    # trainedData["StockReturn"] = preprocessing.scale(trainedData["StockReturn"])
                elif self.paraList["y处理"].value == "rankNorm":
                    from Bean.Tool import ToolFactor
                    trainedData[["StockReturn"]] = trainedData[["StockReturn"]].apply(lambda x: ToolFactor.rankNorm(x, 0, 1, 2))
                else:
                    pass
                trainedData["StockReturn"] = preprocessing.scale(trainedData["StockReturn"])
                if self.getParaValue("收益风险调整") == "是":
                    print(str(datetime.datetime.now()) + ": 收益风险调整: " + str(date))

                    X_train = pd.DataFrame()

                    sectorValues = self.getNeedFactor("行业").getColletionValues(codes,
                                                                               date)
                    sectorRiskMatrix = pd.get_dummies(sectorValues)
                    X_train = sectorRiskMatrix

                    adjustedFactorValues = self.getNeedFactor("市值").getColletionValues(codes,
                                                                               date)
                    adjustedFactorValues=np.log10(adjustedFactorValues)
                    X_train = pd.concat([X_train, adjustedFactorValues], axis=1, join='outer')
                    X_train=X_train.fillna(0)
                    trainedData["StockReturn"] = ToolFactor.normalizeValues( trainedData["StockReturn"] , X_train)
                trainedData["endDate"] =nextDate

            if self.paraList["分组训练"].value != "无":
                trainedData['训练分组'] = self.getNeedFactor("训练分组").getColletionValues(codes, date)

            # 剔除停牌
            if self.paraList["剔除停牌"].value == "是":
                trainedData["TurnoverValue"] = self.getNeedFactor("成交量").getColletionValues(codes, date)
                trainedData = trainedData[trainedData["TurnoverValue"] > 0]
            trainedData = pd.DataFrame(trainedData.values,
                                       index=pd.MultiIndex.from_product([[date], trainedData.index.values]),
                                       columns=trainedData.columns)
            # trainedData=pd.DataFrame(trainedData.values,index=pd.MultiIndex.from_product([[date],codes]),columns=trainedData.columns)

            if trainedDatas is None:
                trainedDatas = trainedData
            else:
                trainedDatas = trainedDatas.append(trainedData)
        if self.paraList["分组训练"].value != "无":
            trainedDatas=trainedDatas.dropna(subset=["训练分组"])
        trainedDatas= trainedDatas.fillna(0)
        trainedDatas.index.names = ['tradingDate', 'code']
        return trainedDatas

    def getTestData(self):
        baseF = self.getNeedFactor(self.selectedFactorNames[0]+"tested")

        df = baseF.getFactorDataSet()
        startDate = dc.DataCenter.dataSet.actionDates.min()
        endDate = dc.DataCenter.dataSet.actionDates.max()
        dateTable = dc.DataCenter.dataSet.tradingDateTable;

        trainedPeriod = self.paraList["周期"].value
        if trainedPeriod == "月":
            tradingPeriodDates = dateTable[(dateTable['IfMonthEnd'] == 1) & (dateTable['TradingDate'] >= startDate & (dateTable['TradingDate'] <= endDate))][
                "TradingDate"]
        elif trainedPeriod == "周":
            tradingPeriodDates = dateTable[(dateTable['IfWeekEnd'] == 1) & (dateTable['TradingDate'] >= startDate)& (dateTable['TradingDate'] <= endDate)][
                "TradingDate"]
        elif trainedPeriod == "日":
            tradingPeriodDates = dateTable[ (dateTable['TradingDate'] >= startDate)& (dateTable['TradingDate'] <= endDate)][
                "TradingDate"]
        #只算时间颗粒度与容器一致的日期
        tradingPeriodDates=dc.DataCenter.dataSet.actionDates
        trainFactors = df.loc[tradingPeriodDates]
        tradingDates = dc.DataCenter.dataSet.tradingDates
        stockPool = dc.DataCenter.dataSet.containerStockPool
        trainedDatas = None

        factorStoredDateList = list(tradingPeriodDates.values)
        for i in range(len(factorStoredDateList)):  # 标准化数据
            date = factorStoredDateList[i]
            codes = stockPool.getStockList(date)
            storedCodes = trainFactors.loc[date].index.values
            codes = list(set(codes).intersection(set(storedCodes)))
            trainedData = trainFactors.loc[date].loc[codes][self.selectedFactorNames]

            if self.adjusted:
                print(str(datetime.datetime.now()) + ": TrainedData Normalize: " + str(date))
                from Bean.Tool import ToolFactor
                X_train = pd.DataFrame()
                if (self.paraList["行业分类"].value != "无"):
                    sectorValues = self.getNeedFactor("行业").getColletionValues(codes,
                                                                               date)
                    sectorRiskMatrix = pd.get_dummies(sectorValues)
                    X_train = sectorRiskMatrix

                if (len(self.selectedRiskAdjustedFactorNames) > 0):
                    adjustedFactorValues = trainFactors.loc[date].loc[codes][self.selectedRiskAdjustedFactorNames]
                    X_train = pd.concat([X_train, adjustedFactorValues], axis=1, join='outer')
                trainedData = trainedData.apply(lambda x: ToolFactor.normalizeValues(x, X_train))


            if self.paraList["分组训练"].value != "无":
                trainedData['训练分组'] = self.getNeedFactor("训练分组").getColletionValues(codes,
                                                                       date)


            trainedData = pd.DataFrame(trainedData.values,
                                       index=pd.MultiIndex.from_product([[date], trainedData.index.values]),
                                       columns=trainedData.columns)
            # trainedData=pd.DataFrame(trainedData.values,index=pd.MultiIndex.from_product([[date],codes]),columns=trainedData.columns)

            if trainedDatas is None:
                trainedDatas = trainedData
            else:
                trainedDatas = trainedDatas.append(trainedData)
        if self.paraList["分组训练"].value != "无":
            trainedDatas=trainedDatas.dropna(subset=["训练分组"])
        trainedDatas= trainedDatas.fillna(0)
        trainedDatas.index.names = ['tradingDate', 'code']
        return trainedDatas


##########
##########
##########
##########
##########
##########
########## 以下为损失函数部分

class get_equal_weight_factor_IC(nn.Module):
    '''输出弱因子等权后的ic，用于orthogonalWeakFactors_ICmax模型'''
    def __init__(self):
        super().__init__()
    def forward(self, x, y, weight):
        """
        传入一个tensor格式的矩阵x(x.shape(m,n))，输出其因子等权后的与y的IC，IC可以加权计算。weight==None时，不加权
        """
        if weight is None:
            f = (x.shape[0] - 1) / x.shape[0]                                        # 修正系数,用来修正相关系数的计算
            x = x.mean(dim=1).reshape(x.shape[0], 1)                                 # 因子等权 15维 --> 1维
            x_reducemean = x - torch.mean(x, axis=0)
            y_reducemean = y - torch.mean(y, axis=0)
            numerator = torch.matmul(x_reducemean.T, y_reducemean) / x.shape[0]      # 得到分子, cov(x,y)
            denominator = x.std() * y.std() * f                                      # 得到分母, std(x) * std(y)
            IC = numerator / denominator                                             # 得到相关系数
        else:
            X = x.mean(dim=1).reshape(x.shape[0], 1).squeeze()
            X_mean = X.mul(weight).sum()
            y_mean = y.mul(weight).sum()
            X_reducemean = X - X_mean
            y_reducemean = y - y_mean
            numerator = weight.mul(X_reducemean).mul(y_reducemean).sum()  # 弱因子和y的协方差，shape(15*1)
            var_X = weight.mul(X_reducemean).mul(X_reducemean).sum()
            var_y = weight.mul(y_reducemean).mul(y_reducemean).sum()

            IC = (numerator / torch.sqrt(var_X * var_y))

        return IC



class corrMat_l2_loss(nn.Module):
    '''相关系数矩阵的l2范数作为loss'''
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        """
        传入一个tensor格式的矩阵x(x.shape(m,n))，输出其 相关系数矩阵的l2范数与n的差值（必定满足 l2-sqrt(n)>0 ）
                             (x-mean(x)) * (x-mean(x))' / N
        correlation(x) = ------------------------------------
                                sqrt(var(x)' * var(x))
        """

        f = (x.shape[0] - 1) / x.shape[0]                                       # 修正系数,用来修正相关系数的计算
        x_reducemean = x - torch.mean(x, axis=0)
        numerator = torch.matmul(x_reducemean.T, x_reducemean) / x.shape[0]     # 得到分子(x-mean(x)) * (x-mean(x))'/N, x的协方差矩阵,15*15
        var_ = x.var(axis=0).reshape(x.shape[1], 1)                             # 得到var(x),x的各个分量的方差,15维
        denominator = torch.sqrt(torch.matmul(var_, var_.T)) * f                # 得到分母sqrt(var(x)' * var(x)) ,15*15的矩阵
        corrcoef = numerator / denominator                                      # 得到相关系数矩阵
        l2 = torch.linalg.norm(corrcoef)                                        # 得到相关系数矩阵l2范数
        l2_limit = torch.tensor(np.sqrt(x.shape[1]))

        return l2 - l2_limit



class get_weakFactor_IC_vol(nn.Module):
    '''返回弱因子的平均IC，vol，用于后续计算loss
    用于orthogonalWeakFactors_ICIRmax模型'''
    def __init__(self):
        super().__init__()

    def get_weakfactor_IC(self, X, y):
        """ 计算截面的IC
        X:截面数据，单期的弱因子
        y:截面数据，单期收益率
        """

        f = (X.shape[0] - 1) / X.shape[0]
        X_reducemean = X - torch.mean(X, axis=0)
        y_reducemean = y - torch.mean(y)
        numerator = torch.matmul(X_reducemean.T, y_reducemean) / X.shape[0]
        var_X = X.var(axis=0).reshape(X.shape[1], 1)
        var_y = y.var(axis=0)
        denominator = torch.sqrt(var_X.T * var_y) * f
        IC = numerator / denominator
        return IC
        # np.corrcoef(X.cpu().detach().T, y.cpu().detach())

    def forward(self, batch_X_list, batch_y_list, list_len):
        """
        传入12期数据，计算IC，vol
        """
        IC_list = []
        for i in range(0, list_len):
            IC_list.append(self.get_weakfactor_IC(batch_X_list[i], batch_y_list[i]))   # 计算每一个截面上，弱因子的IC
        IC_tensorArray = torch.cat(IC_list, dim=0)          # 将这12个截面IC合并 IC_tensorArray的形状: ([12, 15])
        IC = IC_tensorArray.mean(axis=0)                    # 计算IC平均值
        vol = IC_tensorArray.std(axis=0)                    # 计算vol

        return IC, vol




class get_IC_vol(nn.Module):
    '''直接预测股票收益率时用到的，返回pred的IC，vol
    用于StockReturn_pred_Model模型'''
    def __init__(self):
        super().__init__()

    def get_weakfactor_IC(self, X, y, w):
        """ 计算截面的加权IC
        X:截面数据，单期预测值
        y:截面数据，单期收益率
        """
        weight = w
        if weight is not None:

            X_mean = X.mul(weight).sum()
            y_mean = y.mul(weight).sum()
            X_reducemean = X - X_mean
            y_reducemean = y - y_mean
            numerator = weight.mul(X_reducemean).mul(y_reducemean).sum()  # 弱因子和y的协方差，shape(15*1)
            var_X = weight.mul(X_reducemean).mul(X_reducemean).sum()
            var_y = weight.mul(y_reducemean).mul(y_reducemean).sum()

            IC = (numerator / torch.sqrt(var_X * var_y))
            return IC.reshape(1, 1)
        else:
            f = (X.shape[0] - 1) / X.shape[0]
            X_reducemean = X - torch.mean(X, axis=0)
            y_reducemean = y - torch.mean(y)
            numerator = torch.matmul(X_reducemean.T, y_reducemean) / X.shape[0]  # 弱因子和y的协方差，shape(15*1)
            var_X = X.var(axis=0).reshape(X.shape[1], 1)
            var_y = y.var(axis=0)
            denominator = torch.sqrt(var_X.T * var_y) * f
            IC = numerator / denominator
            return IC

    def get_pred_real_IC(self, X, y):
        """ 计算截面的真实IC，方便对比加权ic和真实ic的区别，真实ic不参与loss的计算
        X:截面数据，单期预测值
        y:截面数据，单期收益率
        """

        f = (X.shape[0] - 1) / X.shape[0]
        X_reducemean = X - torch.mean(X, axis=0)
        y_reducemean = y - torch.mean(y)
        numerator = torch.matmul(X_reducemean.T, y_reducemean) / X.shape[0]  # 弱因子和y的协方差，shape(15*1)
        var_X = X.var(axis=0).reshape(X.shape[1], 1)
        var_y = y.var(axis=0)
        denominator = torch.sqrt(var_X.T * var_y) * f
        IC = numerator / denominator
        return IC

    def forward(self, batch_X_list, batch_y_list, batch_weight_list, list_len):
        """
        传入list_len长度的数据，计算IC，vol
        """
        weightIC_list, realIC_list = [], []
        for i in range(0, list_len):
            weightIC_list.append(self.get_weakfactor_IC(batch_X_list[i], batch_y_list[i], batch_weight_list[i]))   # 计算每一个截面上，加权IC
            realIC_list.append(self.get_pred_real_IC(batch_X_list[i], batch_y_list[i]))  # 计算每一个截面上，真实IC
        weightIC_tensorArray = torch.cat(weightIC_list, dim=0)          # 将这12个截面IC合并 IC_tensorArray的形状: ([12, 15])
        weightIC = weightIC_tensorArray.mean(axis=0)                    # 计算IC平均值
        weightICvol = weightIC_tensorArray.std(axis=0)                    # 计算vol

        realIC_tensorArray = torch.cat(realIC_list, dim=0)  # 将这12个截面IC合并 IC_tensorArray的形状: ([12, 15])
        realIC = realIC_tensorArray.mean(axis=0)  # 计算IC平均值
        realICvol = realIC_tensorArray.std(axis=0)  # 计算vol

        return weightIC, weightICvol, realIC, realICvol


##########
##########
##########
##########
##########
##########
########## 以下为网络模型部分



class orthogonalWeakFactors_ICmax(nn.Module):
    '''
    输出正交的弱因子，使得弱因子的ic最大化，需要用到的loss：corrMat_l2_loss，equal_weight_IC_loss
    '''
    def __init__(self, input_factor_number, output_factor_number):
        super(orthogonalWeakFactors_ICmax, self).__init__()
        self.linearTransform = nn.Linear(in_features=input_factor_number,
                                         out_features=output_factor_number, bias=False)
        self.BatchNormalization = nn.BatchNorm1d(num_features=output_factor_number)
        self.corrMat_l2_loss = corrMat_l2_loss()
        self.get_equal_weight_factor_IC = get_equal_weight_factor_IC()

        self.optimizer = None
        self.device = None
        self.weightMethod = None

    def forward(self, x):  # 定义模型, 输入 x --> linearTransform --> BN 输出
        linearTransform = self.linearTransform(x)
        BN = self.BatchNormalization(linearTransform)
        return BN

    def trainModel(self, X_train, y_train):
        """单个epoch的训练,对X_train,y_train中的tradingDate做循环,每一个tradingDate的截面数据为一个batch"""
        trainLossList, trainLossl2List, trainLossICList = [], [], []  # 用来存放每个batch的总loss,相关系数矩阵l2范数loss,因子加权后的ICloss
        ICList = []  # 用来存放每一个batch的因子加权后的IC, IC=sqrt(1-ICloss)
        trainDataSet = pd.concat((X_train, y_train), axis=1)  # 将X_train,y_train合并
        model=self
        model.train()
        for tradingDate, sectionDF in trainDataSet.groupby("endDate"):
            X = torch.tensor(sectionDF.drop("cap", axis=1).values[:, :-1], dtype=torch.float32).to(
                self.device)  # 这个模型好像只接受float32格式,float64不行
            y = torch.tensor(sectionDF.values[:, -1], dtype=torch.float32).to(model.device)

            pred = model(X)  # 得到多个弱因子
            if self.weightMethod ==  "无":
                weight = None
            elif self.weightMethod ==  "打分加权":
                weight = pred
                weight = weight.mean(axis=1)
                a = weight - weight.min() + 1
                weight = a / a.sum()
            elif self.weightMethod ==  "市值加权":
                weight = torch.tensor(sectionDF["cap"].values, dtype=torch.float32).to(self.device)
                weight = weight + weight.min() + weight.min()
                weight = weight / weight.sum()
            else:
                weight = None
            IC = self.get_equal_weight_factor_IC(pred, y, weight)
            loss_l2 = self.corrMat_l2_loss(pred, y)  # 得到loss_l2
            loss = loss_l2 + 1 - IC  # 得到总loss

            # 模型求解,参数更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 保存每一个batch的各种loss以及IC数据
            trainLossl2List.append(loss_l2.item())
            trainLossICList.append((1-IC).item())
            trainLossList.append(loss.item())
            ICList.append(IC.item())

        print(
            f"train_loss: {np.mean(trainLossList) :>8f} , l2_loss: {np.mean(trainLossl2List) :>8f}, IC_loss: {np.mean(trainLossICList) :>8f}, train IC: {np.mean(ICList) :>8f}")

        return model

    def predict(self, Xtest):
        model=self
        model.eval()

        # 查看模型在训练集的表现:
        test_x = torch.tensor(Xtest.values, dtype=torch.float32)

        with torch.no_grad():
            pred = model(test_x.to(self.device))
        xNew=torch.Tensor.cpu(pred).numpy()
        xNew=pd.DataFrame(data=xNew,index=Xtest.index)
        return xNew



class orthogonalWeakFactors_ICIRmax(nn.Module):
    '''正交弱因子，使得弱因子的平均icir最大化'''
    def __init__(self, input_factor_number, output_factor_number):
        super(orthogonalWeakFactors_ICIRmax, self).__init__()
        self.linearTransform = nn.Linear(in_features=input_factor_number,
                                         out_features=output_factor_number, bias=False)   # 定义线性变换层,输入25个因子,输出15个因子
        self.BatchNormalization = nn.BatchNorm1d(num_features=output_factor_number)       # 定义批标准化层,输出批标准化后的15个因子

        self.corrMat_l2_loss = corrMat_l2_loss()
        self.get_weakFactor_IC_vol = get_weakFactor_IC_vol()

        self.optimizer = None
        self.device = None
        self.weightMethod = None

    def forward(self, x):   # 定义模型, 输入 x --> linearTransform --> BN 输出
        linearTransform = self.linearTransform(x)
        BN = self.BatchNormalization(linearTransform)
        return BN

    def trainModel(self, X_train, y_train):
        """
                单个epoch的训练,包含多个batch,对X_train,y_train中的tradingDate做循环,每一个batch训练timeSpan跨度的数据，
                例如：timeSpan=12,则每一个batch包含12个月的截面数据
                batch的滚动模式：             batch1 :  2020-1 : 2020-12
                                            batch2 :  2020-2 : 2021-1
                                            batch3 :  2020-3 : 2021-2
                                                ...
                                                ...

                                     IC
                loss = 1 + l2 - -------------  ， 其中0.2这个超参数可变
                                   vol+0.2
                此外还计算了多种指标：
                1. 每个因子的IC的平均    2. 每个因子IC的vol的平均    3. 每个因子ICIR的平均
                4. 等权因子的IC         5. 等权因子IC的vol         6. 等权因子ICIR
                这些指标每个batch计算，多个batch的平均值作为这一轮的指标
                """
        device = self.device
        corrMat_l2_loss = self.corrMat_l2_loss
        get_weakFactor_IC_vol = self.get_weakFactor_IC_vol
        model = self
        optimizer = self.optimizer
        coefficient = self.coefficient


        model.train()
        trainLossList, trainLossl2List, trainICList, trainvolList, trainICIRList = [], [], [], [], []  # 用来存放每个batch的总loss,相关系数矩阵l2范数loss,因子加权后的ICloss
        eq_wei_factor_ICList, eq_wei_factor_volList, eq_wei_factor_ICIRList = [], [], []

        X_list, y_list = [], []
        X_train.groupby("endDate").apply(
            lambda x: X_list.append(torch.tensor(x.drop(["cap"], axis=1).values, dtype=torch.float32).to(device)))
        y_train.groupby("endDate").apply(
            lambda x: y_list.append(torch.tensor(x.values, dtype=torch.float32).to(device)))

        timeSpan = X_list.__len__() - 5
        for i in range(0, X_list.__len__() - timeSpan):
            # 分batch训练
            batch_loss_l2 = torch.tensor(0., dtype=torch.float32)
            batch_X_list, batch_y_list, eq_wei_factor_IC_list = [], [], []  # 用来存放每一个batch的训练数据，以及得到的等权因子（等权因子=弱因子求平均）
            for j in range(i, i + timeSpan):
                pred = model(X_list[j])  # 得到单期弱因子
                batch_loss_l2 = batch_loss_l2 + corrMat_l2_loss(pred, y_list[j])  # 得到弱因子相关系数矩阵l2范数的loss
                eq_wei_factor = pred.mean(axis=1)  # 得到等权因子
                eq_wei_factor_IC = np.corrcoef(eq_wei_factor.cpu().detach(), y_list[j].cpu().detach())[0, 1]  # 等权因子的IC
                eq_wei_factor_IC_list.append(eq_wei_factor_IC)

                batch_X_list.append(pred)  # 存放弱因子
                batch_y_list.append(y_list[j])
            # batch_X_list 里面有12期的弱因子数据，根据整个list可以计算IR
            IC, vol = get_weakFactor_IC_vol(batch_X_list, batch_y_list, timeSpan)

            loss = 1 + 1 * batch_loss_l2 / timeSpan - (IC / (coefficient + vol)).mean()

            # 模型求解,参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            eq_wei_factor_IC = np.mean(eq_wei_factor_IC_list)  # 保存等权因子的IC
            eq_wei_factor_vol = np.std(eq_wei_factor_IC_list)  # 保存等权因子的vol
            eq_wei_factor_ICIR = eq_wei_factor_IC / eq_wei_factor_vol  # 保存等权因子的ICIR
            # 保存每一个batch的各种loss以及IC数据
            IC = IC.mean()  # 弱因子IC平均值
            vol = vol.mean()  # 弱因子IC标准差平均值

            trainLossList.append(loss.item())
            trainLossl2List.append(batch_loss_l2.item() / timeSpan)
            trainICList.append(IC.item())
            trainvolList.append(vol.item())
            trainICIRList.append((IC / vol).mean().item())
            eq_wei_factor_ICList.append(eq_wei_factor_IC)
            eq_wei_factor_volList.append(eq_wei_factor_vol)
            eq_wei_factor_ICIRList.append(eq_wei_factor_ICIR)

        print("总loss \t 弱因子相关系数矩阵l2 \t 弱因子IC的平均值 \t 等权因子的IC \t 弱因子IC的标准差的平均值 \t 等权因子IC的标准差 \t 弱因子ICIR的平均值 \t 等权因子ICIR")
        print(
            f"{np.mean(trainLossList) :>8f} \t {np.mean(trainLossl2List) :>8f} \t\t\t {np.mean(trainICList) :>8f} \t\t {np.mean(eq_wei_factor_ICList) :>8f} \t\t {np.mean(trainvolList) :>8f} \t\t\t {np.mean(eq_wei_factor_volList) :>8f} \t\t\t {np.mean(trainICIRList) :>8f} \t\t\t {np.mean(eq_wei_factor_ICIRList) :>8f}")

        return model



    def predict(self, Xtest):
        model=self
        model.eval()

        # 查看模型在训练集的表现:
        test_x = torch.tensor(Xtest.values, dtype=torch.float32)

        with torch.no_grad():
            pred = model(test_x.to(self.device))
        xNew=torch.Tensor.cpu(pred).numpy()
        xNew=pd.DataFrame(data=xNew,index=Xtest.index)
        return xNew





class StockReturn_pred_Model(nn.Module):
    '''直接预测收益率，不经过弱因子这一步骤'''
    def __init__(self, input_factor_number, output_factor_number):
        super(StockReturn_pred_Model, self).__init__()
        self.linearTransform = nn.Linear(in_features=input_factor_number,
                                         out_features=output_factor_number, bias=False)   # 定义线性变换层,输入25个因子,输出15个因子
        self.BatchNormalization = nn.BatchNorm1d(num_features=output_factor_number)       # 定义批标准化层,输出批标准化后的15个因子

        self.get_IC_vol = get_IC_vol()
        self.coefficient = None
        self.optimizer = None
        self.device = None
        self.weightMethod = None


    def forward(self, x):   # 定义模型, 输入 x --> linearTransform --> BN 输出
        linearTransform = self.linearTransform(x)
        BN = self.BatchNormalization(linearTransform)
        return BN

    def trainModel(self, X_train, y_train):
        """
         timeSpan: 时间跨度，决定每个batch传入多少个月数据
         coefficien: 系数，决定计算loss时vol项之前的系数。该值越大，模型vol越大。该值越小，模型会让vol越来越小。例如：
                                                     IC
                                 loss = 1 - -------------------
                                              vol+coefficient
         """
        model=self
        device=self.device
        optimizer=self.optimizer

        coefficient = self.coefficient
        model.train()
        trainLossList, ICList, volList, ICIRList, realICList, realICvolList, realICIRList = [], [], [], [], [], [], []

        X_list, y_list, weight_list = [], [], []
        X_train.groupby("endDate").apply(lambda x: X_list.append(torch.tensor(x.drop(["cap"], axis=1).values, dtype=torch.float32).to(device)))
        y_train.groupby("endDate").apply(lambda x: y_list.append(torch.tensor(x.values, dtype=torch.float32).to(device)))
        X_train.groupby("endDate").apply(lambda x: weight_list.append(torch.tensor(x["cap"].values, dtype=torch.float32).to(device)))     # 这里虽然是把市值放到weight_list里面，但是后面也可以不用，直接用pred作为权重

        timeSpan= X_list.__len__()-5

        for i in range(0, X_list.__len__() - timeSpan):
            batch_X_list, batch_y_list, batch_weight_list = [], [], []  # 用来存放每一个batch的训练数据
            for j in range(i, i + timeSpan):
                pred = model(X_list[j])  # 得到预测值
                batch_X_list.append(pred)
                batch_y_list.append(y_list[j])

                if self.weightMethod == "无":
                    weight = None
                elif self.weightMethod == "打分加权":
                    weight = pred
                    weight = weight.mean(axis=1)
                    a = weight - weight.min() + 1
                    weight = a / a.sum()
                elif self.weightMethod == "市值加权":
                    weight = weight_list[j]
                    weight = weight + weight.min() + weight.min()
                    weight = weight / weight.sum()
                else:
                    weight = None
                batch_weight_list.append(weight)
            # 每一个batch_X_list里面存放了timeSpan期预测值，每一个batch_y_list里存放timeSpan期真实值
            IC, vol, realIC, realICvol = self.get_IC_vol(batch_X_list, batch_y_list, batch_weight_list, timeSpan)  # 得到timeSpan期IC均值，标准差

            loss = 1 - (IC / (vol + coefficient))

            # 模型求解,参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 保存每一个batch的各种loss以及指标
            ICIR = IC / vol
            trainLossList.append(loss.item())
            ICList.append(IC.item())
            volList.append(vol.item())
            ICIRList.append(ICIR.item())
            realICList.append(realIC.item())
            realICvolList.append(realICvol.item())
            realICIRList.append(realIC.item() / realICvol.item())
        print(
            f"{np.mean(trainLossList) :>8f},   IC: {np.mean(ICList) :>8f},   vol: {np.mean(volList) :>8f},   ICIR: {np.mean(ICIRList) :>8f},    真实IC: {np.mean(realICList) :>8f},   真实ICvol: {np.mean(realICvolList) :>8f},   真实ICIR: {np.mean(realICIRList) :>8f}")

        return model

    def predict(self, Xtest):
        model = self
        model.eval()

        # 查看模型在训练集的表现:
        test_x = torch.tensor(Xtest.values, dtype=torch.float32)

        with torch.no_grad():
            pred = model(test_x.to(self.device))
        xNew = torch.Tensor.cpu(pred).numpy()
        xNew = pd.DataFrame(data=xNew, index=Xtest.index)
        return xNew





def init_model_param(model, param):
    """将模型参数初始化为param，只能初始化linear层的参数，BN层参数不用初始化"""

    for x in model.modules():
        if isinstance(x, nn.Linear):
            x.weight.data = param
            break
