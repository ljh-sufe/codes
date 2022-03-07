import DataCenter as dc
from Bean import Parameter as pr
import pandas as pd
from Factor.Stock.ParentStockFactor import ParentStockFactor;
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
class ICIRModel(ParentAlphaModel):
    # 因子库因子，因子库
    def __init__(self):
        super().__init__()
        self.name="ICIR模型"
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

        p1 = pr.Parameter("y处理", "无")
        p1.enumList.append("rankNorm")
        p1.enumList.append("rank")
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
        p1.type = "Float";
        self.paraList[p1.name] = p1
        self.fitModel=None
        self.characDic={} #用于存放回归属性
        self.adjusted=False #因子值是否需要计算调整值

    def init(self):
        self.adjusted = False
        testPeriod=self.paraList["周期"].value
        trainedPeriod=self.paraList["训练周期"].value
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
        m.paraList["类型"].value = "CAPAF"
        self.addNeedFactor("市值", q)


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
        selectedTrainedDatas = self.getTrainedData()
        selectedTestDatas = None
        if testPeriod!=trainedPeriod:
            selectedTestDatas=self.getTestData()

        #endregion

        predictions = None
        selectedTrainedDatas.groupby("tradingDate").apply(lambda x: x.shape)
        selectedTrainedDatas[self.selectedFactorNames] = selectedTrainedDatas[self.selectedFactorNames].groupby(
            "tradingDate").apply(self.Symmetric_Orthogonalize)
        logging.info("训练数据已经完成对称正交")
        if testPeriod == trainedPeriod:
            selectedTestDatas = selectedTrainedDatas
        else:
            selectedTestDatas[self.selectedFactorNames] = selectedTestDatas[self.selectedFactorNames].groupby(
                "tradingDate").apply(self.Symmetric_Orthogonalize)
            logging.info("测试数据已经完成对称正交")


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

        selectedDataIC = selectedTrainedDatas.groupby("endDate").apply(
            lambda x: x[self.selectedFactorNames].apply(lambda y: y.corr(x["StockReturn"], method='spearman')))

        maxPeriod = self.paraList["最大训练周期"].value
        dateTable = dc.DataCenter.dataSet.tradingDateTable;
        startDate = self.factorTradingDates.min()

        predictions = None
        if selectedTestDatas is None:
            selectedTestDatas=selectedTrainedDatas

        for actiondate in dc.DataCenter.dataSet.actionDates:
            logging.info("ICIR Model Fitting:" + str(actiondate))

            X_test = selectedTestDatas.loc[actiondate][self.selectedFactorNames].fillna(0)
            minDate = dc.DataCenter.getPeriodStartDate(maxPeriod * 22, actiondate)
            selectedTrainedIC = selectedDataIC.loc[minDate:actiondate]
            selectedDataICIR = selectedTrainedIC.mean() / selectedTrainedIC.std()
            Y_pred = (X_test * selectedDataICIR).sum(axis=1)
            # Y_pred=(Y_pred-Y_pred.mean())/Y_pred.std()
            codes = X_test.index.values
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

        df = baseF.getFactorDataSet()
        startDate = self.factorTradingDates.min()
        endDate = self.factorTradingDates.max()
        dateTable = dc.DataCenter.dataSet.tradingDateTable;

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


            if nextDatePos >= dc.DataCenter.dataSet.tradingDateDic.max():
                trainedData["endDate"]=np.nan
                trainedData["StockReturn"] = None
            else:
                nextDate = dc.DataCenter.dataSet.tradingDates.iat[nextDatePos]
                closePrices1 =self.getNeedFactor("收盘价").getColletionValues(codes,
                                                                                                                  date)
                closePrices2 =self.getNeedFactor("收盘价").getColletionValues(codes,
                                                                                                                  nextDate)
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
                    trainedData[["StockReturn"]] = trainedData[["StockReturn"]].apply(
                        lambda x: ToolFactor.rankNorm(x, 0, 1, 2))
                else:
                    pass
                trainedData["endDate"] =nextDate

            if self.paraList["分组训练"].value != "无":
                trainedData['训练分组'] = self.getNeedFactor("训练分组").getColletionValues(codes,
                                                                       date)

            # 剔除停牌
            if self.paraList["剔除停牌"].value == "是":
                trainedData["TurnoverValue"] = self.getNeedFactor("成交量").getColletionValues(
                    codes, date)
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
