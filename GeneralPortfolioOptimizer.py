
from Optimizer.ParentOptimizer import ParentOptimizer
import Bean.Parameter as pr
from Bean.Portfolio import  Portfolio
import pandas as pd
import numpy as np
from Factor.Stock.IndexWeight import IndexWeight
from Factor.Stock.Sector import Sector
from DataCenter import  DataCenter
from Factor.Stock.Quote import Quote
from cvxpy import Variable, Parameter, Maximize, Problem, sum_squares, norm, quad_form, \
    abs, vstack
import cvxpy as cp
from Factor.Stock.SpecialTrade import MoveLimit
from Factor.Stock.TurnOver import TurnOver
import datetime
import time
import logging
from Bean.Tool import ToolFactor
from Factor.Risk.BarraRiskMatrix import BarraRiskMatrix
from Factor.Risk.BarraRiskMatrix import PCRiskMatrix
class GeneralPortfolioOptimizer(ParentOptimizer):
    def __init__(self):
        super().__init__()
        self.name="通用优化器"
        self.paraList ={}

        p1 = pr.Parameter("基准", "沪深300")
        p1.enumList.append("中证500")
        p1.enumList.append("沪深300")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("风险模型", "Barra")
        p1.enumList.append("Barra")

        self.paraList[p1.name] = p1


        #买入成交上限不超过一定比例
        p1 = pr.Parameter("组合规模", 100000000)
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("日均成交上限", 0)
        self.paraList[p1.name] = p1


        p1 = pr.Parameter("行业配置", "中信一级")
        p1.enumList.append("中信一级")
        p1.enumList.append("申万一级")
        p1.type="MultiValue"
        self.paraList[p1.name] = p1
        # p1.multiSelected["中信一级-银行"] = "b-0.01,b+0.01"


        p1 = pr.Parameter("风险配置", "Style")
        p1.type="MultiValue"
        self.paraList[p1.name] = p1
        # p1.multiSelected["Size"]="-0.1,+0.1"

        #因子的偏离幅度
        p1 = pr.Parameter("因子配置", "")
        p1.type="MultiFactor"
        self.paraList[p1.name] = p1
        # p1.multiSelected[Size]="-0.1,+0.1"

        #调用的优化器
        p1 = pr.Parameter("优化器", "ECOS")
        p1.enumList.append("SCS")
        p1.enumList.append("ECOS")
        self.paraList[p1.name] = p1


        #风险矩阵：选择是用barra还是主成分风险模型
        p1 = pr.Parameter("风险矩阵", "Barra")
        p1.enumList.append("主成分")
        self.paraList[p1.name] = p1


    def init(self):



      indexweightFactor=IndexWeight()
      indexweightFactor.setParaValue("名称",self.getParaValue("基准"))
      self.addNeedFactor("基准权重",indexweightFactor)

      rm=BarraRiskMatrix()
      rm.paraList["模型名"].value=self.paraList["风险模型"].value
      rm.setParaValue("是否计算","否")
      self.addNeedFactor("风险模型", rm)

      rm = PCRiskMatrix()    # 这个模型的数据加载会比较久
      self.addNeedFactor("主成分风险模型", rm)

    def optimize(self,tradingDate,candidatePortfolio,holdingPortfolio=None):

        logging.info( "Optimizing: " + str(tradingDate))

        #region 基础数据准备
        selectedPortfolio=Portfolio()
        selectedPortfolio.startDate=candidatePortfolio.startDate
        selectedPortfolio.endDate=candidatePortfolio.endDate
        benchMarkWeightDF =self.getNeedFactor("基准权重").getIndexWeightValues(
            tradingDate)
        benchMarkWeightDF.name = "benchWeight"
        candidateStocks = candidatePortfolio.tradingStocks
        candidateStocks["candidate"] = 1  # 表示是备选股票池股票
        candidateStocks=pd.concat([candidateStocks, benchMarkWeightDF], join='outer',sort=False, axis=1) #最后的优化股票池范围包括备选股票池和指数权重池




        if holdingPortfolio is not None:
            holdWeight = holdingPortfolio.tradingStocks["weight"]
            holdWeight=holdWeight/holdWeight.sum()
            holdWeight.name="holdedWeight"
            candidateStocks=pd.concat([candidateStocks, holdWeight], join='outer',sort=True, axis=1)
        else:
            candidateStocks["holdedWeight"] = 0.0
        candidateStocks["holdedWeight"]=candidateStocks["holdedWeight"].fillna(0)
        candidateStocks["benchWeight"] = candidateStocks["benchWeight"].fillna(0)
        candidateStocks["score"] = candidateStocks["score"].fillna(0)  # 没有预期收益股票打0分
        benchMarkWeightDF=candidateStocks["benchWeight"]
        #endregion


        #region 主要参数获取处理
        te = float(self.paraList["跟踪误差"].value)
        styleTE = float(self.paraList["风格风险"].value)
        styleTE = None if styleTE <= 0.0001 else styleTE  # 如果太小就不用计算
        sectorTE = float(self.paraList["行业风险"].value)
        sectorTE = None if sectorTE <= 0.0001 else sectorTE  # 如果太小就不用计算
        turnOvePpunish = float(self.paraList["换手惩罚"].value)
        turnOverMax = float(self.paraList["最大换手率"].value)
        minCutWeight = float(self.paraList["最小截取权重"].value)
        benchWeightMin = float(self.paraList["成分股权重"].value)
        riskAversion = float(self.paraList["风险厌恶系数"].value)
        riskAversion = None if riskAversion <= 0.0001 else riskAversion  # 如果太小就不用计算
        turnOvePpunish = None if turnOvePpunish <= 0.0001 else turnOvePpunish  # 换手惩罚系数
        te = None if te <= 0.0001 else te  # 跟踪误差约束
        turnOverMax = None if turnOverMax <= 0.0001 else turnOverMax  # 换手率约束
        cpct = None if benchWeightMin <= 0.0001 else benchWeightMin  # 成分股权重限制
        # endregion


        #region 目标函数设定
        stockList = candidateStocks.index.values  # 备选股票池列表
        n = len(stockList)  # 股票数目
        w0 = candidateStocks["holdedWeight"].values #持有权重
        wb = candidateStocks["benchWeight"].values  #基准权重
        r = candidateStocks["score"].values #个股预期收益
        x = Variable(n)  # 相对权重
        w = x + wb  # 绝对权重
        er = r @ x  # 组合预期收益率

        #风险模型相关
        rm = self.getNeedFactor("风险模型")
        Xf = rm.getXF(tradingDate, stockList).values  # 风险因子载荷
        n, m = Xf.shape  # n 股票数量 m 风险因子数量
        F = rm.getF(tradingDate).values  # 风险因子协方差矩
        D = rm.getD(tradingDate, stockList).values  # 股票残差协方差矩阵
        D = np.sqrt(D)
        D = D * np.sqrt(240)
        F = F * 240
        FBackUp = F  # 后续计算风格因子风险和行业因子风险的时候可以用
        F = .5 * (F + F.T)
        f = Xf.T @  x  # 风险因子暴露

        if self.paraList["风险矩阵"].value == "Barra":
            risk = quad_form(f, F) + sum_squares(cp.multiply(D, x))  # 风险(方差)
            print("正在使用Barra模型")
        elif self.paraList["风险矩阵"].value == "主成分":
            print("正在使用主成分风险模型")
            PCrm = self.getNeedFactor("主成分风险模型")
            CovM, D2 = PCrm.getCovMatrix(tradingDate, stockList)
            D2 = np.sqrt(D2) * np.sqrt(240)    # 股票特质收益波动率
            CovM = 240 * CovM    # 风险矩阵
            risk = quad_form(x, CovM) + sum_squares(cp.multiply(D2, x))
        else:
            print("错误的风险矩阵！！！")

        objfun = er  # 仅预期收益
        if isinstance(turnOvePpunish, (float, int)):
            turnOvePpunish = np.array([turnOvePpunish] * n)
        if turnOvePpunish is not None:
            "换手惩罚项"
            objfun = objfun - cp.sum(cp.multiply(turnOvePpunish, abs(w - w0)))
        if riskAversion is not None:
            "风险惩罚项"
            objfun = objfun - riskAversion @ risk
        #endregion

        #region 基础约束条件设定
        cons = []
        if te is not None:
            "跟踪误差约束"
            cons.append(risk <= te ** 2)
        if sectorTE is not None:
            "行业风险约束"
            # XfSector = Xf[:, 10:41]
            # FSector = FBackUp[10:41, 10:41]
            # FSector = .5 * (FSector + FSector.T)
            # fSector = XfSector.T * x  # 风险因子暴露
            # riskSector = quad_form(fSector, FSector)  # 风险(方差)
            # cons.append(riskSector <= sectorTE ** 2)
        if styleTE is not None:
            "风格风险约束"
            # XfStyle=Xf[:,0:9]
            # FStyle=FBackUp[0:9,0:9]
            # FStyle = .5 * (FStyle + FStyle.T)
            # fStyle = XfStyle.T * x  # 风险因子暴露
            # riskStyle = quad_form(fStyle, FStyle)   # 风险(方差)
            # cons.append(riskStyle <= styleTE ** 2)


        if turnOverMax is not None:
            if holdingPortfolio is not None:
                "换手约束"
                cons.append(norm(w - w0, 1) <= turnOverMax * 2)

        if cpct is not None:
            "成分股权重占比限制"
            cons.append((wb >= 1e-6) @  x >= (cpct - 1))

        # 组合权重之和等于1
        cons.append(cp.sum(w) <= 1 + 1e-6)
        cons.append(cp.sum(w) >= 1 - 1e-6)

        #endregion

        #region  根据停牌，涨跌停，流动性水平等决定个股最大最小权重
        stockWeightDF=self.calStockWeightLimit(candidateStocks,tradingDate)
        minWeight=stockWeightDF["minWeight"].values
        maxWeight=stockWeightDF["maxWeight"].values
        cons.append(w <= maxWeight)
        cons.append(w >= minWeight)
        #endregion

        #region 风险配置约束
        fmin =  [None] *m # 风险因子暴露下限
        fmax =  [None] *m # 风险因子暴露上限
        riskFactorList=rm.getRiskFactorList()
        # 风险模型约束
        for riskName in self.paraList["风险配置"].multiSelected:
            id=riskFactorList.index(riskName)
            valueStr=self.paraList["风险配置"].multiSelected[riskName]
            minV=float(valueStr.split(",")[0])
            maxV =float(valueStr.split(",")[1])
            fmin[id]=minV
            fmax[id] = maxV
        for i in range(m):
            "风险因子暴露约束"
            lower, uper = fmin[i], fmax[i]
            if uper is None:
                if lower is not None:
                    cons.append(Xf[:, i] @ x >= lower)
            else:
                if lower is None:
                    cons.append(Xf[:, i] @ x <= uper)
                elif uper == lower:
                    cons.append(Xf[:, i] @ x == uper)
                else:
                    cons.append(Xf[:, i] @ x <= uper)
                    cons.append(Xf[:, i] @ x >= lower)
       #endregion


        #region 行业配置约束
        sectorDummyMatrixDic={}
        sectorWeightListDic={}
        for sectorType in self.selectedSectorTypes:
            sectorFactor = self.getNeedFactor(sectorType)
            sectorCodeDF = sectorFactor.getColletionValues(stockList, tradingDate)
            sectorCodeDF.name = "sectorName"
            indexWeightSectorDF = pd.concat([sectorCodeDF, benchMarkWeightDF], axis=1, join='inner',sort=False)
            sectorWeightList = indexWeightSectorDF.groupby('sectorName')["benchWeight"].agg('sum')
            sectorWeightList=sectorWeightList/sectorWeightList.sum()
            sectorDummyMatrix = pd.get_dummies(sectorCodeDF)
            sectorDummyMatrixDic[sectorType]=sectorDummyMatrix
            sectorWeightListDic[sectorType]=sectorWeightList
        for sectorfullName in self.paraList["行业配置"].multiSelected:
            sectorType=sectorfullName.split('-')[0]
            sectorName=sectorfullName.split('-')[1]
            minStr=self.paraList["行业配置"].multiSelected[sectorfullName].split(",")[0]
            maxStr = self.paraList["行业配置"].multiSelected[sectorfullName].split(",")[1]
            sectorWeightList=sectorWeightListDic[sectorType]
            sectorDummyMatrix=sectorDummyMatrixDic[sectorType]
            sectorValue = 0
            if sectorName not in sectorDummyMatrix:
                continue
            if sectorName in sectorWeightList:
                sectorValue = sectorWeightList[sectorName]
            minSectorValue = self.caculateRelativeValue(minStr, sectorValue)
            maxSectorValue = self.caculateRelativeValue(maxStr, sectorValue)
            sectorWeightSum = sectorDummyMatrix[sectorName].values @  w
            cons.append(cp.sum(sectorWeightSum) <= maxSectorValue)
            cons.append(cp.sum(sectorWeightSum) >= minSectorValue)

        #endregion

        #region 因子配置
        for f, fChars in self.paraList["因子配置"].multiSelected.items():
            charcsList = fChars.split(",")
            lower=float(charcsList[0])
            uper=float(charcsList[1])
            f=self.getNeedFactor(f.getFactorName())
            fValues=f.getColletionValues(stockList,tradingDate)
            fValues=ToolFactor.zscore(fValues).fillna(0).values
            cons.append(fValues @  x >= lower)
            cons.append(fValues @  x <= uper)


        #endregion

        #region 求解最优组合
        prob = Problem(Maximize(objfun), constraints=cons)
        try:
            argskw = {'max_iters': 5000, 'feastol': 1e-5, 'abstol': 1e-5}
            prob.solve(solver='ECOS', **argskw)
            if prob.status in ('optimal', 'optimal_inaccurate'):
                wval = np.array(w.value)  # 绝对权重
                xval = np.array(x.value)  # 相对权重
            else:
                "优化失败"
                raise ValueError('优化失败')

            # 输出结果
            po = {
                'abswt': wval,
                'relwt': xval,
                'ns': int((wval > 1e-6).sum()),
                'er': er.value,
                # 'sigma': np.sqrt(risk.value),
                'delta': np.abs(wval - w0).sum() / 2,
                'status': prob.status
            }

            selectedStocks = pd.DataFrame(po['abswt'], candidateStocks.index, columns=["weight"])
            selectedStocks = selectedStocks[selectedStocks["weight"] > minCutWeight]
            wSum = selectedStocks.sum()
            selectedStocks = selectedStocks / wSum
            selectedStocks["startDate"] = tradingDate
            selectedStocks = pd.concat([selectedStocks, candidatePortfolio.tradingStocks["score"]], axis=1,
                                       join='inner')

            selectedPortfolio.tradingStocks = selectedStocks
            selectedPortfolio.paraList["预期收益"] = po['er']
            # selectedPortfolio.paraList["预期跟踪误差"] = po['sigma']
        except:
            logging.warning("无解，维持上期持仓"+str(tradingDate))
            selectedStocks = holdingPortfolio.tradingStocks["weight"].to_frame()
            wSum = selectedStocks.sum()
            selectedStocks = selectedStocks / wSum
            selectedStocks["startDate"] = tradingDate
            selectedStocks = pd.concat([selectedStocks, candidatePortfolio.tradingStocks["score"]], axis=1,
                                       join='inner')
            selectedPortfolio.tradingStocks = selectedStocks
            selectedPortfolio.paraList["预期收益"] = 0
            # selectedPortfolio.paraList["预期跟踪误差"] = 0
        #endregion


        return selectedPortfolio


