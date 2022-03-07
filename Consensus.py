
from Bean import Parameter as pr
import pandas as pd
import DataCenter as dc
import Factor.Stock.ParentStockFactor as psf
import numpy as np
from Factor.Stock.Quote import Quote
from statsmodels import regression
from Factor.Stock.Scale import Share
from Factor.Stock.FinReport import FinReport
import itertools
import datetime
from Factor.Stock.FinReport import ProfitNotice
import logging
from Bean.Tool import ToolFactor
class Consensus(psf.ParentStockFactor):
    selectedKey = []

    def __init__(self):
        super().__init__()
        self.name = "利润预测"
        p1 = pr.Parameter("类型", "当年")
        p1.enumList.append("当年")
        p1.enumList.append("去年")
        p1.enumList.append("明年")
        p1.enumList.append("后年")
        p1.enumList.append("时间加权")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("最少机构", 1)
        p1.type = "Float"
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("期限", 120)
        p1.type = "Float"
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("来源", "ZYYX")
        p1.enumList.append("ZYYX")
        p1.enumList.append("WIND")
        p1.enumList.append("JYDB")
        self.paraList[p1.name] = p1


        p1 = pr.Parameter("计算", "均值")
        p1.enumList.append("中位数")
        self.paraList[p1.name] = p1




        self.valueType = None
        self.minInst = 1
        self.validPeriod = 120
        self.source = None
        self.calType=None

    def init(self):
        super().init()
        self.validPeriod = self.paraList["期限"].value

        self.valueType = self.paraList["类型"].value


        self.minInst = self.paraList["最少机构"].value
        self.calType = self.getParaValue("计算")
        self.source = self.paraList["来源"].value
        self.dataSetName = self.name + self.source
    def loadData(self):
        # 先要记载一致预期基础表格
        if self.dataSetName not in dc.DataCenter.dataSet.factorDataSetDic:

            startDate = str(self.factorTradingDates.min())
            endDate = str(self.factorTradingDates.max())
            df = None

            #盈利预测来源：1表示个股，2表示行业
            if self.source == 'ZYYX':
                sqlString = "SELECT  stock_code as code ,organ_id,convert(int,(CONVERT(varchar(100), create_date,112))) create_date,report_year ,forecast_np*10000 as forecast_np,reliability,report_type FROM [zyyx_new].[dbo].rpt_forecast_stk where (reliability>0 or reliability is null) and report_quarter=4  and  [create_date] between '" + startDate + "' and '" + endDate + "'  "
                conn = dc.DataCenter.getConInstance("ZYYX")
                df = pd.read_sql(sqlString, con=conn)
                df = df.drop_duplicates(subset=["code", "report_year", "organ_id", "create_date"])
                df = df.dropna(subset=["forecast_np"])
                df["source"]=1
                df.loc[df.report_type.isin([21,98]), 'source'] = 2
            elif self.source == 'JYDB':
                sqlString = """ SELECT 
                      sm.SecuCode as code
                      ,[ForecastYear] as report_year
                      ,pf.[OrgCode] as organ_id
                      ,rr.ReportType as report_type
                	  ,rr.ResearchDepth
                	  ,convert(int,(CONVERT(varchar(100), pf.[WritingDate],112))) create_date
                	 , case when [PNetProfit] is null then [EPS]*ss.TotalShares  else [PNetProfit]*10000 end as forecast_np
                  FROM [JYDB].[dbo].[C_RR_ProfitForecast] as pf,[JYDB].[dbo].LC_ShareStru as ss,[JYDB].[dbo].SecuMain as sm, C_RR_ResearchReport as rr where pf.OID=rr.ID and  IfForecast=1 and year(pf.WritingDate)>=([ForecastYear]-4) and ss.CompanyCode=sm.CompanyCode and sm.InnerCode=pf.InnerCode and sm.SecuCategory=1 and sm.SecuMarket in (83,90) and ss.EndDate=((SELECT TOP 1 EndDate FROM Jydb.dbo.LC_ShareStru WHERE CompanyCode=sm.CompanyCode AND EndDate<=pf.WritingDate ORDER BY EndDate DESC))"""
                sqlString += " and pf.[WritingDate] between '" + startDate + "' and '" + endDate + "'  "
                conn = dc.DataCenter.getConInstance("JYDB")
                df = pd.read_sql(sqlString, con=conn)
                df = df.drop_duplicates(subset=["code", "report_year", "organ_id", "create_date"])
                df = df.dropna(subset=["forecast_np"])
                df["source"] = 2
                df.loc[df.report_type.isin([6, 8]), 'source'] = 1

            df = df.set_index(["report_year", "create_date"])
            df = df.sort_index()
            self.addFactorDataSet(df)

    def getReportYearConsensus(self,df,reportYear, date):
        startDate = dc.DataCenter.getPeriodStartDate(self.validPeriod, date)
        if reportYear not in df.index:
            return pd.Series()
        currentDF= df.loc[reportYear].loc[startDate:date]
        df1=currentDF[currentDF["source"]==1]
        result1 = df1.groupby(["code", "organ_id"])[
            'forecast_np'].agg('last')
        if self.calType == "均值":
            result1 = result1.groupby("code").mean()
        elif self.calType == "中位数":
            result1 = result1.groupby("code").median()

        result2 = currentDF.groupby(["code", "organ_id"])[
            'forecast_np'].agg('last')
        instNum = result2.groupby("code").count()
        instNum = instNum[instNum >= self.minInst]

        result1 = result1.reindex(instNum.index.values)

        if self.calType == "均值":
            result2 = result2.groupby("code").mean()
        elif self.calType == "中位数":
            result2 = result2.groupby("code").median()

        result=result1.fillna(result2)

        return result


    def getColletionValues(self, codeList, date):

        df = self.getFactorDataSet()

        if self.valueType == "时间加权":
            currentYear = int(date // 10000)
            currentMonth = float(date // 100 % 100)

            nextYear=currentYear+1
            result1=self.getReportYearConsensus(df,currentYear,date)
            result1.name="y1"
            result2 = self.getReportYearConsensus(df, nextYear, date)
            result2.name = "y2"
            result = pd.concat([result1, result2], axis=1, sort=False)
            result["加权"] = result.apply(
                lambda x: (None if np.isnan(x["y2"]) else x["y2"]) if np.isnan(x["y1"]) else (
                    x["y1"] if np.isnan(x["y2"]) else (x["y1"] * (12 - currentMonth) + x[
                        "y2"] * currentMonth) / 12.0), axis=1)
            result = result["加权"]
            result = result if codeList is None else result.reindex(codeList)
            result.name = self.fullName
            return result

        if self.valueType == "明年":
            reportYear = int(date // 10000)+1

        elif self.valueType == "当年":
            reportYear = int(date // 10000)

        elif self.valueType == "后年":
            reportYear = int(date // 10000)+2

        elif self.valueType == "去年":
            reportYear = int(date // 10000)-1
        result=self.getReportYearConsensus(df,reportYear,date)
        result=result.reindex(codeList) if codeList is not None else result
        result.name=self.fullName
        return result
    #获得特定年份收益
    def getYearValues(self, codeList, date,year):
        pf3 = self.getFactorDataSet()
        result=self.getReportYearConsensus(pf3,year,date)
        result=result.reindex(codeList) if codeList is not None else result
        result.name=self.fullName
        return result


class ConsensusQuarter(psf.ParentStockFactor):
    selectedKey = []

    def __init__(self):
        super().__init__()
        self.name = "利润预测季度"

        p1 = pr.Parameter("类型", "净利润")
        p1.enumList.append("ROE")
        p1.enumList.append("DROE")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("最少机构", 1)
        p1.type = "Float"
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("期限", 120)
        p1.type = "Float"
        self.paraList[p1.name] = p1



        p1 = pr.Parameter("来源", "ZYYX")
        p1.enumList.append("ZYYX")
        p1.enumList.append("WIND")
        self.paraList[p1.name] = p1







        self.valueType = None
        self.minInst = 1
        self.validPeriod = 120
        self.source = None

        self.quarterPCTName=None

    def init(self):
        super().init()
        self.validPeriod = self.paraList["期限"].value

        self.valueType = self.paraList["类型"].value
        self.minInst = self.paraList["最少机构"].value

        self.source = self.paraList["来源"].value
        self.dataSetName = self.name + self.source

        consensus=Consensus()
        consensus.setParaValue("最少机构",self.getParaValue("最少机构"))
        consensus.setParaValue("来源", self.getParaValue("来源"))
        consensus.setParaValue("期限", self.getParaValue("期限"))
        self.addNeedFactor("consensus", consensus)

        isq = ProfitNotice()
        isq.paraList["类型"].value = "Quarter"
        self.addNeedFactor("业绩预告Quarter",isq)



        isq = FinReport()
        isq.paraList["指标"].value ="NPParentCompanyOwners"
        isq.paraList["类型"].value = "Quarter"
        self.addNeedFactor("NPParentCompanyOwnersQuarter", isq)

        isq = FinReport()
        isq.paraList["指标"].value ="NPParentCompanyOwners"
        isq.paraList["类型"].value = "All"
        self.addNeedFactor("NPParentCompanyOwnersAll", isq)

        if self.valueType in ["ROE","DROE"]:
            isq = FinReport()
            isq.paraList["指标"].value = "TotalShareholderEquity"
            isq.paraList["类型"].value = "All"
            self.addNeedFactor("TotalShareholderEquity", isq)



    def getColletionValues(self, codeList, date):

        dfReport=self.getNeedFactor("NPParentCompanyOwnersQuarter").getFactorDataSet()
        df=dfReport.reset_index()
        startDate = dc.DataCenter.getPeriodStartDate(120, date)
        lastMonthReport= df[ (df["announceDate"] <=date) & (df["announceDate"]>startDate ) & (df["reportDate"]>startDate )].groupby('code')['reportDate'].agg('last')

        consensusF=self.getNeedFactor("consensus")
        NPAllDF=self.getNeedFactor("NPParentCompanyOwnersAll")
        NPQuarterDF = self.getNeedFactor("NPParentCompanyOwnersQuarter")


        lastMonthReportGroup=lastMonthReport.to_frame().groupby("reportDate")

        def getNextQuarter(currentReportDate, codes):
            currentYear = int(currentReportDate // 10000)
            currentMonthDate = int(currentReportDate % 10000)
            consensusYear = currentYear + 1 if currentMonthDate == 1231 else currentYear
            currentConcusValues = consensusF.getYearValues(codes, date, consensusYear)
            currentReportValues = NPAllDF.getReportValue(currentReportDate, codes)
            if currentMonthDate == 1231:
                nextReportValues = currentConcusValues
            else:
                nextReportValues = currentConcusValues - currentReportValues

            if currentMonthDate == 1231:
                nextReportDate = ((currentYear + 1) * 10000 + 331)
            elif currentMonthDate == 331:
                nextReportDate = (currentYear * 10000 + 630)
            elif currentMonthDate == 630:
                nextReportDate = (currentYear * 10000 + 930)
            else:
                nextReportDate = (currentYear * 10000 + 1231)

            # dfQ = dfReport["NPParentCompanyOwnersQuarter"]
            # dfQ = dfQ.reset_index()
            # dfQ["RYear"] = dfQ["reportDate"] // 10000
            # dfQ["RMonth"]=  dfQ["reportDate"] // 100 % 100
            # T_4Year=nextReportDate-4*10000
            # currentMonth= int(currentReportDate // 100 % 100)
            # dfQ=dfQ[(dfQ["reportDate"]>=T_4Year) & (dfQ["reportDate"]<currentReportDate)]
            # dfQ=dfQ[(dfQ["RMonth"] > currentMonth)]
            # dfQ["PCT"] = dfQ.groupby(["code","RYear"],group_keys=False).apply(lambda x: x["NPParentCompanyOwnersQuarter"]/np.abs(x["NPParentCompanyOwnersQuarter"].sum()))
            # nextMonth=nextReportDate // 100 % 100
            # QPCT=dfQ[dfQ["RMonth"]==nextMonth].groupby("code")["PCT"].median()

            # if currentMonthDate == 1231:
            #     QPCT[QPCT>0.5]=0.5
            #     QPCT[QPCT <0]=0.25
            # elif currentMonthDate == 331:
            #     QPCT[QPCT>0.75]=0.75
            #     QPCT[QPCT <0]=0.33
            # elif currentMonthDate == 630:
            #     QPCT[QPCT > 0.8] = 0.8
            #     QPCT[QPCT < 0] = 0.5
            # else:
            #     QPCT[QPCT>-89999]=1
            if currentMonthDate == 1231:
                QPCT = 0.25
            elif currentMonthDate == 331:

                QPCT = 0.33
            elif currentMonthDate == 630:
                QPCT = 0.5
            else:
                QPCT = 1
            nextReportValues = nextReportValues * QPCT
            return nextReportValues
        result=None
        if self.valueType=="净利润":

            for reportDate, groupData in lastMonthReportGroup:
                trainedGroupData = lastMonthReportGroup.get_group(reportDate)
                reportGroupValues= getNextQuarter(reportDate,trainedGroupData.index.values)
                result=reportGroupValues   if result is None else result.append(reportGroupValues)
        if self.valueType == "ROE":
            SEF = self.getNeedFactor("TotalShareholderEquity")
            for reportDate, groupData in lastMonthReportGroup:
                trainedGroupData = lastMonthReportGroup.get_group(reportDate)
                reportGroupValues = getNextQuarter(reportDate, trainedGroupData.index.values)
                result = reportGroupValues if result is None else result.append(reportGroupValues)
            seValues=SEF.getColletionValues(codeList, date)
            result=result/seValues
        if self.valueType == "DROE":
            SEF = self.getNeedFactor("TotalShareholderEquity")

            def getNextQuarterDROE(currentReportDate, codes):

                currentYear = int(currentReportDate // 10000)

                currentMonthDate = int(currentReportDate % 10000)
                consensusYear = currentYear + 1 if currentMonthDate == 1231 else currentYear
                currentConcusValues = consensusF.getYearValues(codes, date, consensusYear)
                currentReportValues = NPAllDF.getReportValue(currentReportDate, codes)
                if currentMonthDate == 1231:
                    nextReportValues = currentConcusValues
                else:
                    nextReportValues = currentConcusValues - currentReportValues

                if currentMonthDate == 1231:
                    nextReportDate = ((currentYear + 1) * 10000 + 331)
                elif currentMonthDate == 331:
                    nextReportDate = (currentYear * 10000 + 630)
                elif currentMonthDate == 630:
                    nextReportDate = (currentYear * 10000 + 930)
                else:
                    nextReportDate = (currentYear * 10000 + 1231)

                # dfQ = dfReport["NPParentCompanyOwnersQuarter"]
                # dfQ = dfQ.reset_index()
                # dfQ["RYear"] = dfQ["reportDate"] // 10000
                # dfQ["RMonth"]=  dfQ["reportDate"] // 100 % 100
                # T_4Year=nextReportDate-4*10000
                # currentMonth= int(currentReportDate // 100 % 100)
                # dfQ=dfQ[(dfQ["reportDate"]>=T_4Year) & (dfQ["reportDate"]<currentReportDate)]
                # dfQ=dfQ[(dfQ["RMonth"] > currentMonth)]
                # dfQ["PCT"] = dfQ.groupby(["code","RYear"],group_keys=False).apply(lambda x: x["NPParentCompanyOwnersQuarter"]/np.abs(x["NPParentCompanyOwnersQuarter"].sum()))
                # nextMonth=nextReportDate // 100 % 100
                # QPCT=dfQ[dfQ["RMonth"]==nextMonth].groupby("code")["PCT"].median()

                # if currentMonthDate == 1231:
                #     QPCT[QPCT>0.5]=0.5
                #     QPCT[QPCT <0]=0.25
                # elif currentMonthDate == 331:
                #     QPCT[QPCT>0.75]=0.75
                #     QPCT[QPCT <0]=0.33
                # elif currentMonthDate == 630:
                #     QPCT[QPCT > 0.8] = 0.8
                #     QPCT[QPCT < 0] = 0.5
                # else:
                #     QPCT[QPCT>-89999]=1
                if currentMonthDate == 1231:
                    QPCT = 0.25

                elif currentMonthDate == 331:

                    QPCT = 0.33
                elif currentMonthDate == 630:
                    QPCT = 0.5
                else:
                    QPCT = 1
                nextReportValues = nextReportValues * QPCT
                return nextReportValues

            for reportDate, groupData in lastMonthReportGroup:
                trainedGroupData = lastMonthReportGroup.get_group(reportDate)
                reportGroupValues = getNextQuarter(reportDate, trainedGroupData.index.values)
                result = reportGroupValues if result is None else result.append(reportGroupValues)
            seValues = SEF.getColletionValues(codeList, date)
            result = result / seValues
        result=result.reindex(codeList)
        result.name=self.fullName
        return result


class ConsensusChange(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "一致预期变化"
        p1 = pr.Parameter("区间",60)
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("类型", "增幅")
        p1.enumList.append("比净资产")
        self.paraList[p1.name] = p1
    def init(self):
        q1 = Consensus()
        q1.setParaValue("期限",120)
        self.addNeedFactor("Consensus",q1)

        if self.paraList["类型"].value == "比净资产":
            ist = FinReport()
            ist.paraList["指标"].value = "TotalShareholderEquity"
            ist.paraList["类型"].value = "All"
            self.addNeedFactor("净资产", ist)



    def getColletionValues(self, codeList, date):

        period = self.paraList["区间"].value
        startDate = dc.DataCenter.getPeriodStartDate(period, date)
        reportYear=int(date // 10000)
        conF=self.getNeedFactor("Consensus")
        startValues = conF.getYearValues( codeList, startDate,reportYear)
        endValues =conF.getYearValues( codeList, date,reportYear)

        if self.paraList["类型"].value == "增幅":
            result = (endValues - startValues) / startValues
        elif self.paraList["类型"].value == "比净资产":

            nAssetF = self.getNeedFactor("净资产")
            nAssetDf = nAssetF.getColletionValues(codeList, date)
            result = (endValues - startValues) / nAssetDf
        result=result.replace({float('inf'): np.nan, float('-inf'): np.nan})
        result.name = self.fullName
        return result

#一致预期12个月标准化变化
class ConsensusSG(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "预期标准化变化"
        p1 = pr.Parameter("类型", "当年")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("周期", 240)  #多少天
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("期限", 120)
        p1.type = "Float"
        self.paraList[p1.name] = p1
        self.valueName=None

    def init(self):
        super().init()

        self.valueName=self.paraList["类型"].value
        f=Consensus()
        f.setParaValue("类型",self.getParaValue("类型"))
        f.setParaValue("期限", self.getParaValue("期限"))
        self.addNeedFactor("一致预期",f)


    def getColletionValues(self, codeList, date):
        dateTable = dc.DataCenter.dataSet.tradingDateTable;
        period = self.paraList["周期"].value
        startDate=dc.DataCenter.getPeriodStartDate(period,date)
        endDate =date
        tradingPeriodDates = dateTable[(dateTable['IfMonthEnd'] == 1) & (dateTable['TradingDate'] >= startDate)& (dateTable['TradingDate'] <=endDate) & (dateTable['TradingDate'] >= 20050201)][
            "TradingDate"]
        conF=self.getNeedFactor("一致预期")
        currentYear = int(date // 10000)
        df=tradingPeriodDates.apply(lambda x:conF.getYearValues(codeList, x,currentYear))
        valueMean = df.mean()
        valueStd = df.std()
        currentValue = conF.getYearValues(codeList, date,currentYear)
        result = (currentValue - valueMean) / valueStd
        result = result.reindex(codeList)
        result.name = self.fullName
        return result


class Rating(psf.ParentStockFactor):


    def __init__(self):
        super().__init__()
        self.name = "评级"

        p1 = pr.Parameter("最少机构", 1)
        p1.type = "Float"
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("期限", 90)
        p1.type = "Float"
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("来源", "ZYYX")
        p1.enumList.append("ZYYX")
        p1.enumList.append("JYDB")
        self.paraList[p1.name] = p1

        self.paraList[p1.name] = p1
        self.keyName = None
        self.needFactorNameDic = {}

        self.storedKey = None
        self.limitPeriod = 30

        self.minInst = 1

        self.source = None

    def init(self):
        super().init()
        self.limitPeriod = self.paraList["期限"].value
        self.minInst = self.paraList["最少机构"].value
        self.source = self.paraList["来源"].value
        self.dataSetName = self.name + self.source



    def loadData(self):
        # 先要记载一致预期基础表格
        if self.dataSetName not in dc.DataCenter.dataSet.factorDataSetDic:

            startDate = str(self.factorTradingDates.min())
            endDate = str(self.factorTradingDates.max())
            df = None
            #source 1表示个股报告，2表示行业表格，一般1可信度高
            if self.source == 'ZYYX':
                sqlString = "select  [stock_code] as code ,report_type ,[organ_id],convert(int,(CONVERT(varchar(100), current_create_date,112)))  as create_date,current_gg_rating FROM [zyyx_new].[dbo].[rpt_rating_adjust] where current_gg_rating>0  and  [current_create_date] between '" + startDate + "' and '" + endDate + "'  "
                conn = dc.DataCenter.getConInstance("ZYYX")
                df = pd.read_sql(sqlString, con=conn)
                df["source"] =2
                df.loc[df.report_type.isin([22, 23,24,25]), 'source'] =1
            elif self.source == "JYDB":
                sqlString = """SELECT sm.SecuCode as code
                  ,gpr.OrgCode as organ_id
            	  ,convert(int,(CONVERT(varchar(100), gpr.[WritingDate] ,112))) create_date
                  ,[CurrentRating]  as current_gg_rating
                  ,rr.ReportType as report_type
            	  ,rr.ResearchDepth
              FROM [JYDB].[dbo].[C_RR_GoalPriceRate] as gpr, C_RR_ResearchReport as rr,[JYDB].[dbo].SecuMain as sm 
              where gpr.OID=rr.ID and gpr.InnerCode=sm.InnerCode """
                sqlString += " and gpr.[WritingDate] between '" + startDate + "' and '" + endDate + "'  "
                conn = dc.DataCenter.getConInstance("JYDB")
                df = pd.read_sql(sqlString, con=conn)
                df["source"] = 2
                df.loc[df.report_type.isin([6, 8]), 'source'] = 1
            df = df.set_index(["create_date"])
            df["current_gg_rating"] = pd.to_numeric(df["current_gg_rating"], errors='coerce', downcast='float')
            df = df.sort_index()
            self.addFactorDataSet(df,self.dataSetName)

    def getColletionValues(self, codeList, date):
        result = None
        pf3 = self.getFactorDataSet()
        pf3 = pf3[pf3["source"] == 1]
        startDate = dc.DataCenter.getPeriodStartDate(self.limitPeriod, date)
        result = pf3.loc[startDate:date].groupby(["code", "organ_id"])[
            'current_gg_rating'].agg('last')
        instNum = result.groupby("code").count()
        instNum = instNum[instNum >= self.minInst]
        result = result.groupby("code").mean()
        result = result.reindex(instNum.index.values)
        result = result if codeList is None else result.reindex(codeList)
        result.name = self.fullName
        return result


class RatingChange(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "评级变化"
        p1 = pr.Parameter("区间",60)
        self.paraList[p1.name] = p1
    def init(self):
        q1 = Rating()
        q1.setParaValue("期限",120)
        self.addNeedFactor("Rating",q1)



    def getColletionValues(self, codeList, date):
        period = self.paraList["区间"].value
        startDate = dc.DataCenter.getPeriodStartDate(period, date)
        startValues = self.getNeedFactor("Rating").getColletionValues(codeList,
                                                                                                          startDate)
        endValues = self.getNeedFactor("Rating").getColletionValues(codeList,
                                                                                                        date)
        result = (endValues - startValues)
        result.name = self.fullName
        return result


#参考天风做法，先算个股然后再求平均
class NPForecasetChange(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "业绩上调幅度"  #


        p1 = pr.Parameter("区间", 60)
        self.paraList[p1.name] = p1
        p1 = pr.Parameter("类型", "增幅")
        p1.enumList.append("比净资产")
        self.paraList[p1.name] = p1

        self.storedKey = None

    def init(self):
        super().init()

        consensus=Consensus()
        self.addNeedFactor("NPForecaset",consensus)
        self.storedKey=self.name
        if self.paraList["类型"].value=="比净资产":
            ist = FinReport()
            ist.paraList["指标"].value = "TotalShareholderEquity"
            ist.paraList["类型"].value = "All"
            self.addNeedFactor("净资产",ist)


    def loadData(self):
        #预先把所有上调幅度算好
        if self.storedKey in dc.DataCenter.dataSet.factorDataSetDic:
            return
        NPForecasetF = self.getNeedFactor("NPForecaset")
        startDate=dc.DataCenter.dataSet.actionDates.min()
        startDate=dc.DataCenter.getPeriodStartDate(120,startDate)
        df = NPForecasetF.getFactorDataSet()
        df = df[df["source"] == 1]
        df=df.reset_index()
        df = df[df["create_date"] > startDate]
        df["cStartDate"]=(df["report_year"]-1)*10000+630
        df["cEndDate"] = (df["report_year"]) * 10000 + 1231
        df=df[(df["create_date"]>df["cStartDate"])&(df["create_date"]<=df["cEndDate"])]
        df=df.sort_values(by=["report_year","code","organ_id","create_date"])
        dfGroup=df.groupby(["report_year","code", "organ_id"], as_index=False,
                                                           group_keys=False)
        df["forecast_npLast"] = dfGroup.apply(lambda x: x['forecast_np'].shift(1))
        df["create_dateLast"] = dfGroup.apply(lambda x: x['create_date'].shift(1))
        df = df.set_index(["report_year","create_date", "code", "organ_id"]).sort_index()[["forecast_np","forecast_npLast","create_dateLast"]]
        self.addFactorDataSet(df)

    def getColletionValues(self,codeList,date):
       df=self.getFactorDataSet()
       period = self.paraList["区间"].value
       # 当区间内没有任何一家卖方调整盈利预测，那么就用2倍区间的样本
       startDate = dc.DataCenter.getPeriodStartDate(period * 2, date)
       calDate = dc.DataCenter.getPeriodStartDate(period, date)
       currentYear = int(date // 10000)
       finalResult = pd.Series()
       if currentYear in df.index:
           npFCurrent = df.loc[currentYear].loc[startDate:date]
           npFCurrent = npFCurrent[npFCurrent["create_dateLast"]> startDate]
           if self.paraList["类型"].value == "增幅":
               npFCurrent = npFCurrent[npFCurrent["forecast_npLast"] > 3000000]
               result = (npFCurrent['forecast_np'] - npFCurrent['forecast_npLast']) / npFCurrent['forecast_npLast']
               # 优先使用最近的数据Result1，如果最近的数据没有那么也可以用比较历史的数据填充
               result1 = result.loc[calDate:date].groupby(["code", "organ_id"]).agg("last")
               result1 = result1.groupby("code").mean().reindex(codeList)
               result2 = result.loc[startDate:calDate].groupby(["code", "organ_id"]).agg("last")
               if result2.shape[0]>2:
                   result2 = result2.groupby("code").mean().reindex(codeList)
                   result = result1.fillna(result2)
               else:
                   result=result1
           elif self.paraList["类型"].value == "比净资产":
               result = (npFCurrent['forecast_np'] - npFCurrent['forecast_npLast'])
               # 优先使用最近的数据Result1，如果最近的数据没有那么也可以用比较历史的数据填充
               result1 = result.loc[calDate:date].groupby(["code", "organ_id"]).agg("last")
               result1 = result1.groupby("code").mean().reindex(codeList)
               result2 = result.loc[startDate:calDate].groupby(["code", "organ_id"]).agg("last")
               if result2.shape[0] > 2:
                   result2 = result2.groupby("code").mean().reindex(codeList)
                   result = result1.fillna(result2)
               else:
                   result=result1
               nAssetF = self.getNeedFactor("净资产")
               nAssetDf = nAssetF.getColletionValues(codeList, date)
               result = result / nAssetDf
           elif self.paraList["类型"].value == "加权增幅":
               npFCurrent = npFCurrent[npFCurrent["forecast_npLast"] > 3000000]
               npFCurrent["growth"] = (npFCurrent['forecast_np'] - npFCurrent['forecast_npLast']) / npFCurrent['forecast_npLast']
               npFCurrent = npFCurrent.loc[startDate:date].groupby(["code", "organ_id"]).agg("last")
               tradingDates=dc.DataCenter.dataSet.tradingDates
               tradingDates=tradingDates[(tradingDates>=startDate) & (tradingDates<=date)]
               tw = np.arange(tradingDates.count() - 1, -1, -1)
               tw = np.power(0.5, tw / 60.0)
               tw = tw / tw.sum()
               dateWeightSeries=pd.Series(data=tw,index=tradingDates.values)
               npFCurrent["weight"]=npFCurrent["create_dateLast"].replace(dateWeightSeries)
               npFCurrent=npFCurrent[npFCurrent["weight"]<1]
               npFCurrent["growth"]= npFCurrent["growth"]* npFCurrent["weight"]
               result=npFCurrent.groupby(["code"])["growth"].mean()


           finalResult = result
       finalResult=finalResult.reindex(codeList)
       finalResult.name=self.fullName
       return finalResult

#zyyx市场评分体系
class ZYYXScore(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "ZYYXScore"



    def loadData(self):


        startDate = str(self.factorTradingDates.min())
        endDate = str(self.factorTradingDates.max())
        import pymssql
        sqlString = "SELECT   [stock_code] as code,convert(int,(CONVERT(varchar(100), [con_date],112))) as tradingDate  ,[score] as factorValue FROM [zyyx_new].[dbo].[certainty_score_stk] where  index_code=999999   and  [con_date] between '" + startDate + "' and '" + endDate + "'  "
        conn = dc.DataCenter.getConInstance("ZYYX")
        df = pd.read_sql(sqlString, con=conn)
        df=df.set_index(["tradingDate","code"]).sort_index()
        self.addFactorDataSet(df)


    def getColletionValues(self, codeList, date):
        df = self.getFactorDataSet()
        result = df.loc[:date].groupby('code')['factorValue'].agg('last')
        result = result.reindex(codeList)
        result.name = self.fullName
        return result


#zyyx市场覆盖率数据
class ConsensusCoverage(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "一致预期覆盖"


        p1 = pr.Parameter("类型", "时间加权")
        p1.enumList.append("时间加权")
        p1.enumList.append("机构数")
        p1.enumList.append("预测等级加权")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("区间",120)
        self.paraList[p1.name] = p1

        self.valueType=None

    def init(self):
        self.valueType = self.getParaValue("类型")
        self.dataSetName=self.name
        f=Consensus()
        self.addNeedFactor("盈利预测",f)

    def loadData(self):
        if self.valueType=="时间加权":
           df = self.getNeedFactor("盈利预测").getFactorDataSet()
           df=df.reset_index()
           df=df.groupby(["create_date","code"])["forecast_np"].count()
           self.addFactorDataSet(df)
        elif self.valueType=="机构数":
            df = self.getNeedFactor("盈利预测").getFactorDataSet()
            df = df.reset_index()
            df=df.drop_duplicates(["create_date","code","organ_id"])
            self.addFactorDataSet(df)
        elif self.valueType=="预测等级加权":
            df = self.getNeedFactor("盈利预测").getFactorDataSet()
            df = df.reset_index()
            df["weight"] = 1
            df.loc[df["source"]==1, 'weight'] = 2
            df = df.groupby(["create_date", "code"])["weight"].sum()
            self.addFactorDataSet(df)


    def getColletionValues(self, codeList, date):
        pf3 = self.getFactorDataSet()
        period = self.paraList["区间"].value

        startDate = dc.DataCenter.getPeriodStartDate(period, date)
        if self.valueType == "时间加权":  # 半衰期60天
            td = dc.DataCenter.dataSet.tradingDates
            td = td[(td >= startDate) & (td <= date)]
            result = pf3.loc[startDate:date]
            result = result.unstack()
            result = result.reindex(td)
            tw = np.arange(td.count() - 1, -1, -1)
            tw = np.power(0.5, tw / 60.0)
            tw = tw / tw.sum()
            result = result.multiply(tw, axis=0)
            result = result.sum(axis=0)
        elif self.valueType == "预测等级加权":  # 半衰期60天
            td = dc.DataCenter.dataSet.tradingDates
            td = td[(td >= startDate) & (td <= date)]
            result = pf3.loc[startDate:date]
            result = result.unstack()
            result = result.reindex(td)
            tw = np.arange(td.count() - 1, -1, -1)
            tw = np.power(0.5, tw / 60.0)
            tw = tw / tw.sum()
            result = result.multiply(tw, axis=0)
            result = result.sum(axis=0)
        elif self.valueType== "机构数":
            pf3=pf3[(pf3["create_date"]>=startDate) & (pf3["create_date"]<=date) ]
            result = pf3.groupby(["code","organ_id"]).agg("last")
            result=result.groupby("code")["create_date"].count()
        result = result if (codeList is None) else result.reindex(codeList)
        result = result.fillna(0)
        result.name = self.fullName
        return result

#市场覆盖率数据变化
class ConsensusCovChange(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "一致预期覆盖"


        p1 = pr.Parameter("类型", "时间加权")
        p1.enumList.append("时间加权")
        p1.enumList.append("机构数")
        p1.enumList.append("预测等级加权")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("区间",120)
        self.paraList[p1.name] = p1

        self.valueType=None

    def init(self):
        period=self.getParaValue("区间")
        cov=ConsensusCoverage()
        cov.setParaValue("类型",self.getParaValue("类型"))
        cov.setParaValue("区间", 120)
        self.addNeedFactor("coverage",cov)
        from Factor.Stock.Momentum import Momentum
        from Factor.Stock.Vol import Vol

        ret=Momentum()
        ret.setParaValue("区间",period )
        self.addNeedFactor("ret",ret)

        vol = Vol()
        vol.setParaValue("类型","VOL")
        vol.setParaValue("天数", period)
        self.addNeedFactor("vol", vol)


    def getColletionValues(self, codeList, date):
        cov = self.getNeedFactor("coverage")
        period = self.paraList["区间"].value

        startDate = dc.DataCenter.getPeriodStartDate(period, date)
        startValue=cov.getColletionValues(codeList,startDate)
        endValue=cov.getColletionValues(codeList,date)

        result=endValue.rank(pct=True)-startValue.rank(pct=True)
        startValueNonZero=startValue[startValue>0]
        endValueNonZero = endValue[endValue > 0]
        allNoneZero= list(set(startValueNonZero.index.values).union(set(endValueNonZero.index.values)))
        result=result.reindex(allNoneZero)

        retValue=self.getNeedFactor("ret").getColletionValues(allNoneZero,date)
        retValue=ToolFactor.rankNorm(retValue,0,1,2).fillna(0)

        volValue = self.getNeedFactor("vol").getColletionValues(allNoneZero, date)
        volValue = ToolFactor.rankNorm(volValue, 0, 1, 2).fillna(0)

        adjValue=pd.concat([retValue,volValue],axis=1)

        result=ToolFactor.rankNorm(result,0,1,2).fillna(0)
        result=ToolFactor.normalizeValues(result,adjValue)
        result = result if (codeList is None) else result.reindex(codeList)

        result.name = self.fullName
        return result


class ExceedNPForecaset(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "季报超预期"  # 包含业绩快报数据

        p1 = pr.Parameter("最少机构", 1)
        self.paraList[p1.name] = p1


        p1 = pr.Parameter("最近期限", 90)
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("报表", "所有")
        p1.enumList.append("正式")
        p1.enumList.append("所有")
        p1.enumList.append("预告")
        self.paraList[p1.name] = p1

        p1 = pr.Parameter("类型", "比预期")
        p1.enumList.append("比标准差")
        p1.enumList.append("比预期")
        self.paraList[p1.name] = p1

        self.keyName = None
        self.needFactorNameDic = {}
        self.selectedType = None
        self.storedKey = None

    def init(self):
        super().init()

        isq = ProfitNotice()
        isq.paraList["类型"].value = "Quarter"
        self.addNeedFactor("业绩预告Quarter",isq)



        isq = FinReport()
        isq.paraList["指标"].value ="NPParentCompanyOwners"
        isq.paraList["类型"].value = "Quarter"
        self.addNeedFactor("财报", isq)

        consensus=Consensus()
        self.addNeedFactor("NPForecaset", consensus)

        self.dataSetName=self.name+str(self.paraList["最少机构"].value)+str(self.paraList["最近期限"].value)


    def loadData(self):

        if self.dataSetName in dc.DataCenter.dataSet.factorDataSetDic:
            return
        npQuarter="NPParentCompanyOwnersQuarter"
        npQuarterNotice = "NPParentCompanyOwnersQuarter"
        # profitGrowth=ProfitNotice.addGrowthYoY(npQuarterNotice)
        # revenueGrowth = FinReport.addGrowthYoY("OperatingRevenueQuarter")
        # npSU = ProfitNotice.addSU("NPParentCompanyOwnersQuarter")
        npSTD= ProfitNotice.addSTD("NPParentCompanyOwnersQuarter",8)
        # profitGrowthReport = FinReport.addGrowthYoY(npQuarter)
        # profitSTGReport = FinReport.addSTG(npQuarter,8)
        minActionDate=dc.DataCenter.dataSet.actionDates.min()
        maxDate=dc.DataCenter.dataSet.tradingDates.max()
        minCalDate=dc.DataCenter.getPeriodStartDate(480,minActionDate)
        df=dc.DataCenter.dataSet.factorDataSetDic["业绩预告"]
        dfReport = dc.DataCenter.dataSet.factorDataSetDic["财报"]
        NPForecaset=self.getNeedFactor("NPForecaset")
        NPForecasetDF=NPForecaset.getFactorDataSet()

        minCalDate=minCalDate if minCalDate>=20040930 else 20040930
        reportDatesList = list(df.index.levels[0])
        reportDatesList = [one for one in reportDatesList if one >= minCalDate]

        profitNoticeDF=df.reset_index()
        minAnalyNum=self.paraList["最少机构"].value
        maxPeriod = self.paraList["最近期限"].value

        def getAnalystExpectationCollection(aDF):
            reportYear = aDF.index.get_level_values("report_year").values[0]
            announceDate = aDF.index.get_level_values("announceDate").values[0]
            # reportYear=aDF["report_year"].min()
            # announceDate=aDF["announceDate"].min()
            yearNPF = NPForecasetDF.loc[reportYear]
            startDate = dc.DataCenter.getPeriodStartDate(maxPeriod, announceDate)
            endDate=announceDate
            npfGroup = yearNPF.loc[startDate:endDate].groupby(["code","organ_id"])['forecast_np'].agg('last')
            npfGroupNum=npfGroup.groupby("code").count()
            npfGroupResult = npfGroup.groupby("code").mean()
            npfGroupNum=npfGroupNum[npfGroupNum>minAnalyNum]
            npfGroupResult=npfGroupResult.reindex(npfGroupNum.index.values)
            # npfGroupResult["report_year"]=reportYear
            # npfGroupResult["announceDate"] = announceDate
            # npfGroupResult=npfGroupResult.reindex(aDF["code"]).values
            aDF["npfGroupResult"]=npfGroupResult.reindex(aDF["code"]).values
            return aDF["npfGroupResult"]

        print("MinCalDate"+str(minCalDate))
        profitNoticeDF=profitNoticeDF[(profitNoticeDF["reportDate"]>minCalDate)&(profitNoticeDF["announceDate"]<=maxDate)]
        profitNoticeDF["report_year"]=profitNoticeDF.apply(lambda x:x["reportDate"]//10000,axis=1)
        profitNoticeDF=profitNoticeDF[profitNoticeDF["report_year"]>2004]
        profitNoticeDF=profitNoticeDF.set_index(["report_year","announceDate"]).sort_index()

        NPForecaset = profitNoticeDF.groupby(["report_year","announceDate"],as_index=False,group_keys=False).apply(lambda x:getAnalystExpectationCollection(x))
        profitNoticeDF["NPForecaset"]=NPForecaset


        profitNoticeDF=profitNoticeDF.reset_index()
        profitNoticeDF=profitNoticeDF.set_index(["reportDate", "code"])

        announcedReportDateList=list(dfReport.index.levels[0])
        objectiveDF = None

        for i in range(5,len(reportDatesList)):
            T=reportDatesList[i]
            T_1=reportDatesList[i-1]
            T_4 = reportDatesList[i - 4]
            T_5 = reportDatesList[i - 5]
            currentYear=T//10000
            lastYearReport=(currentYear-1)*10000+1231
            currentMonthDate = int(T % 10000)

            if T_1 not in announcedReportDateList:
                continue
            currentObjectiveDF = None

            if T not in profitNoticeDF.index:
                continue

            dfTNotice = profitNoticeDF.loc[T]
            lastYearNP = dfReport.loc[lastYearReport]["NPParentCompanyOwners"]
            lastYearNP.name = "lastYearNP"
            lastYearNP.reindex(dfTNotice.index.values)
            T_4NP = dfReport.loc[T_4]["NPParentCompanyOwners"]
            T_4NP.name="T_4NP"
            T_4NP=T_4NP.reindex(dfTNotice.index.values)

            T_1NP = dfReport.loc[T_1]["NPParentCompanyOwners"]
            T_1NP.name = "T_1NP"
            T_1NP = T_1NP.reindex(dfTNotice.index.values)

            T_4NPQ = dfReport.loc[T_4]["NPParentCompanyOwnersQuarter"]
            T_4NPQ.name = "T_4NPQ"
            T_4NPQ = T_4NPQ.reindex(dfTNotice.index.values)

            T_5NP = dfReport.loc[T_5]["NPParentCompanyOwners"]
            T_5NP.name = "T_5NP"
            T_5NP = T_5NP.reindex(dfTNotice.index.values)
            if currentMonthDate==331:
                dfTNoticeCal=dfTNotice["NPForecaset"].to_frame()
                dfTNoticeCal["lastYearNP"]=lastYearNP
                dfTNoticeCal["T_4NP"] = T_4NP
                dfTNoticeCal["lastNPPCT"] =   dfTNoticeCal["T_4NP"]/dfTNoticeCal["lastYearNP"]
                dfTNoticeCal["lastNPPCT"]=dfTNoticeCal.apply(lambda x: 0.25 if ((x["lastYearNP"]<5000000) | (x["T_4NP"]<2000000)| (x["lastNPPCT"]<0.1)| (x["lastNPPCT"]>0.5)) else x["lastNPPCT"],axis=1)
                dfTNoticeCal["NPForecasetQuarter"] = dfTNoticeCal["lastNPPCT"]*dfTNoticeCal["NPForecaset"]
                dfTNotice["NPForecasetQuarter"]=dfTNoticeCal["NPForecasetQuarter"]
            elif currentMonthDate == 1231:
                dfTNotice["NPForecasetQuarter"] = (dfTNotice["NPForecaset"] - T_1NP)
            elif currentMonthDate == 630:
                dfTNoticeCal = dfTNotice["NPForecaset"].to_frame()
                dfTNoticeCal["lastYearNP"] = lastYearNP
                dfTNoticeCal["T_1NP"] = T_1NP
                dfTNoticeCal["T_4NPQ"] = T_4NPQ
                dfTNoticeCal["T_5NP"] = T_5NP
                dfTNoticeCal["lastNPPCT"] = dfTNoticeCal["T_4NPQ"] / (
                            dfTNoticeCal["lastYearNP"] - dfTNoticeCal["T_5NP"])
                dfTNoticeCal["lastNPPCT"] = dfTNoticeCal.apply(lambda x: 0.33 if (
                            ((x["lastYearNP"] - x["T_5NP"]) < 4000000) | (x["T_4NPQ"] < 2000000) | (
                                x["lastNPPCT"] < 0.15) | (x["lastNPPCT"] > 0.65)) else x["lastNPPCT"], axis=1)
                dfTNotice["NPForecasetQuarter"] = (dfTNoticeCal["NPForecaset"] - dfTNoticeCal["T_1NP"]) * dfTNoticeCal[
                    "lastNPPCT"]
            elif currentMonthDate == 930:
                dfTNoticeCal=dfTNotice["NPForecaset"].to_frame()
                dfTNoticeCal["lastYearNP"]=lastYearNP
                dfTNoticeCal["T_1NP"] = T_1NP
                dfTNoticeCal["T_4NPQ"] = T_4NPQ
                dfTNoticeCal["T_5NP"] = T_5NP
                dfTNoticeCal["lastNPPCT"] =dfTNoticeCal["T_4NPQ"]/(dfTNoticeCal["lastYearNP"]-dfTNoticeCal["T_5NP"])
                dfTNoticeCal["lastNPPCT"]=dfTNoticeCal.apply(lambda x: 0.5 if (((x["lastYearNP"]-x["T_5NP"])<3000000) | (x["T_4NPQ"]<2000000)| (x["lastNPPCT"]<0.2)| (x["lastNPPCT"]>0.8)) else x["lastNPPCT"],axis=1)
                dfTNotice["NPForecasetQuarter"] = (dfTNoticeCal["NPForecaset"]-dfTNoticeCal["T_1NP"])*dfTNoticeCal["lastNPPCT"]

            dfTNotice["ExceedExp"]=(dfTNotice["NPParentCompanyOwnersQuarter"]-dfTNotice["NPForecasetQuarter"])
            dfTNotice["ExceedExpPCT"] =  dfTNotice["ExceedExp"] /np.abs(dfTNotice["NPForecasetQuarter"])
            dfTNotice["ExceedExp2STD"] = dfTNotice["ExceedExp"] /dfTNotice[npSTD]

            currentObjectiveDF=dfTNotice


            currentObjectiveDF.index.name='code'
            currentObjectiveDF = currentObjectiveDF.reset_index()
            currentObjectiveDF["reportDate"] = T
            currentObjectiveDF.set_index(["reportDate", "code"], inplace=True)
            objectiveDF = currentObjectiveDF if objectiveDF is None else objectiveDF.append(currentObjectiveDF)

        df=objectiveDF [["type","ExceedExp","ExceedExpPCT","ExceedExp2STD","announceDate"]]

        df["announceDate"] = df["announceDate"].astype('int64')
        df=df.reset_index().sort_values(by=['code', 'announceDate', 'reportDate'])

        self.addFactorDataSet(df,self.dataSetName)
    def getColletionValues(self,codeList,date):

       df=self.getFactorDataSet()
       minDate=dc.DataCenter.getPeriodStartDate(120,date)
       df=df[(df["announceDate"]>=minDate) & (df["announceDate"]<=date)]
       result=pd.Series()
       if self.paraList["报表"].value == "预告":
           df = df[df["type"] == 1]
       elif self.paraList["报表"].value == "正式":
           df = df[df["type"] == 0]
       if self.paraList["类型"].value=="比预期":
           result=df.groupby('code')['ExceedExpPCT'].agg('last')
       elif self.paraList["类型"].value == "比标准差":
           result = df.groupby('code')['ExceedExp2STD'].agg('last')
       result = result.replace({float('inf'): np.nan, float('-inf'): np.nan})
       result=result.reindex(codeList)
       result.name=self.fullName
       return result

class TargetPrice(psf.ParentStockFactor):
        selectedKey = []

        def __init__(self):
            super().__init__()
            self.name = "目标价"


            p1 = pr.Parameter("最少机构", 1)
            p1.type = "Float"
            self.paraList[p1.name] = p1

            p1 = pr.Parameter("期限", 120)
            p1.type = "Float"
            self.paraList[p1.name] = p1

            p1 = pr.Parameter("来源", "ZYYX")
            p1.enumList.append("ZYYX")
            p1.enumList.append("JYDB")
            self.paraList[p1.name] = p1

            self.paraList[p1.name] = p1
            self.keyName = None
            self.needFactorNameDic = {}

            self.storedKey = None
            self.limitPeriod = 30
            self.valueType = None
            self.minInst = 1

            self.source = None

        def init(self):
            super().init()
            self.limitPeriod = self.paraList["期限"].value


            self.minInst = self.paraList["最少机构"].value

            self.source = self.paraList["来源"].value

            from Factor.Stock.Quote import ExDivi
            f=ExDivi()
            self.addNeedFactor("复权因子",f)
            self.dataSetName = self.name + self.source



        def loadData(self):
            # 先要记载一致预期基础表格
            if self.dataSetName not in dc.DataCenter.dataSet.factorDataSetDic:

                startDate = str(self.factorTradingDates.min())
                endDate = str(self.factorTradingDates.max())
                df = None

                if self.source == 'ZYYX':
                    sqlString = "select  [stock_code] as code,report_type  ,[organ_id],convert(int,(CONVERT(varchar(100), current_create_date,112)))  as create_date,[current_target_price] FROM [zyyx_new].[dbo].[rpt_target_price_adjust] where current_target_price>0  and  [current_create_date] between '" + startDate + "' and '" + endDate + "'  "
                    conn = dc.DataCenter.getConInstance("ZYYX")
                    df = pd.read_sql(sqlString, con=conn)
                    df["source"] =2
                    df.loc[df.report_type.isin([22, 23,24,25]), 'source'] =1
                elif self.source == "JYDB":
                    sqlString = """SELECT sm.SecuCode as code
                                    ,gpr.OrgCode as organ_id
                              	  ,convert(int,(CONVERT(varchar(100), gpr.[WritingDate] ,112))) create_date
                                    ,[ExRightGoalPrice] as current_target_price
                                    ,rr.ReportType as report_type
                              	  ,rr.ResearchDepth
                                FROM [JYDB].[dbo].[C_RR_GoalPriceRate] as gpr, C_RR_ResearchReport as rr,[JYDB].[dbo].SecuMain as sm 
                                where gpr.OID=rr.ID and gpr.InnerCode=sm.InnerCode """
                    sqlString += " and  ExRightGoalPrice>0 and gpr.[WritingDate] between '" + startDate + "' and '" + endDate + "'  "
                    conn = dc.DataCenter.getConInstance("JYDB")
                    df = pd.read_sql(sqlString, con=conn)
                    df["source"] = 2
                    df.loc[df.report_type.isin([6, 8]), 'source'] = 1

                exDiviF=self.getNeedFactor("复权因子")
                exDiviDF=exDiviF.getFactorDataSet()
                exDiviDF=exDiviDF.reset_index()

                uniqueCodes=list(exDiviDF["code"].unique())

                exDiviDF=exDiviDF.set_index(["code",'tradingDate'])
                exDiviDF=exDiviDF.sort_index()

                df=df[df["code"].isin(uniqueCodes)]
                df["exDivi"]=df.apply(lambda x:exDiviDF.loc[x["code"]].loc[:x["create_date"]]['factorValue'].iloc[-1],axis=1)
                df["current_target_priceAdj"] = df["exDivi"]*df["current_target_price"]
                df = df.set_index([ "create_date"])
                df = df.sort_index()
                self.addFactorDataSet(df,self.dataSetName)

        def getColletionValues(self, codeList, date):
            result = None
            pf3 = self.getFactorDataSet()
            startDate = dc.DataCenter.getPeriodStartDate(self.limitPeriod, date)
            result = pf3.loc[startDate:date].groupby(["code", "organ_id"])[
                'current_target_priceAdj'].agg('last')
            instNum = result.groupby("code").count()
            instNum = instNum[instNum >= self.minInst]
            result = result.groupby("code").mean()
            result = result.reindex(instNum.index.values)
            result = result if codeList is None else result.reindex(codeList)
            result.name = self.fullName
            return result

class TargetPriceChange(psf.ParentStockFactor):


    def __init__(self):
        super().__init__()
        self.name = "目标价"




        p1 = pr.Parameter("区间", 60)
        p1.type = "Float"
        self.paraList[p1.name] = p1




        self.needFactorNameDic = {}





    def init(self):
        super().init()
        f=TargetPrice()
        self.addNeedFactor("目标价", f)
        self.period = self.paraList["区间"].value





    def getColletionValues(self, codeList, date):
        result = None

        startDate = dc.DataCenter.getPeriodStartDate(self.period, date)
        tartgetPF=self.getNeedFactor("目标价")
        currentV=tartgetPF.getColletionValues(codeList, date)
        lastV = tartgetPF.getColletionValues(codeList, startDate)
        result =currentV/lastV-1

        result = result if codeList is None else result.reindex(codeList)
        result.name = self.fullName
        return result

class TargetPriceDistance(psf.ParentStockFactor):


    def __init__(self):
        super().__init__()
        self.name = "目标价"


        p1 = pr.Parameter("是否调整", "是")
        p1.enumList.append("否")

        self.paraList[p1.name] = p1

        self.needFactorNameDic = {}

        self.isAdjusted=False


    def init(self):
        super().init()
        f=TargetPrice()
        self.addNeedFactor("目标价",f)

        f=Quote()
        f.setParaValue("类型","ClosePriceAdj")
        self.addNeedFactor("当前价",f)
        if self.paraList["是否调整"].value=="是":
            self.isAdjusted=True
            from Factor.Stock.Momentum import Momentum
            f=Momentum()
            f.setParaValue("区间",60)
            self.addNeedFactor("收益率",f)


    def getColletionValues(self, codeList, date):
        result = None
        tartgetPF=self.getNeedFactor("目标价")
        currentPF = self.getNeedFactor("当前价")
        currentV=currentPF.getColletionValues(codeList, date)
        tartgetV = tartgetPF.getColletionValues(codeList, date)
        result =tartgetV/currentV-1
        if self.isAdjusted:
            periodReturn=self.getNeedFactor("收益率").getColletionValues(codeList, date).fillna(0)
            result=ToolFactor.normalizeValues(result,periodReturn)
        result = result if codeList is None else result.reindex(codeList)
        result.name = self.fullName
        return result


#参考东方做法，只计算超出比例FOM
class FOM(psf.ParentStockFactor):

    def __init__(self):
        super().__init__()
        self.name = "业绩上调比例"  #


        p1 = pr.Parameter("区间", 180)
        self.paraList[p1.name] = p1

        self.storedKey = None

    def init(self):
        super().init()

        consensus=Consensus()
        self.addNeedFactor("NPForecaset",consensus)

        # isq = ProfitNotice()
        # isq.paraList["类型"].value = "All"
        # self.addNeedFactor("业绩预告",isq)



    def getColletionValues(self,codeList,date):
       period = self.paraList["区间"].value
       startDate = dc.DataCenter.getPeriodStartDate(period, date) #计算排序的区间
       calStartDate = dc.DataCenter.getPeriodStartDate(90, date) #最近一次变动采纳区间
       reportYear = int(date // 10000)
       currentMonth = float(date // 100 % 100)

       #region 分析师部分
       df = self.getNeedFactor("NPForecaset").getFactorDataSet()
       if reportYear not in df.index:
           return pd.Series()
       currentDF = df.loc[reportYear].loc[startDate:date]
       currentDF = currentDF[currentDF["source"] == 1].reset_index()
       currentDF = currentDF.groupby(["create_date","code"])[
           'forecast_np'].mean()
       # currentDF=currentDF.reset_index()
       forecast_npCount=currentDF.groupby("code").count()
       forecast_npCount=forecast_npCount[forecast_npCount>=3]
      #endregion
       #region 业绩预告部分 测试后不建议加入
       # currentYearDate=reportYear*10000+1231
       # reportDF=self.getNeedFactor("业绩预告").getFactorDataSet()
       # reportDF=reportDF.loc[currentYearDate]
       # reportDF = reportDF.reset_index()
       # reportDF["create_date"]= reportDF["announceDate"]
       # reportDF=reportDF.set_index(["create_date","code"]).sort_index()["NPParentCompanyOwners"]
       # reportDF.name="forecast_np"
       # reportDF=reportDF.loc[startDate:date]
       # if reportDF.shape[0]>1:
       #     currentDF=currentDF.append(reportDF)
       #     currentDF=currentDF[~currentDF.index.duplicated(keep='last')]
       #endregion
       forecast_npRank = currentDF.groupby("code", group_keys=False, ).apply(lambda x: x.rank(pct=True))

       forecast_npRank=forecast_npRank-0.5
       forecast_npRank = forecast_npRank.sort_index()
       forecast_npRank = forecast_npRank.loc[calStartDate:date]
       # result = forecast_npRank.groupby("code").agg('last')
       #对于预测进行加权
       td = dc.DataCenter.dataSet.tradingDates
       td = td[(td >= startDate) & (td <= date)]

       result = forecast_npRank.unstack()
       result = result.reindex(td)
       tw = np.arange(td.count() - 1, -1, -1)
       tw = np.power(0.5, tw / 60.0)
       tw = tw / tw.sum()
       result = result.multiply(tw, axis=0)
       result=result.stack().groupby("code",axis=0).agg('last')

       result=result.reindex(forecast_npCount.index.values)
       result=result.reindex(codeList)
       result.name=self.fullName
       return result

