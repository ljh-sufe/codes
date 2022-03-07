from datacenter import datacenter
from model.NNfather import NNfather
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import logging

device = "cuda" if torch.cuda.is_available() else "cpu"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class daRnnModel(NNfather):

    def __init__(self):
        super(daRnnModel, self).__init__()

    def init(self):

        btp = self.para["回溯范围"]
        ti = self.para["采样周期"]  # time interval

        trainStartDate = datacenter.trainStartDate
        evalStartDate = datacenter.evalStartDate

        bitcoin = datacenter.bitcoin
        dataSet = bitcoin.dataSet.copy()

        XTrain, yTrain, XEval, yEval, dataSet = self.resampleing(dataSet)

        # %%

        XTrain = pd.concat([dataSet[dataSet.index <= trainStartDate].iloc[-btp:, 0:7], XTrain], axis=0)
        yHisTrain = pd.concat([dataSet[dataSet.index <= trainStartDate].iloc[-btp - ti:, 7], yTrain], axis=0)
        X = torch.Tensor(XTrain.values)
        y = torch.Tensor(yHisTrain.values)
        len_ = len(yTrain)
        dataXT = torch.zeros(size=(len_, int(btp / ti) + 1, 7))
        dataYhisT = torch.zeros(len_, int(btp / ti) + 1, 1)

        logging.info("preparing train data")
        for i in tqdm(range(len_)):
            dataXT[i] = X[np.linspace(i, i + btp, int(btp / ti) + 1)]
            dataYhisT[i] = y[[ti * j + i for j in range(int(btp / ti) + 1)]].unsqueeze(1)

        XEval = pd.concat([dataSet[dataSet.index < evalStartDate].iloc[-btp:, 0:7], XEval], axis=0)
        yHisEval = pd.concat([dataSet[dataSet.index <= evalStartDate].iloc[-btp - ti:, 7], yEval], axis=0)
        len_ = len(yEval)
        dataXE = torch.zeros(size=(len_, int(btp / ti) + 1, 7))
        dataYhisE = torch.zeros(len_, int(btp / ti) + 1, 1)
        X = torch.Tensor(XEval.values)
        y = torch.Tensor(yHisEval.values)
        logging.info("preparing eval data")
        for i in tqdm(range(len_)):
            dataXE[i] = X[np.linspace(i, i + btp, int(btp / ti) + 1)]
            dataYhisE[i] = y[[ti * j + i for j in range(int(btp / ti) + 1)]].unsqueeze(1)

        datayT = torch.Tensor(yTrain.values)
        datayE = torch.Tensor(yEval.values)

        X_train_max = dataXT.max(axis=0).values
        X_train_min = dataXT.min(axis=0).values
        y_his_train_max = dataYhisT.max(axis=0).values
        y_his_train_min = dataYhisT.min(axis=0).values
        target_train_max = datayT.max(axis=0).values
        target_train_min = datayT.min(axis=0).values

        dataXT = (dataXT - X_train_min) / (X_train_max - X_train_min)
        dataXE = (dataXE - X_train_min) / (X_train_max - X_train_min)
        dataYhisT = (dataYhisT - y_his_train_min) / (y_his_train_max - y_his_train_min)
        dataYhisE = (dataYhisE - y_his_train_min) / (y_his_train_max - y_his_train_min)
        datayT = (datayT - target_train_min) / (target_train_max - target_train_min)
        datayE = (datayE - target_train_min) / (target_train_max - target_train_min)

        self.train_(dataXT, dataYhisT, datayT, dataXE, dataYhisE, datayE)

    def train_(self, dataXT, dataYhisT, datayT, dataXE, dataYhisE, datayE):
        epochs = self.para["epochs"]
        batchSize = self.para["batchSize"]
        # model = torchRNNModel(X_train.shape[2], rnnType="LSTM", hidSize=10)
        model = torchdaRnn(dataXT.shape[2], 8, 8, dataXT.shape[1]).to(device)

        model = model.to(device)
        model.writer = SummaryWriter()
        model.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        for epoch in tqdm(range(epochs)):
            ## shuffle data
            shuffledIndex = np.random.permutation(dataXT.shape[0])
            dataX = dataXT[shuffledIndex, :, :].to(device)
            datayhis = dataYhisT[shuffledIndex].to(device)
            datay = datayT[shuffledIndex].to(device)

            for i in range(int(dataX.shape[0] / batchSize)):
                model.oneBatchTrain(dataX[i * batchSize: (i + 1) * batchSize, :, :],
                                    datayhis[i * batchSize: (i + 1) * batchSize],
                                    datay[i * batchSize: (i + 1) * batchSize])
            avgTrainLoss = np.mean(model.trainLossList)
            avgTrainIC = np.mean(model.trainICList)

            self.trainLossList.append(avgTrainLoss)
            self.trainICList.append(avgTrainIC)
            model.writer.add_scalar("train loss", avgTrainLoss, len(self.trainLossList))
            model.writer.add_scalar("train IC", avgTrainIC, len(self.trainICList))
            model.trainLossList = []
            model.eval_(dataXE, dataYhisE, datayE)

        model.writer.close()
        self.model = model

        self.model.eval()
        yPred_train = self.model(dataXT.to(device), dataYhisT.to(device)).detach().cpu().numpy().reshape(-1)

        print("训练集（", datacenter.trainStartDate, "- ", datacenter.trainEndDate, "）上相关系数：",
              np.corrcoef(yPred_train, datayT.numpy())[0, 1])
        yPred_eval = self.model(dataXE.to(device), dataYhisE.to(device)).detach().cpu().numpy().reshape(-1)
        print("验证集（", datacenter.evalStartDate, "- ", datacenter.evalEndDate, "）上相关系数：",
              np.corrcoef(yPred_eval, datayE.numpy())[0, 1])


class torchdaRnn(torch.nn.Module):
    '''
    数据形式为： 数据点个数 * 时间长度 * 变量数
    '''
    trainLossList = []
    trainICList = []
    evalICList = []
    writer = None
    optimizer = None
    lossFunc = torch.nn.MSELoss()

    def __init__(self, N, M, P, T, stateful_encoder=False, stateful_decoder=False):
        super(self.__class__, self).__init__()
        self.encoder = InputAttentionEncoder(N, M, T, stateful_encoder).to(device)
        self.decoder = TemporalAttentionDecoder(M, P, T, stateful_decoder).to(device)

    def forward(self, X_history, y_history):
        x = self.encoder(X_history)
        out = self.decoder(x, y_history)
        return out

    def oneBatchTrain(self, X, yhis, y):
        self.train()
        ypred = self.forward(X, yhis)
        ypred = (ypred - ypred.mean()) / ypred.std()
        # loss = -self.get_corr(ypred, y)
        loss = self.lossFunc(ypred, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        IC = np.corrcoef(ypred.detach().cpu().numpy().reshape(-1), y.detach().cpu().numpy())[1, 0]
        nploss = loss.detach().cpu().numpy()
        self.trainLossList.append(nploss)
        self.trainICList.append(IC)

    def eval_(self, X_eval, yhis_eval, y_eval):
        1
        self.eval()
        y_pred = self.forward(X_eval.to(device), yhis_eval.to(device))

        IC = np.corrcoef(y_pred.detach().cpu().numpy().reshape(-1), y_eval.detach().cpu().numpy())[1, 0]

        self.evalICList.append(IC)
        self.writer.add_scalar("eval IC", IC, len(self.evalICList))

    def get_corr(self, ypred, y):  # 计算两个向量person相关系数

        ypred_mean, Y_mean = torch.mean(ypred), torch.mean(y)
        corr = (torch.sum((ypred - ypred_mean) * (y - Y_mean))) / (
                torch.sqrt(torch.sum((ypred - ypred_mean) ** 2)) * torch.sqrt(torch.sum((y - Y_mean) ** 2)))
        return corr


class InputAttentionEncoder(torch.nn.Module):
    def __init__(self, N, M, T, stateful=False):
        """
        :param: N: int
            number of time serieses
        :param: M:
            number of LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T

        self.encoder_lstm = nn.LSTMCell(input_size=self.N, hidden_size=self.M)

        # equation 8 matrices

        self.W_e = nn.Linear(2 * self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)

    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.T, self.M)).to(device)

        # initiale hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.M)).to(device)
        s_tm1 = torch.zeros((inputs.size(0), self.M)).to(device)

        for t in range(self.T):
            # concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)

            # attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.N, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))

            # normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)

            # weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :]

            # calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))

            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs


class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T, stateful=False):
        """
        :param: M: int
            number of encoder LSTM units
        :param: P:
            number of deocder LSTM units
        :param: T:
            number of timesteps
        :param: stateful:
            decides whether to initialize cell state of new time window with values of the last cell state
            of previous time window or to initialize it with zeros
        """
        super(self.__class__, self).__init__()
        self.M = M
        self.P = P
        self.T = T
        self.stateful = stateful

        self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.P)

        # equation 12 matrices
        self.W_d = nn.Linear(2 * self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias=False)

        # equation 15 matrix
        self.w_tilda = nn.Linear(self.M + 1, 1)

        # equation 22 matrices
        self.W_y = nn.Linear(self.P + self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)

    def forward(self, encoded_inputs, y):
        # initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(device)
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.P)).to(device)
        for t in range(self.T):
            # concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            # print(d_s_prime_concat)
            # temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)

            # normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)

            # create context vector (equation_14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)

            # concatenate c_t and y_t
            y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
            # create y_tilda
            y_tilda_t = self.w_tilda(y_c_concat)

            # calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))

        # concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        # calculate output
        y_Tp1 = self.v_y(self.W_y(d_c_concat))
        return y_Tp1
