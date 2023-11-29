import torch
import Model.ModelHelper
import Sampling.getTrainingData
from Dataset.MyDataset import MyDataset
from PhaseDiagram_Fx import Fx_handler
from Learner.Learner import Learner
from Verifier.Verifier import Verifier
import superp
import Plot.Draw
from Clock.TimeOverhead import TimeOverhead
from Log import LogHandler


# 获取fx,向量场函数句柄
fx_ = Fx_handler.fx_incubator_with_col()
device = superp.device
torch.set_default_dtype(torch.float64)
train_Clock = TimeOverhead()
verify_Clock = TimeOverhead()


if __name__ == "__main__":

    Sampling.getTrainingData.Sample_Handler.getTrainingData()

    Xi_path = "Sampling/SamplingData/Xi_set_data.csv"
    Xu_path = "Sampling/SamplingData/Xu_set_data.csv"
    X_path = "Sampling/SamplingData/X_set_data.csv"

    sampling_dataset =  MyDataset(Xi_path, Xu_path, X_path)
    learner = Learner(sampling_dataset, fx_)
    verifier = Verifier(fx_)

    loop = 1
    train_Clock.timeStart()
    learner.startTrain()
    train_Clock.timeEnd()
    train_Clock.timeCost("训练过程")
    LogHandler.train_record_log(loop, train_Clock, "训练过程")
    eye = False
    while not eye:
        verify_Clock.timeStart()
        eye = verifier.startVerify()
        verify_Clock.timeEnd()
        LogHandler.verify_record_log(loop, verify_Clock, "验证过程")
        if eye:
            break
        else:
            loop += 1
            train_Clock.timeStart()
            learner.datasetReload(Xi_path, Xu_path, X_path)
            learner.continue_train()
            train_Clock.timeEnd()
            LogHandler.train_record_log(loop, train_Clock, "训练过程")

# 使用训练完成的NN
else:
    # 使用验证通过的控制器 来绘制模拟运动路径
    Control_model = Model.ModelHelper.loadNN("NN_Train_Result/Final_Result/Control_model.pth")
    Plot.Draw.path_simulation(fx_, Control_model)

