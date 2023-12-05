import superp
import Model.Barrier_Func_Model as Barrier_Func_Model
import Model.Control_Func_Model as Control_Func_Model
import Model.ModelHelper as ModelHelper
import Loss_Encoding.LossHandler as LossHandler
import Log.LogHandler as LogHandler
import torch

class Learner():

    def __init__(self, sampling_dataset, fx_):
        self.k_ICBC_model = None
        self.Control_model = None
        self.sampling_dataset = sampling_dataset
        self.device = superp.device
        self.fx_ = fx_

    def init_NN_Model(self):
        kICBCs_input_dim = superp.Bx_INPUT_DIM

        kICBCs_output_dim = superp.Bx_OUTPUT_DIM

        Control_input_dim = superp.Col_INPUT_DIM

        Control_output_dim = superp.Col_OUTPUT_DIM

        self.k_ICBC_model = Barrier_Func_Model.NN_Barrier(input_dim=kICBCs_input_dim,
                                                        output_dim=kICBCs_output_dim).to(device=self.device)

        self.Control_model = Control_Func_Model.NN_Control(input_dim=Control_input_dim,
                                                           output_dim=Control_output_dim).to(device=self.device)

    def datasetReload(self, path_Xi, path_Xu, path_X):
        self.sampling_dataset.reload(path_Xi, path_Xu, path_X)


    def train(self, lr = 0.001, flag = False):

        # optimizer = torch.optim.SGD(model.parameters(), 0.001)
        optimizer1 = torch.optim.Adam(self.k_ICBC_model.parameters(), lr)
        optimizer2 = torch.optim.Adam(self.Control_model.parameters(), lr)

        # —————————————————————————————————————— train ——————————————————————————————————————
        n_epochs = 0
        tr_set_X_init, tr_set_X_unsafe, tr_set_X, tr_set_X_bounded_area = self.sampling_dataset.fullBatch()
        while True:
            self.k_ICBC_model.train()
            self.Control_model.train()

            # batch
            #tr_set_X_init, tr_set_X_unsafe, tr_set_X, tr_set_X_bounded_area = self.sampling_dataset.batchDivision(
            #    superp.Xi_batch_size, superp.Xu_batch_size, superp.X_batch_size, superp.X_bounded_area_batch_size)

            n_epochs = n_epochs + 1
            totalLoss = 0
            totalLoss1 = 0
            totalLoss2 = 0
            totalLoss3 = 0
            totalLoss4 = 0
            batch_num = 0

            # for x, y in zip(tr_set_X_Not_Target, tr_set_X_biggerThan_zero):
            for x, y, z, other in zip(tr_set_X_init, tr_set_X_unsafe, tr_set_X, tr_set_X_bounded_area):
                batch_num = batch_num + 1

                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)
                z = torch.tensor(z).to(self.device)
                other = torch.tensor(other).to(self.device)

                x = x.type(torch.float64)
                y = y.type(torch.float64)
                z = z.type(torch.float64)
                other = other.type(torch.float64)


                optimizer1.zero_grad()
                optimizer2.zero_grad()

                # 初始区域
                Xi_k_point = LossHandler.calc_K_iteration(x, self.fx_, self.Control_model)
                Loss1 = LossHandler.Xi_Loss_Func(Xi_k_point, self.k_ICBC_model)

                # 不安全区域
                pre_y = self.k_ICBC_model(y)
                Loss2 = LossHandler.Xu_Loss_Func(pre_y)

                # 第三个蕴含的条件
                pre_z = LossHandler.Filter_Of_Loss3(z, self.fx_, self.k_ICBC_model, self.Control_model)

                Loss3 = 0.
                if pre_z.shape[0] == 0:
                    Loss3 = torch.tensor(0.0)
                else:
                    Loss3 = LossHandler.X_Loss_Func(pre_z)

                pre_other = self.k_ICBC_model(other)
                Loss4 =  LossHandler.Xu_Loss_Func(pre_other)


                Loss = Loss1 + Loss2 + Loss3 + Loss4
                print(f"——————————————————Epoch No.{n_epochs}——————————————————————————————————————————————")
                print(f"------No.{batch_num} batch------")
                print(f"Loss1 value is {Loss1}")
                print(f"Loss2 value is {Loss2}")
                print(f"Loss3 value is {Loss3}")
                print(f"Loss4 value is {Loss4}")

                totalLoss1 = totalLoss1 + Loss1
                totalLoss2 = totalLoss2 + Loss2
                totalLoss3 = totalLoss3 + Loss3
                totalLoss4 = totalLoss4 + Loss4
                totalLoss = totalLoss + Loss
                if Loss != 0:
                    Loss.backward()
                    optimizer1.step()
                    optimizer2.step()

            print(f"No. {n_epochs}, totalLoss = {totalLoss}")
            LogHandler.info_log(n_epochs, totalLoss, totalLoss1, totalLoss2, totalLoss3, totalLoss4)
            if totalLoss == 0.:
                break

        # save
        ModelHelper.saveNN(self.k_ICBC_model, "NN_Train_Result/Final_Result/k_ICBC_model.pth")
        ModelHelper.saveNN(self.Control_model, "NN_Train_Result/Final_Result/Control_model.pth")
        if flag:
            ModelHelper.saveNN(self.k_ICBC_model, "NN_Train_Result/First_Train/k_ICBC_model.pth")
            ModelHelper.saveNN(self.Control_model, "NN_Train_Result/First_Train/Control_model.pth")

        return n_epochs


    def counterexample_train(self, lr=0.001):
        self.k_ICBC_model = ModelHelper.loadNN("NN_Train_Result/Final_Result/k_ICBC_model.pth")
        self.Control_model = ModelHelper.loadNN("NN_Train_Result/Final_Result/Control_model.pth")
        return self.train(lr)

    def startTrain(self):
        self.init_NN_Model()
        return self.train(flag=True)
