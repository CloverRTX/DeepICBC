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


    # 首次训练
    def train_firstly(self, lr = 0.001):

        # optimizer = torch.optim.SGD(model.parameters(), 0.001)
        optimizer1 = torch.optim.Adam(self.k_ICBC_model.parameters(), lr)
        optimizer2 = torch.optim.Adam(self.Control_model.parameters(), lr)

        # —————————————————————————————————————— train ——————————————————————————————————————
        eye = False
        n_epochs = 0
        while not eye:
            self.k_ICBC_model.train()
            self.Control_model.train()

            # 分batch
            tr_set_X_init, tr_set_X_unsafe, tr_set_X = self.sampling_dataset.batchDivision(
                superp.Xi_batch_size, superp.Xu_batch_size, superp.X_batch_size)

            n_epochs = n_epochs + 1
            totalLoss = 0
            batch_num = 0
            total_loss1 = 0.
            total_loss2 = 0.
            total_loss3 = 0.
            print(f"——————————————————第{n_epochs}次迭代——————————————————————————————————————————————")
            # for x, y in zip(tr_set_X_Not_Target, tr_set_X_biggerThan_zero):
            for x, y, z in zip(tr_set_X_init, tr_set_X_unsafe, tr_set_X):
                batch_num = batch_num + 1

                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)
                z = torch.tensor(z).to(self.device)

                x = x.type(torch.float64)
                y = y.type(torch.float64)
                z = z.type(torch.float64)


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
                # 如果没有满足蕴含条件的点
                print(f"满足蕴含条件的点的个数： {pre_z.shape[0]}")
                Loss3 = 0
                if pre_z.shape[0] == 0:
                    Loss3 = torch.tensor(0.0)
                else:
                    Loss3 = LossHandler.X_Loss_Func(pre_z)

                Loss = Loss1 + Loss2 + Loss3

                print(f"------No.{batch_num} batch------")
                print(f"Loss1 value is {Loss1}")
                print(f"Loss2 value is {Loss2}")
                print(f"Loss3 value is {Loss3}")
                total_loss1 = total_loss1 + Loss1
                total_loss2 = total_loss2 + Loss2
                total_loss3 = total_loss3 + Loss3
                totalLoss = totalLoss + Loss
                if Loss != 0:
                    Loss.backward()
                    optimizer1.step()
                    optimizer2.step()

            print(f"No. {n_epochs}, totalLoss = {totalLoss}")
            LogHandler.info_log(n_epochs, totalLoss, total_loss1, total_loss2, total_loss3)
            if totalLoss == 0.:
                eye = True
                break

            # 保存
        ModelHelper.saveNN(self.k_ICBC_model, "NN_Train_Result/Final_Result/k_ICBC_model.pth")
        ModelHelper.saveNN(self.Control_model, "NN_Train_Result/Final_Result/Control_model.pth")

        ModelHelper.saveNN(self.k_ICBC_model, "NN_Train_Result/First_Train/k_ICBC_model.pth")
        ModelHelper.saveNN(self.Control_model, "NN_Train_Result/First_Train/Control_model.pth")

        return n_epochs

    def continue_train(self, lr=0.001):

        self.k_ICBC_model =  ModelHelper.loadNN("NN_Train_Result/Final_Result/k_ICBC_model.pth")
        self.Control_model = ModelHelper.loadNN("NN_Train_Result/Final_Result/Control_model.pth")

        # optimizer = torch.optim.SGD(model.parameters(), 0.001)
        optimizer1 = torch.optim.Adam(self.k_ICBC_model.parameters(), lr)
        optimizer2 = torch.optim.Adam(self.Control_model.parameters(), lr)

        # —————————————————————————————————————— train ——————————————————————————————————————
        eye = False
        n_epochs = 0
        while not eye:
            self.k_ICBC_model.train()
            self.Control_model.train()

            # 分batch
            tr_set_X_init, tr_set_X_unsafe, tr_set_X = self.sampling_dataset.batchDivision(
                superp.Xi_batch_size, superp.Xu_batch_size, superp.X_batch_size)

            n_epochs = n_epochs + 1
            totalLoss = 0
            batch_num = 0

            total_loss1 = 0.
            total_loss2 = 0.
            total_loss3 = 0.
            print(f"——————————————————第{n_epochs}次迭代——————————————————————————————————————————————")
            # for x, y in zip(tr_set_X_Not_Target, tr_set_X_biggerThan_zero):
            for x, y, z in zip(tr_set_X_init, tr_set_X_unsafe, tr_set_X):
                batch_num = batch_num + 1

                x = torch.tensor(x).to(self.device)
                y = torch.tensor(y).to(self.device)
                z = torch.tensor(z).to(self.device)

                x = x.type(torch.float64)
                y = y.type(torch.float64)
                z = z.type(torch.float64)

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
                # 如果没有满足蕴含条件的点
                print(f"满足蕴含条件的点的个数： {pre_z.shape[0]}")
                Loss3 = 0
                if pre_z.shape[0] == 0:
                    Loss3 = torch.tensor(0.0)
                else:
                    Loss3 = LossHandler.X_Loss_Func(pre_z)

                Loss = Loss1 + Loss2 + Loss3

                print(f"------No.{batch_num} batch------")
                print(f"Loss1 value is {Loss1}")
                print(f"Loss2 value is {Loss2}")
                print(f"Loss3 value is {Loss3}")
                total_loss1 = total_loss1 + Loss1
                total_loss2 = total_loss2 + Loss2
                total_loss3 = total_loss3 + Loss3
                totalLoss = totalLoss + Loss
                if Loss != 0:
                    Loss.backward()
                    optimizer1.step()
                    optimizer2.step()

            print(f"No. {n_epochs}, totalLoss = {totalLoss}")
            LogHandler.info_log(n_epochs, totalLoss, total_loss1, total_loss2, total_loss3)
            if totalLoss == 0.:
                eye = True
                break

            # 保存
        ModelHelper.saveNN(self.k_ICBC_model, "NN_Train_Result/Final_Result/k_ICBC_model.pth")
        ModelHelper.saveNN(self.Control_model, "NN_Train_Result/Final_Result/Control_model.pth")

        return n_epochs

    def startTrain(self):

        self.init_NN_Model()
        return self.train_firstly()
