import Verifier.MILP.milp_verify as milp_verify
import Model.ModelHelper as ModelHelper
class Verifier():

    def __init__(self, fx_):
        self.k_ICBC_model = None
        self.Control_model = None
        self.fx_ = fx_

    def startVerify(self):
        self.k_ICBC_model = ModelHelper.loadNN("NN_Train_Result/Final_Result/k_ICBC_model.pth")
        self.Control_model = ModelHelper.loadNN("NN_Train_Result/Final_Result/Control_model.pth")

        print("——————————————————————————————————初始条件——————————————————————————————————")
        eye0, counter_ex_init, _ = milp_verify.MILP_opt_initCon(self.k_ICBC_model, self.Control_model, self.fx_)

        print("——————————————————————————————————不安全条件——————————————————————————————————")
        eye1, counter_ex_unsafe, _ = milp_verify.MILP_opt_unsafeCon(self.k_ICBC_model)

        # 蕴含条件验证
        print("——————————————————————————————————蕴含条件——————————————————————————————————")
        eye2, counter_ex_thirdCon, _ = milp_verify.MILP_opt_thirdCond(self.k_ICBC_model, self.Control_model, self.fx_)

        if eye0 and eye1 and eye2:
            print("合成了真正的k-ICBC!!!")
        else:
            print("存在反例  已完成反例采样  继续训练...")

        return eye0 & eye1 & eye2