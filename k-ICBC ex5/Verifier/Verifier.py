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

        print("——————————————————————————————————X0——————————————————————————————————")
        eye0, counter_ex_init, _ = milp_verify.MILP_opt_initCon(self.k_ICBC_model, self.Control_model, self.fx_)

        print("——————————————————————————————————Xu——————————————————————————————————")
        eye1, counter_ex_unsafe, _ = milp_verify.MILP_opt_unsafeCon(self.k_ICBC_model)

        print("——————————————————————————————————X——————————————————————————————————")
        eye2, counter_ex_thirdCon, _ = milp_verify.MILP_opt_thirdCond(self.k_ICBC_model, self.Control_model, self.fx_)

        if eye0 and eye1 and eye2:
            print("Obtain a true k-ICBC!!!")
        else:
            print("Find counterexamples,  continue training...")

        return eye0 & eye1 & eye2