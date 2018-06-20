import numpy as np

from simulation.SnetAgent import SnetAgent


class SISTER(SnetAgent):
    def __init__(self, unique_id, model, message, parameters):
        super().__init__(unique_id, model, message, parameters)
        if not message:
            vs = self.vector_size()
            floatVec = np.random.uniform(low=0.0, high=1.0, size=(vs,))
            self.message = self.float_vec_to_trade_plan(floatVec)
            model.blackboard.append(self.message)

    def step(self):
        # stubbed, just keep the message as it is
        # print(str(self.unique_id) + " in step" )
        print(self.b[self.unique_id]['label'] + ' in step')

    def paymentNotification(self, agiTokens, tradenum):
        print(
            self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
