from simulation.SnetAgent import SnetAgent


class Human(SnetAgent):
    def step(self):
        # stubbed, just keep the message as it is
        print(self.b[self.unique_id]['label'] + ' in step')

    def paymentNotification(self, agiTokens, tradenum):
        print(
            self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))

