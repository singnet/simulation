import numpy as np
import random
import cma
import json
import copy
import math
import sys
import os

from simulation.SnetAgent import SnetAgent


class SISTER(SnetAgent):
    def __init__(self, unique_id, model, message, parameters):
        super().__init__(unique_id, model, message, parameters)

        # SISTER uses CMA-ES on float vectors, that are then converted
        # to trade plans.  If an initial message is put on the board,
        # this is converted to a floatvec, and is used to initialize cma-es

        vs = self.vector_size()
        self.initialVec = np.random.uniform(low=0.0, high=1.0, size=(vs,))
        if not message:
            self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec)
            self.initial_trade_plan = None
        else:

            self.initial_trade_plan = copy.deepcopy(self.message)
            mask = copy.deepcopy(self.initial_trade_plan)
            self.message, float_vec = self.float_vec_to_trade_plan(self.initialVec, mask)

        self.float_vec = float_vec
        self.model.blackboard.append(self.message)

        seed = random.randint(1,1000000)
        params = {'bounds': [0.0, 1.0], 'seed': seed, 'CMA_elitist': self.parameters['elitist']}
        self.es = cma.CMAEvolutionStrategy(self.initialVec, self.parameters['sigma'], params )
        self.solutions = self.es.ask()
        self.results = []
        self.next_solution = 0

        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0

        print ("IN SISTER init,"+self.b[self.unique_id]['label'])



    def step(self):

        print("IN SISTER step,"+self.b[self.unique_id]['label']+ " time "+ str(self.model.schedule.time))
        # print(self.b[self.unique_id]['label'] + ' in step')

        # append put the cumulative token reward as the reward for the last solution
        result = self.agiTokens * self.parameters['fitness_weights']['agi_tokens']  \
                + self.max_buyer_score * self.parameters['fitness_weights']['buyer_score'] \
                + self.max_seller_score * self.parameters['fitness_weights']['seller_score']

        bought_items = self.get_bought_items()
        self.results.append(result)
        self.model.print_reproduction_report_line(self,result, bought_items)


        # move a cursor that tells which solution you are on.
        # if there are none left, tell then ask, clearing reward buffer, resettinng cursor
        # take the next solutions and put the message on the blackboard
        self.next_solution += 1
        if self.next_solution >= len(self.solutions):
            #Checkpoints as false enables the cma-es to accept the fixed solution seeds (partially fixed solutions)
            # that are set parts of the space to evolve around
            self.es.tell(self.solutions, self.results, check_points=False)
            self.results= []
            self.next_solution = 0
            self.solutions = self.es.ask()


        mask = None
        if self.initial_trade_plan:
            step = math.floor(self.model.schedule.time)
            initial_message = self.initial_trade_plan['initial_message'] \
                if 'initial_message' in self.initial_trade_plan else 0
            final_message = self.initial_trade_plan['final_message'] \
                if 'final_message' in self.initial_trade_plan else sys.maxsize
            message_period = self.initial_trade_plan['message_period'] \
                if 'message_period' in self.initial_trade_plan else 1

            if step >= initial_message and step <= final_message and step % message_period == 0:
                mask = copy.deepcopy(self.initial_trade_plan)

        new_message, float_vec = self.float_vec_to_trade_plan(self.solutions[self.next_solution],mask)



        self.agiTokens = 0
        self.max_buyer_score = 0
        self.max_seller_score = 0
        self.set_message(new_message)
        self.float_vec = float_vec


    def buyer_score_notification(self, score, tradenum):
        #print(self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
        if self.max_buyer_score < score:
            self.max_buyer_score = score

    def seller_score_notification(self, score, tradenum):
        #print(self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
        if self.max_seller_score < score:
            self.max_seller_score = score

    def payment_notification(self, agiTokens, tradenum):
        #print(self.b[self.unique_id]['label'] + ' wealth changes by ' + agiTokens + ' because of trade ' + str(tradenum))
        self.agiTokens += agiTokens