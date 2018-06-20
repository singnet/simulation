import random
from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np
from mesa import Agent
from sklearn.metrics.pairwise import cosine_similarity


class SnetAgent(Agent, ABC):
    def __init__(self, unique_id, model, message, parameters):
        # In the subclass, make sure to submit an initial message to the blackboard, if the message field is blank.
        # When the simulation initializes, the first thing created are the agents with initial messages defined in
        # the configuration file.  In these cases, the message field to this routine is filled in with the agent's
        # message from the config file and the unique id of the agent is the order in the file. Next, agents are
        # created that do not have a specified message in the configuration file. The parameters of the config file
        # refers to them as the random_agents, and specifies the number to generate of each. A function is supplied
        # ,float_vec_to_trade_plan, that converts a vector of floats to a message, that could be a random vector.
        # However, it is up the individual agent what initial message to submit to the board. Regardless of the origin
        #  of their initial message,  agents are parameterized in the config file parameters under the name
        # agent_parameters.  These parameters are tailored to their subclasses and come into this routine as the
        # "parameters" parameter.

        super().__init__(unique_id, model)
        self.message = message
        self.p = self.model.parameters  # convenience method:  its shorter
        self.b = self.model.blackboard  # convenience method:  its shorter
        self.o = self.model.ontology  # convenience method:  its shorter
        self.parameters = parameters
        self.wealth = 0

    @abstractmethod
    def step(self):
        # Each agent has a message slot on the blackboard, and the purpose of this step is to submit a new message
        # into this slot, if the subclassed agent so desires.  Human agents do not submit new messages during the run
        #  of the simulation, which operates on a different time scale than they do.  Machine learning agents that
        # submit a message here can see the detailed results of their message added to that message the next time
        # this function is called.  These results include the entire blackboard, with each agent indicating the
        # messages they bought from self, selfs test scores, and the money given to self.  All this is taken account
        # in the net AGI tokens, which is available to agents to use as part of the reward for reinforcement learning
        #  or machine learning. The change in tokens from the last time step has been called is the result of the
        # previous message.  A notification is sent that this agent can use to keep track of the net,
        # in the payment_notification convenience method. For the final step the agent can observe the results of the
        # last and submit None to the blackboard.
        pass

    # todo: find out why this doesnt work @abstractmethod
    def payment_notification(self, agi_tokens, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an agi_tokens amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc.  This is called after each
        pass

    @staticmethod
    def price_overlap(buy, sell):
        # overlap occurs when the trades are
        # sorted and the lowest price of an offer is higher then the highest price of the previous offer

        price_overlap = False
        if buy['low'] <= sell['low'] and sell['low'] <= buy['high'] \
                or sell['low'] <= buy['low'] and buy['low'] <= sell['high']:
            price_overlap = True

        return price_overlap

    @staticmethod
    def price(buy, sell):
        # overlap occurs when the trades are
        # sorted and the lowest price of an offer is higher then the highest price of the previous offer
        # The agreed upon price is the midpoint of the overlap

        price = None
        if buy['low'] <= sell['low'] and sell['low'] <= buy['high']:
            price = (sell['low'] + buy['high']) / 2

        elif sell['low'] <= buy['low'] and buy['low'] <= sell['high']:
            price = (buy['low'] + sell['high']) / 2

        return price

    def gather_offers(self):
        # for every buy offer an agent has, look for sell offers from other agents for an item that is the same
        # category asked for, for which there is an overlap in price. One can tell the same category because the
        # ontology name begins in the same way list the possible trades to be considered in the individual buy trades
        #  of the agents tradeplan. which there is overlap in price. list offers in the message, uniqueId:tradeNum
        # The lowest cosine similarity never wins, because a random cosine similarity can still be around 60,
        # and we want the ones that have learned signs to have even greater chance of succeeding.

        for buy in self.message['trades']:
            if buy['type'] == 'buy':
                offers = []
                for i, message in enumerate(self.b):
                    offer = None
                    for j, sell in enumerate(message['trades']):
                        stop_cut_off = buy['item'].split('_stop')
                        if (sell['type'] == 'sell'
                            and sell['item'].startswith(stop_cut_off[0]) and self.price_overlap(buy, sell)):
                            if not offer:
                                sought_sign = np.array(buy['sign']).reshape(-1, len(buy['sign']))
                                displayed_sign = np.array(self.b[i]['sign']).reshape(-1, len(self.b[i]['sign']))
                                # print ('sought_sign.shape')
                                # print (sought_sign.shape)
                                sim = cosine_similarity(sought_sign, displayed_sign)
                                if sim:
                                    sim = sim.flatten()[0]
                                offer = {'agent': i, 'cosine_sim': sim, 'trades': []}
                                offers.append(offer)
                            offer['trades'].append(j)
                buy['offers'] = offers

        # convert cosine distances into probabilities
        for buy in self.message['trades']:
            if buy['type'] == 'buy':
                if len(buy['offers']) > 1:
                    minimum = 1.0
                    for offer in buy['offers']:
                        if offer['cosine_sim'] < minimum:
                            minimum = offer['cosine_sim']
                    simsum = 0
                    for offer in buy['offers']:
                        simsum += (offer['cosine_sim'] - minimum)
                    for offer in buy['offers']:
                        offer['probability'] = (offer['cosine_sim'] - minimum) / simsum
                elif len(buy['offers']) == 1:
                    buy['offers'][0]['probability'] = 1.0

                    # print(str(self.unique_id) + " in gather offers" )

    def retrieve_ontology_item(self, cumulative_category):
        # return the ontology item from the underscore notation moregeneral_lessgeneralandmorespecific_evenmorespecific
        # it must start with the very first category
        # print('cumulative_category')
        # print(cumulative_category)
        adict = {}
        if cumulative_category:
            levels = cumulative_category.split('_')
            adict = self.o
            for level in levels:
                if level in adict:
                    adict = adict[level]
                elif level == 'ontology':
                    pass
                else:
                    return adict
        return adict

    def descendants(self, cumulative_category):
        # see if this ontology item is the most specific possible, which it must be for example, if it is a test and
        # is to be called with one item : the tested program, or if it is to be called and yield data.
        descendants = []
        cat_dict = self.retrieve_ontology_item(cumulative_category)

        no_descendants = True
        for name, adict in cat_dict.items():
            if not name.startswith('_') and isinstance(adict, dict):
                no_descendants = False
                if cumulative_category == 'ontology':
                    descendants.extend(self.descendants(name))
                else:
                    descendants.extend(self.descendants(cumulative_category + '_' + name))
        if no_descendants:
            descendants.append(cumulative_category)
        return descendants

    def perform_test(self, function_list):
        score = 0

        self.model.gepResult = self.gep( function_list)  # put the ordered Dictionary in the global so a decorated function can access
        if any(self.model.gepResult.values()):
            root = next(iter(self.model.gepResult.items()))[0]
            score_tuple = self.model.call_memoise_pickle(root)
            if score_tuple and len(score_tuple) >1 and score_tuple[1]:
                score = score_tuple[1]
        return score

    def arity(self, func):
        description = self.retrieve_ontology_item(func)
        return len(description["_args"])

    def gep(self, function_list_dont_modify):
        # assign input and output functions as defined by the Karva notation.
        # get arity of the items and divide the levels according to that arity,
        # then make the assignments across the levels

        # example.  for the following program list with the following arity, the karva notation result is the following
        # self.learnedProgram = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
        # self.arity = {'a':2,'b':3,'c':2,'d':2,'e':1,'f':1,'g':2,'h':1,'i':1,'j':1,'k':0,'l':0,'m':1,'n':1,'o':0,'p':0,'q':0,'r':0,'s':0}
        # self.results = {'a':['b','c'],'b':['d','e','f'],'c':['g','h'], 'd':['i','j'],'e':['k'],
        #      'f':['l'],'g':['m','n'],'h':['o'],'i':['p'],'j':['q'],'m':['r'], 'n':['s']}

        # divide into levels

        function_list = []
        function_list.extend(function_list_dont_modify)
        levels = {1: [function_list.pop(0)]}

        current_level = 1
        while function_list:
            length_next_level = 0
            for func in levels[current_level]:
                length_next_level += self.arity(func)
            current_level += 1
            levels[current_level] = function_list[0:length_next_level]
            function_list = function_list[length_next_level:]
            if not length_next_level:
                break

        # make assignments

        gep_result = OrderedDict()
        for level, function_list in levels.items():
            next_level = level + 1
            cursor = 0
            for func in function_list:
                next_cursor = cursor + self.arity(func)
                if next_level in levels:
                    gep_result[func] = levels[next_level][cursor:next_cursor]
                cursor = next_cursor


        return gep_result

    @staticmethod
    def transfer_funds(buyer, buynum, seller, sellnum, price):
        buyer.wealth -= price
        seller.wealth += price
        buyer.payment_notification(-price, buynum)
        seller.payment_notification(price, sellnum)

    def distribute_funds(self, buy, buynum):

        offer = buy['offers'][buy['chosen']]
        sellnum = offer['trades'][0]

        sells = self.b[offer['agent']]['trades'][offer['trades'][0]:]

        found = -1
        for i, trade in enumerate(sells):
            if trade['item'].startswith('stop') or trade['type'] == 'stop':
                found = i

        sells = sells[0:found]
        # now find the buys and insert settled software (or ignore)
        # list.insert(index, elem)

        if sells:
            self.transfer_funds(self, buynum, self.model.schedule.agents[offer['agent']], sellnum, buy['price'])
        buylist = [i for i, sell in enumerate(sells) if sell['type'] == buy and 'chosen' in sell and sell['chosen']]

        for i in buylist:
            sellnum = offer['trades'][0] + i
            seller = offer['agent']
            seller.distribute_funds(sells[i], sellnum)

    def obtain_program_list(self, offer):

        trade_list = self.obtain_trade_list(offer)
        program_list = [trade['item'] for trade in trade_list]
        clean_list = []
        for program in program_list:
            stop_cut_off = program.split('_stop')
            clean_list.append(stop_cut_off[0])
        return clean_list

    def obtain_trade_list(self, offer):
        # todo implement more than one item sold by agent, for now just take the first one
        # software that one is selling is defined as the trades from the first one marked sold to either
        # the end of the trades or a stop codon. if one of them is a buy,then that software is obtained as well.
        # the buy must have a chosen offer number and the agent who offered the goods must have them, and so on.
        # todo:  Right now, this does not implement subroutines. That is the very next thing to do.  The list here
        # goes into the gep calculation of the purchaser, mixing it up so it isnt a subroutine.  Rather this should
        # be put in the ontology, even if a temporary one, with an arity figured out by gep, and then be put in the
        # registry.  the Gep should only apply within an agent, and then the result, is a single subroutine which
        # is then counted as a single united program for the gep that buys it.  Until then, this program can only
        # handle multi agent pipelines (each member having arity 1 or 0).  Gep within agent, pipeline outside agent.
        # print("in obtain software")
        sells = self.b[offer['agent']]['trades'][offer['trades'][0]:]
        # print("sells")
        # print(sells)
        found = -1
        for i, trade in enumerate(sells):
            if trade['item'].startswith('stop') or trade['type'] == 'stop':
                found = i

        sells = sells[0:found]

        # now find the buys and insert settled software (or ignore)

        buys_inserted = []
        buylist = [i for i, sell in enumerate(sells) if sell['type'] == 'buy' and 'chosen' in sell and sell['chosen']]
        cursor = 0

        if not buylist:
            buys_inserted = sells
        for i in buylist:
            next_cursor = i
            buys_inserted.extend(sells[cursor:next_cursor])
            cursor = next_cursor + 1
            sell = buylist[i]
            buys_inserted.extend(self.obtain_trade_list(sell['offers'][sell['chosen']]))

        return buys_inserted

    def pass_all_tests(self, buy, offernum):

        # Implements a testing system where a human or agent can require any test in a category on any data in a
        # category if generality is required, or all tests on data that are listed. This function tells if the agent
        # has passed all tests, and saves the scores for feedback to the agent. One test that must be passed is the
        # offerer must have the item.  It does not have to be all bought, but the part that is said to be the item
        # must be present, if the offer is the result of a sell that was not constructed,but bought.  Second,
        # every test listed in the tests, before the first stop codon, must be passed The test score on each test is
        # noted in the test score description

        # First retrieve item to buy.  Item is defined as including all from the sell statement to the next stop or the
        # end of the list.  if there are buys, and they are settled, retrieve the bought software

        # a stop in the first place of either the test or the data is a stop for all tests.
        # if the stop is midway through the test or the data, it indicates general tests or data,
        # that is, if it passes for any test or data in this category over this threshold it passes
        pass_all_tests = True
        itemlist = None
        if offernum:
            itemlist = self.obtain_program_list(buy['offers'][offernum])
        if itemlist:
            stop_codon_reached = False
            for test_dict in buy['tests']:
                if not stop_codon_reached:
                    stop_cut_off = test_dict['test'].split('_stop')
                    clean_test = stop_cut_off[0]
                    stop_cut_off = test_dict['data'].split('_stop')
                    clean_data = stop_cut_off[0]
                    if not clean_test or not clean_data:
                        stop_codon_reached = True
                    else:
                        # The data or tests may be general categories, so see if there are tests
                        # or data in those categories
                        testlist = self.descendants(clean_test)
                        if not testlist:
                            testlist.append(clean_test)
                        datalist = self.descendants(clean_data)
                        if not datalist:
                            datalist.append(clean_data)


                        anypass = False
                        for test in testlist:
                            for data in datalist:
                                # if any is passed in this group, then give a pass
                                program_list = [test]
                                program_list.extend(itemlist)
                                program_list.append(data)

                                #non-coding segments are implemented when non completed functions are ignored
                                program_list = [program for program in program_list if program in self.model.registry]

                                score = self.perform_test(program_list)

                                # record the score no matter what, as feedback
                                if 'results' not in test_dict:
                                    test_dict['results'] = []
                                result = {'offer': offernum, 'score': score, 'time': self.model.schedule.time,
                                          'test': test, 'data': data}
                                test_dict['results'].append(result)

                                if score and score > test_dict['threshold']:
                                    anypass = True
                        if not anypass:
                            pass_all_tests = False
        else:
            pass_all_tests = False

        return pass_all_tests

    def choose_partners(self):
        # for every buy trade, roll to pick who fills the slot. If a test is required, run what the agent has
        # through the test, and if it doesnt pass, nothing fills the slot this time.  If an agent buys goods from
        # an agent that does ot have them, the same thing happens. All other buys can not be redone once accepted.
        # Whatever the state of the called routine when the human accepts it is what is paid, as opposed to when the
        # buying agent accepts it. In other words, an agent can If the agent is a human, who accepts the trade,
        # money is disbursed throughout the network according to the chain of contracted prices.

        for buynum, buy in enumerate(self.message['trades']):
            if buy['type'] == 'buy':
                weighted_choices = {offernum: offer['probability'] for offernum, offer in enumerate(buy['offers'])}
                roll = random.uniform(0, 1)
                winning_num = self.choose_weighted(weighted_choices, roll)
                if self.pass_all_tests(buy, winning_num):
                    buy['chosen'] = winning_num
                    sell = self.b[buy['offers'][winning_num]['agent']]['trades'][buy['offers'][winning_num]['trades'][0]]
                    buy['price'] = self.price(buy, sell)
                    if self.message['type'] == 'Human':
                        self.distribute_funds(buy, buynum)
                        #self.distribute_funds(buy['offers'][winning_num], buynum)


    # functions that translate the float vec to a trade plan that can go on the blackboard

    def vector_size(self):
        return (
            self.p['sign_size'] +
            self.p['num_trade_plans'] * self.trade_size())

    def trade_size(self):
        return (1 + self.p['sign_size'] +
                self.p['item_size'] + 2 +
                self.p['num_tests'] * self.test_size())

    def test_size(self):
        return 2 * self.p['item_size'] + 2

    def trade_type(self, afloat):
        trade_type = 'stop'
        weighted_choices = {'buy': 1, 'construct': 1, 'sell': 1}
        choice = self.choose_weighted(weighted_choices, afloat, previously_taken_space=self.p['chance_of_stop_codon'])
        if choice:
            trade_type = choice
        return trade_type

    def hidden(self, afloat):
        weighted_choices = {True: 1, False: 1}
        hidden = self.choose_weighted(weighted_choices, afloat)
        return hidden

    def weights_for_level(self, cumulative_category):
        weights_for_level = {}

        cat_dict = self.retrieve_ontology_item(cumulative_category)
        for name, adict in cat_dict.items():
            if not name.startswith('_') and isinstance(adict, dict) and '_weight' in adict:
                weights_for_level[name] = adict['_weight']

        return weights_for_level

    def ontology_item(self, float_vec, category='ontology', include_stop=True):
        # if category is filled in, the float starts from the given category
        cumulative_category = category
        for afloat in float_vec:
            weighted_choices = self.weights_for_level(cumulative_category)

            if not any(weighted_choices.values()):
                break
            # roll = random.uniform(0, 1)
            roll = afloat
            if include_stop:
                choice = 'stop'
                previously_taken_space = self.p['chance_of_stop_codon']
            else:
                previously_taken_space = 0
            guess = self.choose_weighted(weighted_choices, roll, previously_taken_space=previously_taken_space)
            if guess:
                choice = guess
            if cumulative_category == 'ontology':
                cumulative_category = choice
            else:
                cumulative_category = cumulative_category + "_" + choice

        return cumulative_category

    def agi_token(self, afloat):
        return self.p["min_token_price"] + (self.p["max_token_price"] - self.p["min_token_price"]) * afloat

    @staticmethod
    def choose_weighted(weighted_choices, roll, previously_taken_space=0):

        # transform to cumulative distribution
        total = 0
        for choice, weight in weighted_choices.items():
            total += weight

        normalized = {choice: (weight / total) for choice, weight in weighted_choices.items()}

        previous_weight = 0
        cumulative = {}
        for choice, weight in normalized.items():
            cumulative[choice] = weight + previous_weight
            previous_weight += weight

        space = 1 - previously_taken_space
        chosen = None

        for choice, weight in cumulative.items():
            if roll < weight * space:
                chosen = choice
                break


        return chosen

    def float_vec_to_trade_plan(self, float_vec):
        cursor = 0
        trade_plan = {'type': self.__class__.__name__}
        trade_plan['label'] = trade_plan['type'] + " Agent " + str(self.unique_id)

        # First find the sign, which is the raw float representation
        next_cursor = self.p['sign_size']
        trade_plan['sign'] = list(float_vec[cursor:next_cursor])
        cursor = next_cursor

        trade_plan['trades'] = []
        cursor_before_trade_plans = cursor


        # Then each trade plan.
        for i in range(self.p['num_trade_plans']):
            trade_plan['trades'].append(dict())
            cursor = cursor_before_trade_plans + i * self.trade_size()


            # First the type
            trade_plan['trades'][i]['type'] = self.trade_type(float_vec[cursor])

            # Next the sign, which is the raw float representation
            cursor += 1
            next_cursor = cursor + self.p['sign_size']
            trade_plan['trades'][i]['sign'] = list(float_vec[cursor:next_cursor])
            cursor = next_cursor

            # Next the item
            next_cursor = cursor + self.p['item_size']
            trade_plan['trades'][i]['item'] = self.ontology_item(float_vec[cursor:next_cursor])
            cursor = next_cursor

            # lowest price accepted
            trade_plan['trades'][i]['low'] = self.agi_token(float_vec[cursor])
            cursor += 1

            # highest price accepted

            trade_plan['trades'][i]['high'] = self.agi_token(float_vec[cursor])
            cursor += 1

            cursor_before_tests = cursor


            # Finally, each test that they buyer wants to have passed before he will accept the product

            trade_plan['trades'][i]['tests'] = []
            for j in range(self.p['num_tests']):
                trade_plan['trades'][i]['tests'].append(dict())
                cursor = cursor_before_tests + j * self.test_size()

                # The test
                next_cursor = cursor + self.p['item_size']
                trade_plan['trades'][i]['tests'][j]['test'] = self.ontology_item(float_vec[cursor:next_cursor],
                                                                                 category='test')
                cursor = next_cursor

                # The data
                next_cursor = cursor + self.p['item_size']
                trade_plan['trades'][i]['tests'][j]['data'] = self.ontology_item(float_vec[cursor:next_cursor],
                                                                                 category='data')
                cursor = next_cursor

                trade_plan['trades'][i]['tests'][j]['threshold'] = float_vec[cursor]
                cursor += 1

                trade_plan['trades'][i]['tests'][j]['hidden'] = self.hidden(float_vec[cursor])
                cursor += 1

        return trade_plan
