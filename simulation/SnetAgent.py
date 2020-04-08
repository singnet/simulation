from abc import ABC, abstractmethod
from collections import OrderedDict
import copy
import random
import re
import sys

import numpy as np
from mesa import Agent
from sklearn.metrics.pairwise import cosine_similarity


class SnetAgent(Agent, ABC):
    def __init__(self, unique_id, model, message, parameters):
        # In the subclass, make sure to submit an initial message to the blackboard if the message field is blank.
        # When the simulation initializes, the first thing created are the agents with initial messages defined in
        # the configuration file.  In these cases, the message field to this routine is filled in with the agent's
        # message from the config file and the unique id of the agent is the order in the file. Next, agents are
        # created that do not have a specified message in the configuration file. The parameters of the config file
        # refers to them as the random_agents, and specifies how many of each to generate. The function
        # float_vec_to_trade_plan converts a vector of floats to a message, that could be a random vector.
        # However, it is up the individual agent what initial message to submit to the board. Regardless of the origin
        # of their initial message, agents are parameterized in the config file parameters under the name
        # agent_parameters. These parameters are tailored to their subclasses and come into this routine as the
        # "parameters" parameter.

        # Asynchronous Optimization (an ask and tell interface) is needed in subclasses because SnetSim
        # has the program control

        super().__init__(unique_id, model)
        self.message = message
        self.p = self.model.parameters  # convenience method: its shorter
        self.b = self.model.blackboard  # convenience method: its shorter
        self.o = self.model.ontology  # convenience method: its shorter
        self.parameters = parameters
        self.wealth = 0
        self.emergent_pattern = re.compile(r'^f\d+\.\d+_\d+_\d+_\d+')
        self.item_type_pattern = re.compile(r'^([a-zA-Z0-9]+)_?')
        self.test_item_type_pattern = re.compile(r'^test_([a-zA-Z0-9]+)_?')

    @abstractmethod
    def step(self):
        # Each agent has a message slot on the blackboard, and the purpose of this step is to submit a new message
        # into this slot, if the subclassed agent so desires. Human agents do not submit new messages during the run
        # of the simulation, which operates on a different time scale than they do. Machine learning agents that
        # submit a message here can see the detailed results of their message added to that message the next time
        # this function is called. These results include the entire blackboard, with each agent indicating the
        # messages they bought from self, self test scores, and the money given to self. All this is taken into account
        # in the net AGI tokens, which is available to agents to use as part of the reward for reinforcement learning
        # or machine learning. The change in tokens from the last time `step` was called is the result of the
        # previous message. A notification is sent; this agent can use it to keep track of the net,
        # in the `payment_notification` convenience method. For the final step the agent can observe the results of the
        # last, and submit None to the blackboard.
        pass

    @abstractmethod
    def payment_notification(self, agi_tokens, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an `agi_tokens` amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc. This is called after each step.
        pass

    @abstractmethod
    def seller_score_notification(self, score, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an `agi_tokens` amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc. This is called after each step.
        pass

    @abstractmethod
    def buyer_score_notification(self, score, tradenum):
        # This routine is called to notify the agent that his wealth has been changed by an `agi_tokens` amount,
        # which can be negative. The blackboard may be examined for more information on which parts of the trade
        # plans resulted in what payments, test scores, etc. This is called after each step.
        pass

    @staticmethod
    def price_overlap(buy, sell):
        # overlap occurs when the trades are
        # sorted and the lowest price of an offer is higher then the highest price of the previous offer
        # buy_low = min(buy['midpoint']-buy['range'], 0.0)
        # buy_high = max(buy['midpoint']+buy['range'], 1.0)
        # sell_low = min(sell['midpoint'] - sell['range'], 0.0)
        # sell_high = max(sell['midpoint'] + sell['range'], 1.0)
        #
        # price_overlap = False
        # if buy_low <= sell_low and sell_low <= buy_high \
        #         or sell_low <= buy_low and buy_low <= sell_high:
        #     price_overlap = True
        price_overlap = False
        if buy['midpoint'] >= sell['midpoint']:
            price_overlap = True

        return price_overlap

    @staticmethod
    def price(buy, sell):
        # overlap occurs when the trades are
        # sorted and the lowest price of an offer is higher then the highest price of the previous offer
        # The agreed upon price is the midpoint of the overlap
        # buy_low = min(buy['midpoint'] - buy['range'], 0.0)
        # buy_high = max(buy['midpoint'] + buy['range'], 1.0)
        # sell_low = min(sell['midpoint'] - sell['range'], 0.0)
        # sell_high = max(sell['midpoint'] + sell['range'], 1.0)
        # price = None
        #
        # if buy_low <= sell_low and sell_low <= buy_high:
        #     price = (sell_low + buy_high) / 2
        #
        # elif sell_low <= buy_low and buy_low <= sell_high:
        #     price = (buy_low + sell_high) / 2
        price = (buy['midpoint'] + sell['midpoint']) / 2.0

        return price

    def set_message(self, message):
        self.message = message
        self.model.blackboard[self.unique_id] = self.message

    def gather_offers(self):
        # For every buy offer an agent has, look for sell offers from other agents for an item that is the same
        # category asked for, for which there is an overlap in price. One can tell the same category because the
        # ontology name begins in the same way. List the possible trades to be considered in the individual buy trades
        # of the agent's tradeplan for which there is overlap in price. List offers in the message, uniqueId:tradeNum.
        # The lowest cosine similarity never wins, because a random cosine similarity can still be around 60,
        # and we want the ones that have learned signs to have even greater chance of succeeding.
        print("In gather_offers," + self.b[self.unique_id]['label'])

        buyer_stop_codon_reached = False
        for buy in self.message['trades']:
            if (not buyer_stop_codon_reached) and buy['type'] == 'buy':
                offers = []
                for i, message in enumerate(self.b):
                    if i != self.unique_id:
                        offer = None
                        seller_stop_codon_reached = False
                        for j, sell in enumerate(message['trades']):
                            if (not seller_stop_codon_reached) and sell['type'] == 'sell':
                                stop_cut_off = buy['item'].split('_stop')
                                if sell['item'].startswith(stop_cut_off[0]) and self.price_overlap(buy, sell):
                                    if not offer:
                                        # First the distance between the buyers sought and the sellers displayed
                                        sought_sign = np.array(buy['sign']).reshape(-1, len(buy['sign']))
                                        displayed_sign = np.array(self.b[i]['sign']).reshape(-1, len(self.b[i]['sign']))
                                        # print ('sought_sign.shape')
                                        # print (sought_sign.shape)
                                        buyers_sim = cosine_similarity(sought_sign, displayed_sign)
                                        if buyers_sim:
                                            buyers_sim = buyers_sim.flatten()[0]

                                        # Next the distance between the sellers sought and the buyers displayed
                                        sought_sign = np.array(sell['sign']).reshape(-1, len(sell['sign']))
                                        displayed_sign = np.array(self.message['sign']).reshape(-1,
                                                                                                len(self.message[
                                                                                                        'sign']))
                                        # print ('sought_sign.shape')
                                        # print (sought_sign.shape)
                                        sellers_sim = cosine_similarity(sought_sign, displayed_sign)
                                        if sellers_sim:
                                            sellers_sim = sellers_sim.flatten()[0]

                                        # weighted sum of the buyers and sellers similarities
                                        sim = (self.p['buyers_weight'] * buyers_sim) + (
                                                (1 - self.p['buyers_weight']) * sellers_sim)
                                        offer = OrderedDict([('agent', i), ('cosine_sim', sim), ('trades', [])])
                                        offers.append(offer)
                                    offer['trades'].append(j)
                            elif sell['type'] == 'stop':
                                seller_stop_codon_reached = True
                buy['offers'] = offers
            elif buy['type'] == 'stop':
                buyer_stop_codon_reached = True

        # convert cosine distances into probabilities
        buyer_stop_codon_reached = False
        for buy in self.message['trades']:
            if (not buyer_stop_codon_reached) and buy['type'] == 'buy':
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
            elif buy['type'] == 'stop':
                buyer_stop_codon_reached = True
                # print(str(self.unique_id) + " in gather offers" )

    def retrieve_ontology_item(self, cumulative_category):
        # Return the ontology item from the underscore notation moregeneral_lessgeneralandmorespecific_evenmorespecific
        # it must start with the very first category
        # print('cumulative_category')
        # print(cumulative_category)
        adict = OrderedDict()
        if cumulative_category:
            levels = cumulative_category.split('_')
            adict = self.o
            # print('levels')
            # print(levels)
            for level in levels:
                if level in adict:
                    adict = adict[level]
                elif level == 'ontology':
                    pass
                else:
                    return OrderedDict()
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
        pickle_name = ""

        # put the ordered Dictionary in the global so a decorated function can access
        gepResult = self.modular_gep(function_list)
        if any(gepResult.values()):
            root = next(iter(gepResult.items()))[0]
            score_tuple = self.model.call_emergent_function(gepResult, root)
            if score_tuple and len(score_tuple) and score_tuple[0]:
                pickle_name = score_tuple[0]
            if score_tuple and len(score_tuple) > 1 and score_tuple[1]:
                score = score_tuple[1]

        return gepResult, score, pickle_name

    def clean_item_name(self, name):
        new_name = self.model.remove_prefix(name)
        new_name = self.model.remove_suffix(new_name)
        return new_name

    def original_arity(self, func):
        arity = None
        description = self.retrieve_ontology_item(func)
        # print('func')
        # print(func)
        if description and "_args" in description:
            arity = len(description["_args"])
        # if description and  "_args" not in description:
        # print ('description')
        # print (description)
        return arity

    def arity(self, func):
        new_func = self.clean_item_name(func)
        arity = self.model.emergent_functions_arity[new_func] \
            if new_func in self.model.emergent_functions_arity \
            else self.original_arity(new_func)
        return arity

    def function_list_arity(self, function_list_dont_modify):
        function_list = []
        function_list.extend(function_list_dont_modify)
        arity = 0
        if function_list:
            levels = OrderedDict([(1, [function_list.pop(0)])])
            current_level = 1
            more = True
            while more:
                length_next_level = 0
                for func in levels[current_level]:
                    arity = self.arity(func)
                    length_next_level += arity
                current_level += 1
                levels[current_level] = function_list[0:length_next_level]
                function_list = function_list[length_next_level:]
                arity = length_next_level - len(levels[current_level])
                if not length_next_level:
                    break
                more = function_list or levels[current_level]
                # more = length_next_level

        return arity

    def next_call_number_prefix(self):
        prefix = 'f' + str(self.model.emergent_functions_call_number) + '_'
        self.model.emergent_functions_call_number += 1
        return prefix

    def prefix_call_numbers(self, function_list):
        prefix_call_numbers = []
        functions_once = OrderedDict()
        for function in function_list:
            if function not in prefix_call_numbers:
                prefix = self.next_call_number_prefix()
                functions_once[function] = prefix + function

            prefix_call_numbers.append(functions_once[function])
        return prefix_call_numbers

    def gep(self, function_list_dont_modify):
        # Assign input and output functions as defined by the Karva notation.
        # get arity of the items and divide the levels according to that arity,
        # then make the assignments across the levels

        # Example: for the following program list with the following arity, the Karva notation result is the following
        # self.learnedProgram = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s']
        # self.arity = {'a':2,'b':3,'c':2,'d':2,'e':1,'f':1,'g':2,'h':1,'i':1,'j':1,'k':0,'l':0,'m':1,'n':1,'o':0,
        #               'p':0,'q':0,'r':0,'s':0}
        # self.results = {'a':['b','c'],'b':['d','e','f'],'c':['g','h'], 'd':['i','j'],'e':['k'],
        #                 'f':['l'],'g':['m','n'],'h':['o'],'i':['p'],'j':['q'],'m':['r'], 'n':['s']}

        # divide into levels

        # function_list = []
        # function_list.extend(function_list_dont_modify)
        gep_result = OrderedDict()
        function_list = self.prefix_call_numbers(function_list_dont_modify)
        if function_list:
            levels = OrderedDict([(1, [function_list.pop(0)])])
            current_level = 1
            while function_list:
                length_next_level = 0
                for func in levels[current_level]:
                    noprefix = self.model.remove_prefix(func)
                    arity = self.arity(noprefix)
                    length_next_level += arity
                current_level += 1
                levels[current_level] = function_list[0:length_next_level]
                function_list = function_list[length_next_level:]
                if not length_next_level:
                    break

            # make assignments
            for level, function_list in levels.items():
                next_level = level + 1
                cursor = 0
                for func in function_list:
                    noprefix = self.model.remove_prefix(func)
                    arity = self.arity(noprefix)
                    next_cursor = cursor + arity
                    if next_level in levels:
                        gep_result[func] = levels[next_level][cursor:next_cursor]
                    else:
                        gep_result[func] = []
                    cursor = next_cursor

        return gep_result

    def get_all_emergent_subroutines(self, function_list):
        # if any function in the list is emergent, then add it and its own emergent
        # subroutines to the list

        emergent_subroutines = set()

        for function in function_list:
            if function in self.model.emergent_functions:
                emergent_subroutines.update(self.get_all_emergents_set(function))
        return list(emergent_subroutines)

    def get_all_emergents_set(self, function_name, call_depth=0):
        emergent_subroutines = set()
        if call_depth < self.p["recursive_trade_depth_limit"]:
            if function_name in self.model.emergent_functions:
                children = self.model.emergent_functions[function_name]
                for child in children:
                    level = call_depth + 1
                    descendants = self.get_all_emergents_set(child, call_depth=level)
                    emergent_subroutines.update(descendants)
                emergent_subroutines.add(function_name)
            # print('emergent_subroutines')
            # print(emergent_subroutines)
        return emergent_subroutines

    def gep_clean(self, gepResult):
        # return true if this gepResult has an emergent function anywhere
        geptuple = self.geptuple(gepResult)
        clean = not any((x and self.emergent_pattern.match(x)) for x in geptuple)
        return clean

    def modular_gep(self, function_list):
        # take a function list, that has the emergent functions in it.  go through
        #  the list and for each emergent function it has, get its list,and for each
        # they have, get its list, until you have the original function list and
        # the entire tree of its subroutines.
        # now, send each list to gep and get a list of gep results.  send that list to
        #  another routine, combine_modules, that takes the list of gep results, and
        # creates a gep result that has only registry functions in it.

        # print('function_list')
        # print(function_list)
        emergent_functions = self.get_all_emergent_subroutines(function_list)

        # print('emergent_functions')
        # print(emergent_functions)
        gep_ordered_dict = OrderedDict()
        gep_ordered_dict['root'] = self.gep(function_list)
        gep_dict = OrderedDict([(f, self.gep(self.model.emergent_functions[f])) for f in emergent_functions])
        gep_ordered_dict.update(gep_dict)
        gep_result = self.combine_modules(gep_ordered_dict)

        # print ('gep_result')
        # print (gep_result)

        if not self.gep_clean(gep_result):
            print('recursive function not allowed :')
            print(gep_result)
            gep_result = OrderedDict()
        return gep_result

    def make_equivalent_gep(self, gep_result):
        prefix_map = OrderedDict()
        equivalent_gep = OrderedDict()
        for func_name, arglist in gep_result.items():
            prefix = self.model.get_call_prefix(func_name)
            if prefix not in prefix_map:
                prefix_map[prefix] = self.next_call_number_prefix()
            for arg in arglist:
                prefix = self.model.get_call_prefix(arg)
                if prefix not in prefix_map:
                    prefix_map[prefix] = self.next_call_number_prefix()

        for func_name, arglist in gep_result.items():
            new_arglist = []
            for arg in arglist:
                prefix = self.model.get_call_prefix(arg)
                if prefix and (prefix in prefix_map):
                    new_arglist.append(prefix_map[prefix] + self.model.remove_prefix(arg))
                else:
                    print('null prefix')
            prefix = self.model.get_call_prefix(func_name)
            new_func_name = prefix_map[prefix] + self.model.remove_prefix(func_name)
            equivalent_gep[new_func_name] = new_arglist

        return equivalent_gep

    def find_unbounds(self, gep_result):
        unbounds = set()
        for func, arglist in gep_result.items():
            regular_function_arity = self.arity(func)
            if regular_function_arity is not None:
                if regular_function_arity > 0 and len(gep_result[func]) == 0:
                    unbounds.add(func)
            # elif func in self.model.emergent_functions_arity and self.model.emergent_functions_arity[func] > 0 \
            # and len(gep_result[func]) == 0:
            # unbounds.add(func)
            for arg in arglist:
                regular_function_arity = self.arity(arg)
                if regular_function_arity is not None:
                    if regular_function_arity > 0 and (arg not in gep_result or len(gep_result[arg]) == 0):
                        unbounds.add(arg)
                # elif arg in self.model.emergent_functions_arity and self.model.emergent_functions_arity[arg]> 0 \
                # and arg not in gep_result or len(gep_result[arg]) ==0:
                # unbounds.add(arg)
        return list(unbounds)

    def flattern(self, a):
        rt = []
        for i in a:
            if isinstance(i, list):
                rt.extend(self.flattern(i))
            else:
                rt.append(i)
        return rt

    def geptuple(self, gepresult):
        # flatten, remove prefix, make a tuple
        geptuple = tuple()
        if len(gepresult):
            alist = self.flattern(gepresult.values())
            alist.extend(gepresult.keys())
            blist = [self.model.remove_prefix(x) for x in alist]
            geptuple = tuple(set(blist))
        return geptuple

    def combine_modules(self, emergent_function_dict_dont_modify):
        # print ('emergent_function_dict_dont_modify')
        # print(emergent_function_dict_dont_modify)
        emergent_function_dict = copy.deepcopy(emergent_function_dict_dont_modify)
        # take the list of gep results, and
        # create a gep result that has only registry functions in it.

        #  for every function that uses emergent functions Q,in dictionary of all emergent functions as gep results D
        # take each emergent separately.  the emergent functions that have arity will appear as a key.
        # remove them from the list and set aside  a, b, cOrderedDict([('root', OrderedDict([('f0_test_clusterer_silhouette', ['f1_clusterer_sklearn_affinityPropagation_10clusters']), ('f1_clusterer_sklearn_affinityPropagation_10clusters', ['f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq']), ('f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq', ['f3_preprocessor_freetext_tag']), ('f3_preprocessor_freetext_tag', ['f4_preprocessor_freetext_shuffle_stochastic4']), ('f4_preprocessor_freetext_shuffle_stochastic4', ['f5_data_freetext_internetResearchAgency'])]))])
        # for each of these usages, create a new copy of the emergent function that has different call order numbers (but arranged the same) d, e, f
        # in d e f, change the call number of the root, if there is one,  to be the call number from the removed functions, and change every place that the emergent function is in an input list to its new name (a b c)
        # then, in d e f,  map functions that are missing inputs to the inputs of the removed functions(a b c)
        # put contents of modified emergent functions d e f  back into the original function list Q
        # remove the emergent function from the larger list D, emergent_function_dict
        # OrderedDict([('root', OrderedDict([('f0_test_clusterer_silhouette', ['f1_clusterer_sklearn_affinityPropagation_10clusters']), ('f1_clusterer_sklearn_affinityPropagation_10clusters', ['f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq']), ('f2_vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq', ['f3_preprocessor_freetext_tag']), ('f3_preprocessor_freetext_tag', ['f4_preprocessor_freetext_shuffle_stochastic4']), ('f4_preprocessor_freetext_shuffle_stochastic4', ['f5_data_freetext_internetResearchAgency'])]))])

        # all of these are emergent functions, but some of them dont use other emergent functions and some do

        use_emergent_functions = OrderedDict(
            [(fname, gep_result) for fname, gep_result in emergent_function_dict.items() if
             not self.gep_clean(gep_result)])
        previous_length = sys.maxsize
        while (not len(use_emergent_functions) == 0) and len(use_emergent_functions) < previous_length:
            previous_length = len(use_emergent_functions)

            use_only_non_emergent_functions = OrderedDict(
                [(fname, gep_result) for fname, gep_result in emergent_function_dict.items() \
                 if self.gep_clean(gep_result)])
            # we are depending on emergent_function_dict contents to be passed by value
            # for every function that uses emergent functions Q (user_gep_result),
            # take each emergent separately.  the emergent functions that have arity will appear as a key.
            # remove them from the list and set aside  a, b, c (emergent_funct_usages)

            for user_name, user_gep_result in use_emergent_functions.items():
                for non_user_name, non_user_gep_result in use_only_non_emergent_functions.items():
                    # emergent_funct_usages = {name: gep_results for name, gep_results in
                    # use_emergent_functions.items() \
                    # if non_user_name in name}
                    emergent_funct_usages = OrderedDict([(name, gep_results) for name, gep_results in \
                                                         user_gep_result.items() if non_user_name in name])

                    # for each of these usages, create a new copy of the emergent function non_user_gep_result that has
                    # different call order numbers (but arranged the same) d, e, f (combined functions)

                    combined_functions = [self.make_equivalent_gep(non_user_gep_result) for _ in emergent_funct_usages]

                    # in d e f, (combined functions) change the call number of each root key
                    # to be the call number from the removed functions, and change every place that the
                    # emergent function is in an input list to its new name (a b c)(emergent_funct_usages)

                    for i, (name, arg_list) in enumerate(emergent_funct_usages.items()):
                        prefix = self.model.get_call_prefix(name)
                        # for gep_result in combined_functions:
                        gep_result = combined_functions[i]
                        if len(gep_result):
                            root = next(iter(gep_result.items()))[0]
                            new_name = prefix + self.model.remove_prefix(root)
                            gep_result[new_name] = gep_result.pop(root)
                            gep_result.move_to_end(new_name, last=False)
                            for fname, arglist in user_gep_result.items():
                                if name in arglist:
                                    arglist[arglist.index(name)] = new_name

                            # then, in d e f, (combined functions) map functions that are missing inputs to the inputs
                            # of the removed functions(a b c) (emergent_funct_usages).
                            # Match the prefix of the root in combined functions to the prefix of the key in emergent
                            # funct usages

                            input_assignments = OrderedDict()
                            unbound_list = self.find_unbounds(gep_result)

                            cursor = 0
                            for unbound in unbound_list:
                                # arity = self.model.emergent_functions_arity[unbound]
                                arity = self.arity(unbound)
                                next_cursor = cursor + arity
                                input_assignments[unbound] = arg_list[cursor:next_cursor]
                                cursor = next_cursor
                            # put contents of modified emergent functions d e f  back into the original function list
                            # Q (user_gep_result)

                            user_gep_result.update(gep_result)
                            if list(user_gep_result.keys()).index(name) == 0:  # this is the root
                                user_gep_result.move_to_end(new_name, last=False)

                            user_gep_result.update(input_assignments)
                            user_gep_result.pop(name)
                        else:  # empty function
                            user_gep_result.pop(name)
                            for fname, arglist in user_gep_result.items():
                                if name in arglist:
                                    arglist[arglist.index(name)] = None

                        # remove the emergent function from the original function list D (emergent_function_dict)
                        ename = self.model.remove_prefix(name)
                        if ename in emergent_function_dict:
                            emergent_function_dict.pop(ename)

            use_emergent_functions = OrderedDict(
                [(fname, gep_result) for fname, gep_result in emergent_function_dict.items()
                 if not self.gep_clean(gep_result)])

            # use_emergent_functions = {fname: gep_result for fname, gep_result in emergent_function_dict.items() \
            # if any ( f in tuple(self.model.emergent_functions.keys()) for f in  tuple(gep_result.values()))}

        # print ('emergent_function_dict')
        # print(emergent_function_dict)
        result = emergent_function_dict['root']
        return result

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

        found = False
        stopval = -1
        for i, trade in enumerate(sells):
            if (not found) and (trade['item'].startswith('stop') or trade['type'] == 'stop'):
                found = True
                stopval = i

        sells = sells[0:stopval]
        # now find the buys and insert settled software (or ignore)
        # list.insert(index, elem)

        if sells:
            self.transfer_funds(self, buynum, self.model.schedule.agents[offer['agent']], sellnum, buy['price'])
        buylist = [i for i, sell in enumerate(sells) if sell['type'] == 'buy' and 'chosen' in sell and sell['chosen']]

        for i in buylist:
            sellnum = offer['trades'][0] + i
            seller = self.model.schedule.agents[offer['agent']]
            seller.distribute_funds(sells[i], sellnum)

    def obtain_trade_list(self, offer, call_depth=0):
        # todo implement more than one item sold by agent, for now just take the first one
        # software that one is selling is defined as the trades from the first one marked sold to either
        # the end of the trades or a stop codon. if one of them is a buy,then that software is obtained as well.
        # the buy must have a chosen offer number and the agent who offered the goods must have them, and so on.
        # a recursion depth limit exists in the parameters

        if call_depth > self.p["recursive_trade_depth_limit"]:
            unique_id = None
        else:
            unique_id = "f" + str(self.model.schedule.time) + "_" + str(self.unique_id) \
                        + "_" + str(offer['agent']) + "_" + str(offer['trades'][0])
            if unique_id not in self.model.emergent_functions:
                # print("in obtain software")
                sells = self.b[offer['agent']]['trades'][offer['trades'][0]:]
                # print("sells")
                # print(sells)
                found = False
                stopval = -1
                for i, trade in enumerate(sells):
                    if (not found) and (trade['item'].startswith('stop') or trade['type'] == 'stop'):
                        found = True
                        stopval = i

                sells = sells[0:stopval]

                # We want to refer to this piece of code again in the same iteration without making an
                # extra copy of it in the emergent_functions routine.  to do so give it a unique id.
                # uniqueid is made of self.model.schedule.time, self.unique_id, offer['agent'], offer['trades'][0]

                # now find the buys and insert settled software (or ignore)

                # buys_inserted = []
                buylist = [i for i, sell in enumerate(sells) if sell['type'] == 'buy'
                           and 'chosen' in sell and sell['chosen'] is not None]

                # this is the straight up insertion of code, rather than a subroutine
                # cursor = 0
                # if not buylist:
                #     buys_inserted = sells
                # for i in buylist: #iterate by writing out all the sells, and then the buy insertion
                #     next_cursor = i
                #     buys_inserted.extend(sells[cursor:next_cursor])
                #     cursor = next_cursor + 1 #skip the actual buy
                #     #sell = buylist[i]
                #     sell = sells[i]
                #     buys_inserted.extend(self.obtain_trade_list(sell['offers'][sell['chosen']]))

                clean_funcs = []
                for sell in sells:
                    func = sell['item']
                    clean_func = func.split('_stop')
                    clean_funcs.append(clean_func[0])

                for i in buylist:
                    sell = sells[i]
                    depth = call_depth + 1
                    clean_funcs[i] = self.obtain_trade_list(sell['offers'][sell['chosen']], call_depth=depth)

                only_existing = [program for program in clean_funcs
                                 if program and ((self.model.remove_suffix(program) in self.model.registry or
                                                  program in self.model.emergent_functions))]

                self.model.emergent_functions[unique_id] = only_existing
                self.model.emergent_functions_arity[unique_id] = self.function_list_arity(only_existing)

        return unique_id  # self.model.emergent_functions[unique_id]

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
        cumulative_score = 0
        pickle_name = ""
        gepResult = None
        pass_all_tests = True
        numtests = 0
        if offernum is not None:
            func_name = self.obtain_trade_list(buy['offers'][offernum])
            itemlist = self.model.emergent_functions[func_name]
            if itemlist:
                stop_codon_reached = False

                for test_dict in buy['tests']:
                    if test_dict['stophere']:
                        stop_codon_reached = True
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
                                    # however, do not run it unless the software is the same type as the test
                                    item_type = self.item_type(itemlist[0])
                                    test_item_type = self.test_item_type(test)
                                    if item_type and (item_type == test_item_type) and len(
                                            self.retrieve_ontology_item(test)):
                                        program_list = [test]
                                        program_list.extend(itemlist)
                                        program_list.append(data)

                                        # non-coding segments are implemented when non completed functions are ignored
                                        # so dont put non completed funcitons in the registry.
                                        program_list = [program for program in program_list
                                                        if (self.model.remove_suffix(program) in self.model.registry or
                                                            program in self.model.emergent_functions)]

                                        gepResult, score, pickle_name = self.perform_test(program_list)

                                        # record the score no matter what, as feedback
                                        if 'results' not in test_dict:
                                            test_dict['results'] = []
                                        # result = {'offer': offernum, 'score': score, 'time': self.model.schedule.time,
                                        # 'test': test, 'data': data}
                                        result = OrderedDict(
                                            [('offer', offernum), ('score', score), ('time', self.model.schedule.time),
                                             ('test', test), ('data', data)])
                                        seller = self.model.schedule.agents[buy['offers'][offernum]["agent"]]
                                        tradenum = buy['offers'][offernum]["trades"][0]
                                        seller.seller_score_notification(score, tradenum)

                                        numtests += 1
                                        cumulative_score += score
                                        test_dict['results'].append(result)

                                        if score is not None and score > test_dict['threshold']:
                                            anypass = True
                                    else:
                                        pass_all_tests = False

                            if not anypass:
                                pass_all_tests = False

                    elif gepResult is None and itemlist:
                        gepResult = self.modular_gep(itemlist)
            else:
                pass_all_tests = False
        else:
            pass_all_tests = False

        final_score = cumulative_score / numtests if numtests else 0
        results = (pass_all_tests, gepResult, final_score, pickle_name)
        return results

    def choose_partners(self):
        # for every buy trade, roll to pick who fills the slot. If a test is required, run what the agent has
        # through the test, and if it doesnt pass, nothing fills the slot this time.  If an agent buys goods from
        # an agent that does ot have them, the same thing happens. All other buys can not be redone once accepted.
        # Whatever the state of the called routine when the human accepts it is what is paid, as opposed to when the
        # buying agent accepts it. In other words, an agent can If the agent is a human, who accepts the trade,
        # money is disbursed throughout the network according to the chain of contracted prices.

        for buynum, buy in enumerate(self.message['trades']):
            if buy['type'] == 'buy' and 'offers' in buy and buy['offers'] and 'chosen' not in buy:
                weighted_choices = OrderedDict([(offernum, offer['probability'])
                                                for offernum, offer in enumerate(buy['offers'])])
                sorted_choices = sorted(weighted_choices.items(), key=lambda x: x[1], reverse=True)
                found = False
                count = 0
                while not found and count < len(sorted_choices):
                    winning_num = sorted_choices[count][0]
                    count += 1
                    pass_all_tests, gepResult, max_score, pickle_name = self.pass_all_tests(buy, winning_num)
                    if pass_all_tests:
                        found = True
                        buy['chosen'] = winning_num
                        sell = self.b[buy['offers'][winning_num]['agent']]['trades'][
                            buy['offers'][winning_num]['trades'][0]]
                        buy['price'] = self.price(buy, sell)
                        buy['code'] = gepResult
                        buy['pickle'] = self.model.pickles[pickle_name]
                        self.buyer_score_notification(max_score, buynum)
                        if 'distributes' in self.message and self.message['distributes']:
                            self.distribute_funds(buy, buynum)
                            # self.distribute_funds(buy['offers'][winning_num], buynum)

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
        return 2 * self.p['item_size'] + 3

    def trade_type(self, afloat):
        trade_type = 'stop'

        weighted_choices = OrderedDict([('buy', 1), ('construct', 1), ('sell', 1)])
        # OrderedDict needed if you are not using python 3.7 or above!
        # weighted_choices = {'buy': 1, 'construct': 1, 'sell': 1}
        previously_taken_space = self.p['chance_of_stop_codon']
        choice = self.choose_weighted(weighted_choices, afloat, previously_taken_space=previously_taken_space)
        if choice:
            trade_type = choice
        return trade_type

    def float_for_trade_type(self, trade_type):
        float_for_trade_type = random.uniform(self.p['chance_of_stop_codon'], 1.0)

        weighted_choices = OrderedDict([('buy', 1), ('construct', 1), ('sell', 1)])
        # weighted_choices = {'buy': 1, 'construct': 1, 'sell': 1}
        float_for_choice = self.float_for_weighted_choice(weighted_choices, trade_type,
                                                          previously_taken_space=self.p['chance_of_stop_codon'])
        if float_for_choice is not None:
            float_for_trade_type = float_for_choice
        return float_for_trade_type

    def hidden(self, afloat):

        weighted_choices = OrderedDict([(True, 1), (False, 1)])
        # weighted_choices = {True: 1, False: 1}
        hidden = self.choose_weighted(weighted_choices, afloat)
        return hidden

    def stop(self, afloat):
        weighted_choices = OrderedDict([(True, 1), (False, 1)])
        stop = self.choose_weighted(weighted_choices, afloat)
        return stop

    def float_for_stop(self, isStop):
        weighted_choices = OrderedDict([(True, 1), (False, 1)])
        float_for_stop = self.float_for_weighted_choice(weighted_choices, isStop)
        return float_for_stop

    def float_for_hidden(self, isHidden):
        weighted_choices = OrderedDict([(True, 1), (False, 1)])
        float_for_hidden = self.float_for_weighted_choice(weighted_choices, isHidden)
        return float_for_hidden

    def weights_for_level(self, cumulative_category):
        weights_for_level = OrderedDict()
        if not cumulative_category.endswith('_stop') and not cumulative_category == 'stop':
            cat_dict = self.retrieve_ontology_item(cumulative_category)
            for name, adict in cat_dict.items():
                if not name.startswith('_') and isinstance(adict, dict) and '_weight' in adict:
                    weights_for_level[name] = adict['_weight']

        return weights_for_level

    def parameters_set(self, cumulative_category):
        parameters_set = False
        # check if this category is at the last level, whether it has a stop condon or not
        if cumulative_category.endswith('_stop'):
            cumulative_category = cumulative_category[:-5]
        if not cumulative_category == 'stop' and not self.weights_for_level(cumulative_category):
            parameters_set = True

        return parameters_set

    def is_stochastic(self, cumulative_category):
        description = self.retrieve_ontology_item(cumulative_category)
        is_stochastic = not description['_deterministic'] if '_deterministic' in description else False
        return is_stochastic

    def stochastic_roll(self):
        n = self.p['stochastic_copies']
        weighted_choices = OrderedDict([(x, 1) for x in range(n)])
        roll = random.uniform(0, 1)
        return self.choose_weighted(weighted_choices, roll)

    def ontology_item(self, float_vec, category='ontology', include_stop=True):
        # if category is filled in, the float starts from the given category
        cumulative_category = category
        # print ('in ontology item , category')
        # print (category)
        for afloat in float_vec:
            weighted_choices = self.weights_for_level(cumulative_category)
            # print('weighted_choices')
            # print(weighted_choices)

            if not any(weighted_choices.values()):
                # You have come to the end of what is determined.
                # Now see if that is stochastic
                if self.parameters_set(cumulative_category) and self.is_stochastic(cumulative_category):
                    cumulative_category = cumulative_category + "_stochastic" + str(self.stochastic_roll())

                break
            # roll = random.uniform(0, 1)
            roll = afloat
            # print ('afloat')
            # print (afloat)
            if include_stop:
                choice = 'stop'
                previously_taken_space = self.p['chance_of_stop_codon']
            else:
                previously_taken_space = 0

            # print('previously_taken_space')
            # print(previously_taken_space)

            guess = self.choose_weighted(weighted_choices, roll, previously_taken_space=previously_taken_space)
            # print('guess')
            # print(guess)
            if guess:
                choice = guess
            if cumulative_category == 'ontology':
                cumulative_category = choice
            else:
                cumulative_category = cumulative_category + "_" + choice

        return cumulative_category

    def floats_for_ontology_item(self, ontology_item, include_stop=True, skip=0):
        # if category is filled in, the float starts from the given category
        # print("floats_for_ontology_item")
        # print ('ontology_item')
        # print (ontology_item)
        float_list = []
        category_list = ontology_item.split("_")

        cumulative_category = "ontology"
        for choice in category_list:

            # print('cumulative_category')
            # print(cumulative_category)

            weighted_choices = self.weights_for_level(cumulative_category)
            # print('weighted_choices')
            # print(weighted_choices)

            if include_stop:
                previously_taken_space = self.p['chance_of_stop_codon']
            else:
                previously_taken_space = 0

            # print('previously_taken_space')
            # print(previously_taken_space)

            float_guess = self.float_for_weighted_choice(weighted_choices, choice, previously_taken_space)
            if float_guess is None:
                float_guess = random.uniform(self.p['chance_of_stop_codon'], 1.0)
            float_list.append(float_guess)
            # print('float_guess')
            # print(float_guess)

            if cumulative_category == 'ontology':
                cumulative_category = choice
            else:
                cumulative_category = cumulative_category + "_" + choice

        floats_left = self.p['item_size'] - len(category_list)
        floatVec = np.random.uniform(low=0.0, high=1.0, size=(floats_left,))
        float_list.extend(list(floatVec))
        for i in range(skip):
            float_list.pop(0)
            float_list.append(np.random.uniform(low=0.0, high=1.0))

        # print ('len(float_list)')
        # print (len(float_list))
        # print ('self.p[item_size]')
        # print (self.p['item_size'])
        # print ('float_list')
        # print (float_list)
        return float_list

    def float_for_agi_token(self, tokens):
        afloat = (tokens - self.p["min_token_price"]) / (self.p["max_token_price"] - self.p["min_token_price"])
        if afloat > 1.0:
            afloat = 1.0
        elif afloat < 0.0:
            afloat = 0.0
        return afloat

    def agi_token(self, afloat):
        return self.p["min_token_price"] + (self.p["max_token_price"] - self.p["min_token_price"]) * afloat

    def float_for_weighted_choice(self, weighted_choices, atype, previously_taken_space=0):
        # reverse of choose weighted function
        # return null if roll returns within previously taken space

        cumulative = self.normalized_cumulative(weighted_choices)
        space = 1 - previously_taken_space

        choice_float = None
        last_weight = 0.0
        for choice, weight in cumulative.items():
            if choice == atype:
                choice_float = random.uniform(last_weight * space, weight * space)
            last_weight = weight
        if choice_float is None:
            choice_float = random.uniform(last_weight * space, 1.0)
        return choice_float

    @staticmethod
    def normalized_cumulative(weighted_choices):
        # transform to cumulative distribution
        total = 0
        for choice, weight in weighted_choices.items():
            total += weight

        normalized = OrderedDict([(choice, (weight / total)) for choice, weight in weighted_choices.items()])

        previous_weight = 0
        cumulative = OrderedDict()
        for choice, weight in normalized.items():
            cumulative[choice] = weight + previous_weight
            previous_weight += weight

        return cumulative

    def choose_weighted(self, weighted_choices, roll, previously_taken_space=0):
        # return null if roll returns within previously taken space
        cumulative = self.normalized_cumulative(weighted_choices)
        space = 1 - previously_taken_space
        chosen = None

        for choice, weight in cumulative.items():
            if roll < weight * space:
                chosen = choice
                break

        return chosen

    def convert_to_cumulative_category(self, function_name, call_depth=0):
        # this could either be a ontology item already, or an emergent function
        # it its emergent, recursively take the first item until a
        # cumulative category is reached, within the recursion limit

        name = function_name
        if name in self.model.emergent_functions:
            emergent_root = self.model.emergent_functions[name][0]
            if call_depth < self.p["recursive_trade_depth_limit"]:
                name = self.convert_to_cumulative_category(emergent_root, call_depth=call_depth + 1)
        return name

    def item_type(self, general_function):
        cumulative_category = self.convert_to_cumulative_category(general_function)
        item_type = self.item_type_pattern.search(cumulative_category)
        if item_type:
            item_type = item_type.group(1)
        return item_type

    def test_item_type(self, cumulative_category):
        item_type = self.test_item_type_pattern.search(cumulative_category)
        if item_type:
            item_type = item_type.group(1)
        return item_type

    def float_vec_to_trade_plan(self, float_vec_dont_change, mask=None):
        float_vec = copy.deepcopy(float_vec_dont_change)
        first_level = ["distributes", "initial_message", "final_message", "message_period"]

        cursor = 0
        trade_plan = OrderedDict([('type', self.__class__.__name__)])

        trade_plan['label'] = trade_plan['type'] + " Agent " + str(self.unique_id)
        if mask and "label" in mask:
            trade_plan['label'] = mask["label"] + ", " + trade_plan['label']

        for name in first_level:
            if mask and name in mask:
                trade_plan[name] = mask[name]
        # First find the sign, which is the raw float representation
        next_cursor = self.p['sign_size']
        trade_plan['sign'] = list(float_vec[cursor:next_cursor])
        if mask and "sign" in mask:
            for i in range(min(len(mask["sign"]), self.p['sign_size'])):
                trade_plan['sign'][i] = mask["sign"][i]
                float_vec[cursor + i] = mask["sign"][i]
        cursor = next_cursor

        trade_plan['trades'] = []
        cursor_before_trade_plans = cursor

        # Then each trade plan.
        for i in range(self.p['num_trade_plans']):
            trade_plan['trades'].append(dict())
            cursor = cursor_before_trade_plans + i * self.trade_size()

            # First the type
            if mask and "trades" in mask and i < len(mask['trades']) and 'type' in mask['trades'][i]:
                trade_plan['trades'][i]['type'] = mask['trades'][i]['type']
                float_vec[cursor] = self.float_for_trade_type(trade_plan['trades'][i]['type'])
            else:
                trade_plan['trades'][i]['type'] = self.trade_type(float_vec[cursor])

            # Next the sign, which is the raw float representation
            cursor += 1
            next_cursor = cursor + self.p['sign_size']
            trade_plan['trades'][i]['sign'] = list(float_vec[cursor:next_cursor])
            if mask and "trades" in mask and i < len(mask['trades']) and 'sign' in mask['trades'][i]:
                for j in range(min(len(mask['trades'][i]['sign']), self.p['sign_size'])):
                    trade_plan['trades'][i]['sign'][j] = mask['trades'][i]['sign'][j]
                    float_vec[cursor + j] = mask['trades'][i]['sign'][j]

            cursor = next_cursor

            # Next the item
            next_cursor = cursor + self.p['item_size']

            if mask and "trades" in mask and i < len(mask['trades']) and 'item' in mask['trades'][i]:
                floats_for_item = self.floats_for_ontology_item(mask['trades'][i]['item'])
                ontlist = mask['trades'][i]['item'].split("_")
                for k in range(len(ontlist)):
                    float_vec[cursor + k] = floats_for_item[k]
            trade_plan['trades'][i]['item'] = self.ontology_item(float_vec[cursor:next_cursor])

            item_type = self.item_type(trade_plan['trades'][i]['item'])
            cursor = next_cursor

            # lowest price accepted
            if mask and "trades" in mask and i < len(mask['trades']) and 'midpoint' in mask['trades'][i]:
                trade_plan['trades'][i]['midpoint'] = mask['trades'][i]['midpoint']
                float_vec[cursor] = self.float_for_agi_token(trade_plan['trades'][i]['midpoint'])
            else:
                trade_plan['trades'][i]['midpoint'] = self.agi_token(float_vec[cursor])
            cursor += 1

            # highest price accepted

            if mask and "trades" in mask and i < len(mask['trades']) and 'range' in mask['trades'][i]:
                trade_plan['trades'][i]['range'] = mask['trades'][i]['range']
                float_vec[cursor] = self.float_for_agi_token(trade_plan['trades'][i]['range'])
            else:
                trade_plan['trades'][i]['range'] = self.agi_token(float_vec[cursor])
            cursor += 1

            cursor_before_tests = cursor

            # Finally, each test that they buyer wants to have passed before he will accept the product

            trade_plan['trades'][i]['tests'] = []
            for j in range(self.p['num_tests']):
                trade_plan['trades'][i]['tests'].append(dict())
                cursor = cursor_before_tests + j * self.test_size()

                # Are we to count this and all subsequent tests?
                if (mask and "trades" in mask and i < len(mask['trades']) and 'tests' in mask['trades'][i]
                        and j < len(mask['trades'][i]['tests']) and 'stophere' in mask['trades'][i]['tests'][j]):
                    trade_plan['trades'][i]['tests'][j]['stophere'] = mask['trades'][i]['tests'][j]['stophere']
                    float_vec[cursor] = self.float_for_stop(trade_plan['trades'][i]['tests'][j]['stophere'])
                else:
                    trade_plan['trades'][i]['tests'][j]['stophere'] = self.stop(float_vec[cursor])
                cursor += 1

                # The test
                next_cursor = cursor + self.p['item_size']
                category = 'test_' + item_type

                if (mask and "trades" in mask and i < len(mask['trades']) and 'tests' in mask['trades'][i]
                        and j < len(mask['trades'][i]['tests']) and 'test' in mask['trades'][i]['tests'][j]):
                    floats_for_item = self.floats_for_ontology_item(mask['trades'][i]['tests'][j]['test'], skip=2)
                    ontlist = mask['trades'][i]['tests'][j]['test'].split("_")
                    for k in range(len(ontlist) - 1):
                        float_vec[cursor + k] = floats_for_item[k]
                trade_plan['trades'][i]['tests'][j]['test'] = self.ontology_item(float_vec[cursor:next_cursor],
                                                                                 category=category)

                cursor = next_cursor

                # The data
                next_cursor = cursor + self.p['item_size']

                if (mask and "trades" in mask and i < len(mask['trades']) and 'tests' in mask['trades'][i]
                        and j < len(mask['trades'][i]['tests']) and 'test' in mask['trades'][i]['tests'][j]):
                    floats_for_item = self.floats_for_ontology_item(mask['trades'][i]['tests'][j]['data'], skip=1)
                    ontlist = mask['trades'][i]['tests'][j]['data'].split("_")
                    for k in range(len(ontlist) - 1):
                        float_vec[cursor + k] = floats_for_item[k]
                trade_plan['trades'][i]['tests'][j]['data'] = self.ontology_item(float_vec[cursor:next_cursor],
                                                                                 category='data')
                cursor = next_cursor
                if (mask and "trades" in mask and i < len(mask['trades']) and 'tests' in mask['trades'][i]
                        and j < len(mask['trades'][i]['tests']) and 'threshold' in mask['trades'][i]['tests'][j]):
                    trade_plan['trades'][i]['tests'][j]['threshold'] = mask['trades'][i]['tests'][j]['threshold']
                    float_vec[cursor] = mask['trades'][i]['tests'][j]['threshold']
                else:
                    trade_plan['trades'][i]['tests'][j]['threshold'] = float_vec[cursor]

                cursor += 1
                if (mask and "trades" in mask and i < len(mask['trades']) and 'tests' in mask['trades'][i]
                        and j < len(mask['trades'][i]['tests']) and 'hidden' in mask['trades'][i]['tests'][j]):
                    trade_plan['trades'][i]['tests'][j]['hidden'] = mask['trades'][i]['tests'][j]['hidden']
                    float_vec[cursor] = self.float_for_hidden(trade_plan['trades'][i]['tests'][j]['hidden'])
                else:
                    trade_plan['trades'][i]['tests'][j]['hidden'] = self.hidden(float_vec[cursor])
                cursor += 1

        # print('trade_plan')
        # print(trade_plan)
        return trade_plan, float_vec

    def trade_plan_to_float_vec(self, trade_plan):
        # print ("trade_plan_to_float_vec")

        float_list = []

        # First translate the sign, which is the raw float representation
        float_list.extend(trade_plan['sign'])

        # Then each trade plan.
        for i in range(self.p['num_trade_plans']):
            if i >= len(trade_plan['trades']):
                floats_left = self.trade_size()
                floatVec = np.random.uniform(low=0.0, high=1.0, size=(floats_left,))
                float_list.extend(list(floatVec))
                # print ('trade plan floats')
                # print (floatVec)
            else:
                # First the type
                float_list.append(self.float_for_trade_type(trade_plan['trades'][i]['type']))

                # Next the sign, which is the raw float representation
                float_list.extend(trade_plan['trades'][i]['sign'])

                # Next the item
                float_list.extend(self.floats_for_ontology_item(trade_plan['trades'][i]['item']))

                # lowest price accepted
                low = self.float_for_agi_token(trade_plan['trades'][i]['midpoint'])
                float_list.append(low)
                # print('midpoint')
                # print(low)

                # highest price accepted
                high = self.float_for_agi_token(trade_plan['trades'][i]['range'])
                float_list.append(high)
                # print('range')
                # print(high)

                # Finally, each test that they buyer wants to have passed before he will accept the product

                for j in range(self.p['num_tests']):
                    # import pdb;
                    # pdb.set_trace()

                    if j >= len(trade_plan['trades'][i]['tests']):
                        floats_left = self.test_size()
                        floatVec = np.random.uniform(low=0.0, high=1.0, size=(floats_left,))
                        float_list.extend(list(floatVec))
                        # print('test floats')
                        # print (floatVec)
                    else:
                        # Whether to count this and all tests after this
                        float_for_stop = self.float_for_stop(trade_plan['trades'][i]['tests'][j]['stophere'])
                        float_list.append(float_for_stop)
                        # print('float_for_stop')
                        # print(float_for_stop)
                        # The test
                        floats_for_ontology_item = self.floats_for_ontology_item(
                            trade_plan['trades'][i]['tests'][j]['test'], skip=2)
                        float_list.extend(floats_for_ontology_item)
                        # print ('floats_for_ontology_item')
                        # print(floats_for_ontology_item)
                        # The data
                        floats_for_ontology_item = self.floats_for_ontology_item(
                            trade_plan['trades'][i]['tests'][j]['data'], skip=1)
                        float_list.extend(floats_for_ontology_item)
                        # print ('floats_for_ontology_item')
                        # print(floats_for_ontology_item)

                        float_list.append(trade_plan['trades'][i]['tests'][j]['threshold'])
                        # print("trade_plan['trades'][i]['tests'][j]['threshold']")
                        # print(trade_plan['trades'][i]['tests'][j]['threshold'])
                        float_for_hidden = self.float_for_hidden(trade_plan['trades'][i]['tests'][j]['hidden'])
                        float_list.append(float_for_hidden)
                        # print('float_for_hidden')
                        # print(float_for_hidden)
        # print ('len(float_list)')
        # print (len(float_list))
        # print ('self.vector_size()')
        # print (self.vector_size())
        # print (float_list)
        return np.asarray(float_list)

    def blank_message(self):
        trade_plan = OrderedDict([('type', self.__class__.__name__)])
        trade_plan['label'] = trade_plan['type'] + " Agent " + str(self.unique_id)

        # First find the sign, which is the raw float representation
        trade_plan['sign'] = [0.0] * self.p['sign_size']
        trade_plan['trades'] = []
        trade_plan['trades'].append(dict())
        trade_plan['trades'][0]['type'] = 'stop'

        return trade_plan

    def get_bought_items(self):
        bought_items = OrderedDict()
        for trade in self.message['trades']:
            if ('code' in trade) and ('pickle' in trade) and trade['code'] and trade['pickle']:
                bought_items[trade['pickle']] = trade['code']

        return bought_items
