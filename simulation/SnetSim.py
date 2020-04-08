import json
import os
import pickle
import sys
import re
from collections import OrderedDict
import copy

from boltons.cacheutils import LRU
from boltons.cacheutils import cachedmethod
from mesa import Model
from mesa.time import StagedActivation

from simulation.Registry import registry
from simulation.pickleThis import pickleThis
from simulation.Exogenous import Exogenous
from simulation.SISTER import SISTER


class SnetSim(Model):
    def __init__(self, study_path='study.json'):
        # Get data from config file
        with open(study_path) as json_file:
            config = json.load(json_file, object_pairs_hook=OrderedDict)
        self.parameters = config['parameters']
        super().__init__(self.parameters['seed'])
        self.blackboard = config['blackboard']
        self.ontology = config['ontology']

        # Copy config file to output folder
        outpath = config['parameters']['output_path']
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        filename = outpath + study_path
        pretty = json.dumps(config, indent=2, separators=(',', ':'))
        with open(filename, 'w') as outfile:
            outfile.write(pretty)

        # Initialize class attributes
        self.gepResult = None
        self.registry = registry
        self.reproduction_report = self.reproduction_report()
        self.emergent_functions = OrderedDict()
        self.emergent_functions_arity = OrderedDict()
        self.emergent_functions_call_number = 0
        self.stochastic_pattern = re.compile(r'_stochastic\d+')
        self.prefix_pattern = re.compile(r'^f\d+_')

        # Pickling parameters
        pickle_config_path = config['parameters']['output_path'] + 'pickles/' + 'index.p'
        if pickle_config_path and os.path.exists(pickle_config_path):
            with open(pickle_config_path, 'rb') as cachehandle:
                pickle_config = pickle.load(cachehandle)
        else:
            pickle_config = OrderedDict([("count", 0), ("pickles", OrderedDict())])
        self.pickle_count = pickle_config['count']  # contains the next number for the pickle file
        self.pickles = pickle_config['pickles']

        self.resultTuple = ()
        # self.cache = LRU(max_size = 512)
        self.cache = LRU()

        # Buyers gather offers by ranking those who offer to sell the product that have a price overlap.
        # Call `choose_partners` several times to ensure that the supply chain has a
        # chance to be settled in multiple trades, or offer networks have a chance to be filled.
        # In `step`, the agent has a chance to put out a new message given the knowledge of purchases
        # made on the last round.
        stage_list = ['step', 'gather_offers',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      ]

        self.schedule = StagedActivation(self, stage_list=stage_list, shuffle=True, shuffle_between_stages=True)

        # Create initial agents as requested in `blackboard agents`
        initial_blackboard = copy.deepcopy(self.blackboard)
        self.blackboard = []
        agent_count = 0
        for i, message in enumerate(initial_blackboard):
            if message['type'] in self.parameters['agent_parameters']:
                agent_parameters = self.parameters['agent_parameters'][message['type']]
            else:
                agent_parameters = None
            # Create as many copies of agent `i` as requested
            for _ in range(self.parameters['blackboard_agents'][i]):
                a = globals()[message['type']](agent_count, self, message, agent_parameters)
                self.schedule.add(a)
                agent_count += 1

        # Create random agents
        for agent_type, n in self.parameters['random_agents'].items():
            if agent_type in self.parameters['agent_parameters']:
                agent_parameters = self.parameters['agent_parameters'][agent_type]
            else:
                agent_parameters = None
            for i in range(n):
                a = globals()[agent_type](agent_count, self, None, agent_parameters)
                self.schedule.add(a)
                agent_count += 1

        print("Initialized SnetSim instance!")

    def remove_suffix(self, func_name):
        cut_tuple = func_name
        if func_name:
            stochastic_suffix = self.stochastic_pattern.search(func_name)
            if stochastic_suffix:
                stochastic_suffix = stochastic_suffix.group()
                cut_tuple = func_name[:-len(stochastic_suffix)]
        return cut_tuple

    @cachedmethod('cache')
    @pickleThis
    def memoise_pickle(self, tuple_key):
        result = None

        try:
            cut_tuple = self.remove_suffix(tuple_key[0])
            if len(self.resultTuple) and cut_tuple:
                result = self.registry[cut_tuple](*self.resultTuple)()
            else:
                result = self.registry[cut_tuple]()
        except IOError as e:
            print("I/O error({0})".format(e))
        except ValueError as e:
            print("ValueError({0})".format(e))
        except AttributeError as e:
            print("AttributeError({0})".format(e))
        except TypeError as e:
            print("TypeError({0})".format(e))
        except RuntimeError as e:
            print("RuntimeError({0})".format(e))
        except IndexError as e:
            print("IndexError({0})".format(e))
        except:
            print("Unexpected error:", sys.exc_info()[0])
            raise

        return result

    def remove_prefix(self, func_name):
        cut_tuple = func_name
        if func_name:
            call_number_prefix = self.prefix_pattern.search(func_name)
            if call_number_prefix:
                call_number_prefix = call_number_prefix.group()
                cut_tuple = func_name[len(call_number_prefix):]
        return cut_tuple

    def get_call_prefix(self, func_name):
        call_prefix = None
        if func_name:
            call_number_prefix = self.prefix_pattern.search(func_name)
            if call_number_prefix:
                call_prefix = call_number_prefix.group()
        return call_prefix

    def call_emergent_function(self, gep_result, root):
        print("SnetSim calling emergent function with root {0}  :  {1}".format(root, gep_result))
        self.gepResult = copy.deepcopy(gep_result)
        func_tuple = self.call_memoise_pickle(root)

        print("SnetSim called emergent function with root {0} with result {1}".format(root, func_tuple))
        return func_tuple

    def call_memoise_pickle(self, root):
        # right now, self.emergentfunctions looks like:
        # 		f1: a,b,f2,c,d
        # 		f2: e,d,f3,f,g
        # 		f3: h,i,j
        #
        # You should go through the original problem that you had in the modulargep.txt file in the singularitynet
        # directory its going to be a matter of creating a registry for a functionlist on the fly from the real
        # registry I think.
        result = None
        func_tuple = (None, None)
        if root:
            result_list = []
            func_list = []
            # argTuple = ()
            if root in self.gepResult:
                args = self.gepResult[root]
                # argTuple = tuple(args)
                for arg in args:
                    temp_func_tuple, temp_result = self.call_memoise_pickle(arg)
                    result_list.append(temp_result)
                    func_list.append(temp_func_tuple)
            carried_back = tuple(func_list)
            strip_prefix = self.remove_prefix(root)
            func_tuple = (strip_prefix, carried_back)

            self.resultTuple = tuple(result_list)  # You have to set a global to memoise and pickle correctly
            if strip_prefix is not None:
                result = self.memoise_pickle(func_tuple)

        return func_tuple, result

    def reproduction_report(self):
        path = self.parameters['output_path'] + 'reproduction_report.csv'
        file = open(path, "w")
        file.write("time;agent;label;utility;agi_tokens;buyer_score;seller_score;sign_displayed;bought_items\n")

        return file

    def print_reproduction_report_line(self, agent, utility, bought_items):
        a = self.schedule.time
        b = agent.unique_id
        c = agent.message['label']
        d = utility
        e = agent.agiTokens
        f = agent.max_buyer_score
        g = agent.max_seller_score
        h = agent.message['sign']
        i = bought_items

        self.reproduction_report.write("{0};{1};{2};{3};{4};{5};{6};{7};{8}\n".format(a, b, c, d, e, f, g, h, i))
        self.reproduction_report.flush()

    def print_logs(self):
        log_path = self.parameters['output_path'] + "logs/"
        filename = log_path + "log" + str(self.schedule.time) + ".txt"
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        pretty = json.dumps(self.blackboard, indent=2, separators=(',', ':'))
        with open(filename, 'w') as outfile:
            outfile.write(pretty)
            # json.dump(self.blackboard, outfile)

        pickle_path = self.parameters['output_path'] + 'pickles/'
        pickle_config_path = pickle_path + 'index.p'
        pickle_config = OrderedDict([('count', self.pickle_count), ('pickles', self.pickles)])

        if not os.path.exists(pickle_path):
            os.makedirs(pickle_path)
        with open(pickle_config_path, 'wb') as outfile:
            pickle.dump(pickle_config, outfile)

    def visualize(self):
        # todo: visualize changes in price and test score and relative wealth
        pass

    def step(self):
        """Advance the model by one step."""
        print("IN SnetSim step, time " + str(self.schedule.time))
        self.print_logs()
        # self.visualize() after learning agents are implemented
        self.schedule.step()

    def go(self):
        for i in range(self.parameters['max_iterations']):
            print("iteration " + str(i))
            self.step()


def main():
    snetsim = SnetSim(sys.argv[1]) if len(sys.argv) > 1 else SnetSim()
    snetsim.go()


if __name__ == '__main__':
    main()
