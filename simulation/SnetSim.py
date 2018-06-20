import json
import os
import pickle
import sys

from boltons.cacheutils import LRU
from boltons.cacheutils import cachedmethod
from mesa import Model
from mesa.time import StagedActivation

from simulation.Registry import registry
from simulation.pickleThis import pickleThis


class SnetSim(Model):
    def __init__(self, config_path='config.json'):
        super().__init__()

        self.gepResult = None
        with open(config_path) as json_file:
            config = json.load(json_file)
        # print(json.dumps(config['ontology'], indent=2))
        self.parameters = config['parameters']
        self.blackboard = config['blackboard']
        self.ontology = config['ontology']
        self.registry = registry


        pickle_config_path = config['parameters']['output_path'] + 'pickles/' + 'index.p'

        if pickle_config_path and os.path.exists(pickle_config_path):
            with open(pickle_config_path, 'rb') as cachehandle:
                pickle_config =  pickle.load(cachehandle)
        else:
            pickle_config = {"count": 0, "pickles": {}}


        self.pickle_count = pickle_config['count']  # contains the next number for the pickle file
        self.pickles = pickle_config['pickles']

        self.resultTuple = ()
        # self.cache = LRU(max_size = 512)
        self.cache = LRU()

        # Buyers gather offers by ranking those who offer to sell the product that have a price overlap.
        # call choose partners several times to ensure that all parts that the supply chain has a
        # chance to be settled in multiple trades, or offer networks have a chance to be filled.
        # In step the agent has a chance to put out a new message given the knowledge of purchases
        # made on the last round

        stage_list = ['gather_offers',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners', 'choose_partners',
                      'step'
                      ]

        self.schedule = StagedActivation(self, stage_list=stage_list, shuffle=True, shuffle_between_stages=True)

        # Create agents

        # first initial agents then random agents
        agent_count = 0
        for message in self.blackboard:
            if message['type'] in self.parameters['agent_parameters']:
                agent_parameters = self.parameters['agent_parameters'][message['type']]
            else:
                agent_parameters = None
            a = globals()[message['type']](agent_count, self, message, agent_parameters)
            self.schedule.add(a)
            agent_count += 1

        for agent_type, n in self.parameters['random_agents'].items():
            if agent_type in self.parameters['agent_parameters']:
                agent_parameters = self.parameters['agent_parameters'][agent_type]
            else:
                agent_parameters = None
            for i in range(n):
                a = globals()[agent_type](agent_count, self, None, agent_parameters)
                self.schedule.add(a)
                agent_count += 1

    @cachedmethod('cache')
    @pickleThis
    def memoise_pickle(self, tuple_key):
        result = None
        try:
            if len(self.resultTuple):
                result = self.registry[tuple_key[0]](*self.resultTuple)()
            else:
                result = self.registry[tuple_key[0]]()
        except IOError as e:
            print ("I/O error({0})".format(e))
        except ValueError as e:
            print ("ValueError({0})".format(e))
        except AttributeError as e:
            print ("AttributeError({0})".format(e))
        except TypeError as e:
            print ("TypeError({0})".format(e))
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            raise

        return result

    def call_memoise_pickle(self, root):

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
        func_tuple = (root, carried_back)

        self.resultTuple = tuple(result_list)  # You have to set a global to memoise and pickle correctly
        result = self.memoise_pickle(func_tuple)

        return func_tuple, result

    def print_logs(self):
        filename = self.parameters['output_path'] + "logs/log" + str(self.schedule.time) + ".txt"
        with open(filename, 'w') as outfile:
            json.dump(self.blackboard, outfile)

        pickle_config_path = self.parameters['output_path'] + 'pickles/' + 'index.p'
        pickle_config = {'count':self.pickle_count, 'pickles':self.pickles}

        with open(pickle_config_path, 'wb') as outfile:
            pickle.dump(pickle_config, outfile)


    def visualize(self):
        # todo: visualize changes in price and test score and relative wealth
        pass

    def step(self):
        """Advance the model by one step."""
        self.print_logs()
        # self.visualize() after learning agents are implemented
        self.schedule.step()

    def go(self):
        for i in range(self.parameters['max_iterations']):
            self.step()

if __name__ == '__main__':

    snetsim = SnetSim()
    snetsim.go()