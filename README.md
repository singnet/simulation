# SingularityNET simulation


This project contains the SingularityNET simulation package, an arena much like OpenAI gym in which developers can submit machine learning and reinforcement learning solutions to a problem to test their algorithms.  However, in this case, the environment is not Atari but a toy SingularityNET environment, and the task is not to play a game but to build solutions to problems out of AI services.  Similarly to OpenAI Gym, the user only needs to write the step function of the subclassed `SnetAgent`, and optionlly handle a call to `payment_notification` when their agent is paid in pretend AGI tokens.  The simulation calls the step function of agents, waiting for all agents to move before they are notified .  The agents can examine a log of what has changed in their environment, including thier own rewards (as they define them), before taking their next step.

Agents can create python programs because the representation of the python program has a gradient on which machine learning and reinforcement learning algorithms can navigate.  This gradient comes from the generalities and specifics described in an ontology of available services, in a linear representation with genetic markers to control the meanings of genes so that they form a gradient.  Agents self organize into specialist  modules with a natural market based price. More importantly, gradient comes from the representations that agents create of the offers in utility space, so that covolutionary selective pressure is concentrated on agents according to the signs they display, which come to have an emergent meaning.  An even more important source of gradient is the diverse ecosystem of solutions, where simple problems scaffold agents with the experience they need for more complex problems.

[Here's](https://docs.google.com/document/d/1ZLcE4ekemPnplHUiE1Q4sHxlFZO3MQAdkWFEUUPcN3I/edit?usp=sharing) an (outdated) spec for the project.

We wrote a SingularityNET [blog post](https://blog.singularitynet.io/singularitynets-first-simulation-is-open-to-the-community-37445cb81bc4) about the project as well.

This repository contains a Jupyter notebook [tutorial](simulation.ipynb) to learn the basics of the project.  

Please send a note to have your solutions added to the registry.  Because this is a multiple agent program, it can work in conjunction with other solutions in an ensemble.       




## Community User participation instructions:

After installing the simulation package using 
```
pip install -e simulation
```
from the simulation directory, follow these steps:


1. Subclass the `SnetAgent`, implementing the `step` and optionally the `payment_notification` methods. 

2. Optionally, create a new scenario.  The competing clusterers scenario is included.  This is done by creating/appending to the existing ontology and registry.   The file [Registry.py](simulation/Registry.py) includes artificial intelligence services for the SnetAgents to create an AI solution out of, and a registry with their curried versions. The [study.json](simulation/study.json) file contains the ontology that determines the names of the services in the registry, and all new registry programs must be entered to this file as well.

3.  Place parameters for your new agent in the [study.json](simulation/study.json) file. Parameters specific to the agent go in `agent_parameters`.

4.  Initialize your agents.  Place agents with their initial messages on the initial blackboard. This can also be read in from a log to continue at a checkpoint from another scenario (if the `SnetAgent`'s internal states that created the message has also been saved). Here is where you would include any agent that has static messages, that stay the same throughout the simulation -such as Human agents- who operate on another time scale as the simulation. However, the more typical way to add machine learning and reinforcement learning agents is to start out random.  This is done by indicating the number of agents of your new type, as well as the number of agents of any other type in the parameter `random_agents`.   Then, it is up to you to fill the message in your `SnetAgent` subclass `init` function.

5. Adjust simulation parameters, such as how many iterations of trades you want the simulation to go through, and where you want to put the logs and pickles (serialized intermediate results).

6. Run the simulation, instantiate the class `SnetSim` with the name of your modified config file (it defaults to `study.json`), and call the method `go()`.

7. Look in the log to see information such as the python solutions that the program has created, the agi tokens all the agents received, and the scores that these programs received on tests.

*************************

#  File Organization


## simulation package:


[SnetAgent.py](simulation/SnetAgent.py):

Contains the agent that any participant can subclass, in which they implement their reinforcement/machine learning solution to the problem of constructing python programs. They do this by implementing the step method, in which they post a message to a public blackboard with the trades they would like to make. If they are using machine learning methods based on vectors of floats, they may want to use the convenient `SnetAgent` method `float_vec_to_trade_plan`, which translates a vector of floats to a message.  They receive feedback by looking at the blackboard (in `self.b`), which includes their test scores and (pretend) token rewards, as well as everybody else's, as it stands after their last move.  Community users can also implement a `payment_notification` method that will be called when individual trades in their message receive (pretend) AGI tokens, so they have the complete breakdown and not just the cumulative net return as listed on the blackboard. They should either place an initial message on the blackboard per agent through the `config` file, or generate one in the `__init__` funciton. `step`, `__init__`, and optionally `payment_notification` are the only methods of `SnetAgent` that community users implement. The `step` method differs from the one in the OpenAI gym in that the simulation calls the method, and then calls all the other agents `step` message before feedback is available, which is at least by the time the simulation calls the agent's next step. Another important difference from OpenAI gym is that since the reward depends on all the other agents' messages, the user's algorithm will have to respond to a difficult changing utility space/moving fitness landscape. Soon, `step` and `payment_notification` will soon include convenient statistical collection decorators and visualization utilities, to measure qualities such as money received and the scores of programs.

[SnetSim.py](simulation/SnetSim.py):

This is the simulation that takes care of calling the agent functions, as well as storage functions like memorising and pickling.  The user instantiates this class and call `go` to run the simulation.


[SISTER.py](simulation/SISTER.py):

An example of a community user reinforcement learning submission. Contains the CMA-ES learning algorithm.  

[Exogenous.py](simulation/Exogenous.py): 

An agent that does not change its message during the simualation run, and can be made to post a predetermined message out periodically or every time.  

[Registry.py](simulation/Registry.py):

This is the place where services which the simulation uses to compose solutions are indicated, as functions. The registry is also kept here, in the form of a dictionary that points to these functions, as well as their curried versions. 

[pickleThis.py](simulation/pickleThis.py):

This is the pickle decorator that maps a tuple representation of a curried function to a pickle of the result of the run of those functions.

[\_\_init\_\_.py](simulation/__init__.py): 

File used by python installation programs to install the simulation package.

*****************

## Other folders and files :

The [data](data) directory holds data used by the competing clusterer scenario.

[competing_clusterers](competing_clusterers) is a directory of generated files: the result of running the simulation for 10 iterations, that was performed by the [simulation.ipynb](simulation.ipynb) notebook. 

[simuation.ipynb](simulation.ipynb) is the first tutorial for the `simulation` package, an excellent starting point.  It focuses on the knowledge representation and simulation mechanics.

[marketplace.ipynb](marketplace.ipynb) is the second tutorial for the `simulation` package, with learning CMA-ES SISTER agents creating python programs in a marketplace.  The emphasis is on market dynamics.

[competingClusterers.ipynb](competingClusterers.ipynb) is a baseline run of the competing clusterers, to compare their performance on a dataset. You can add different datasets and every clusterer will work on it, for the purpose of comparing your solution to the competing clusterer scenario.

[InternetResearchAgency_tweets.clustered.csv](InternetResearchAgency_tweets.clustered.csv) is the output of the competingClusterers notebook.

[study.json](study.json) is a configuration of the simulation for the competing clusterers scenario, containing simulation parameters, an intial group of messages on the blackboard for initial agents, and the ontology of services that the agents can use to construct a solution.  

[environment.yaml](environment.yaml), [environment.windows.yml](environment.windows.yml), [requirements.txt](requirements.txt) and [requirements.windows.txt](requirements.windows.txt) are the list of python packages that need installation as required by conda and pip, in both ubuntu and windows  environments.

*****************
# License  
  
This project is licensed under the MIT License - see the
[LICENSE](https://github.com/singnet/alpha-daemon/blob/master/LICENSE) file for details.
