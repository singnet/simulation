# simulation

This project contains the Singularity Net simulation package, an arena much like openAI gym in that developers can submit machine learning and reinforcement learning solutions to a problem to test their algorithms.  However, in this case, the environment is not Atari but a toy Singularity Net environment, and the task is not to play a game but to build solutions to problems out of AI services.  Similarly to Open AI Gym, the user only has to write the step function of the subclassed SnetAgent, and optionlly handle a call to payment_notification when their agent is paid in pretend AGI tokens.  The simulation calls the step function of agents, waiting for all agents to move before they are notified .  The agents can examine a log of what has changed in their environment, including thier own rewards (as they define them), before taking the next step.

Agents can create python programs because the representation of the python program has a gradient on which machine learning and reinforcement learning algorithms can navigate.  This gradient comes from the generalities and specifics described in an ontology of available services, in a linear representation with genetic markers to control the meanings of genes so that they form a gradient.  Agents self organize into specialist  modules with a natural market based price. More importantly, gradient comes from the representations that agents create of the offers in utility space, so that covolutionary selective pressure is concentrated on agents according to the signs they display, which come to have an emergent meaning.  An even more important source of gradient is the diverse ecosystem of solutions, where simple problems scaffold agents with the experience they need for more complex problems.

An (outdated) spec for the project is at:  https://docs.google.com/document/d/1ZLcE4ekemPnplHUiE1Q4sHxlFZO3MQAdkWFEUUPcN3I/edit?usp=sharing

There is a Singularity Net blog about the project.

A tutorial on the simulation is in the simulation.ipynb notebook.  Another notebook, competingClusterers.ipynb, is a baseline for comparing how the cluster services behave on data outside the simulation environment.

Please send a note to have your solutions added to the registry.  Because this is a multiple agent program, it can work in conjunction with other solutions in an ensemble.       

## License  
  
This project is licensed under the MIT License - see the
[LICENSE](https://github.com/singnet/alpha-daemon/blob/master/LICENSE) file for details.