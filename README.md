# Deep Reinforcement Learning Project - Navigation

This is the first project from the [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893), where we have to create a [Deep Q-Network](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and teach it to navigate (and collect bananas) in a large square world. 

![alt-text](images/banana.gif)

A reward of +1 is provided for collecting a *yellow* banana, and a reward of -1 is provided for collecting a *blue* banana. Thus, the goal is to collect as many *yellow* bananas as possible while avoiding the *blue* ones. 

The task is episodic, and the goal is to achieve an accumulated reward of, at least, **+13** over 100 episodes.

## The Environment

### Basic setup

This code was developed on a `conda environment` with Python 3.6. You are encouraged to use a virtual environment as well (conda, virtualenv preferably). You can install all the packages on the `requirements.txt`.

To replicate our environment, you can run the following command: 
```shell
conda create --name <env> python=3.6 --file requirements.txt
```

### Unity Environment

You can download the environment on one of the links below:

* [Linux Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OS X Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* [Windows 32-bit Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* [Windows 64-bit Environment](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Then you should unzip the file (you'll need to provide the *path* of this environment to run it on the code). 

### Explore the Environment

You can navigate through the `Navigation.ipynb` file to explore the Unity Environment and how to run an random-action agent on this task. 

