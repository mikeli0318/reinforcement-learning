"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions=num_actions;
        self.q=np.random.uniform(-1,1,(num_states,num_actions));
        self.s = 0#last state
        self.a = 0#last action
        self.alpha=alpha;
        self.gamma=gamma;
        self.rar=rar;
        self.radr=radr;
        self.dyna=dyna;
        self.list=[];
        self.doDyna=dyna>0;

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if np.random.rand() < self.rar:
            action=np.random.randint(low=0,high=self.num_actions);
        else:
            action=np.argmax(self.q[s]);
        self.a=action;
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        self.q[self.s,self.a]=(1-self.alpha)*self.q[self.s,self.a]+self.alpha*(r+self.gamma*np.max(self.q[s_prime]));
        #refresh q
        if np.random.rand()<self.rar:
            action=np.random.randint(low=0,high=self.num_actions);
        else:
            action=np.argmax(self.q[s_prime]);
        #get the action




        #dyna begin
        if self.doDyna:
            self.list.append((self.s, self.a, s_prime,r));  # refresh list
            for i in range(0, self.dyna):
                tup = self.list[np.random.randint(0, len(self.list))];
                self.q[tup[0], tup[1]] = (1 - self.alpha) * self.q[tup[0], tup[1]] + self.alpha * (tup[3] + self.gamma * np.max(self.q[tup[2]]));
        #dyna end
        self.s = s_prime;
        self.a = action;
        self.rar = self.radr * self.rar;
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
