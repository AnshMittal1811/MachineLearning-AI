# Downloaded from http://www.work.caltech.edu/~htlin/program/libsvm/doc/platt.py

#!/usr/bin/env python
from sys import argv
#from svm import *
from math import log, exp
#from string import atof
from random import randrange
#--[Basic Function]---------------------------------------------------------------------
#input decision_values, real_labels{1,-1}, #positive_instances, #negative_instances
#output [A,B] that minimize sigmoid likilihood
#refer to Platt's Probablistic Output for Support Vector Machines
def SigmoidTrain(deci, label, prior1=None, prior0=None):

	#Count prior0 and prior1 if needed
	if prior1==None or prior0==None:
		prior1, prior0 = 0, 0
		for i in range(len(label)):
			if label[i] > 0:
				prior1+=1
			else:
				prior0+=1
	
	#Parameter Setting
	maxiter=100	#Maximum number of iterations
	minstep=1e-10	#Minimum step taken in line search
	sigma=1e-12	#For numerically strict PD of Hessian
	eps=1e-5
	
	#Construct Target Support
	hiTarget=(prior1+1.0)/(prior1+2.0)
	loTarget=1/(prior0+2.0)
	length=prior1+prior0
	t=[]

	for i in range(length):
		if label[i] > 0:
			t.append(hiTarget)
		else:
			t.append(loTarget)

	#Initial Point and Initial Fun Value
	A,B=0.0, log((prior0+1.0)/(prior1+1.0))
	fval = 0.0

	for i in range(length):
		fApB = deci[i]*A+B
		if fApB >= 0:
			fval += t[i]*fApB + log(1+exp(-fApB))
		else:
			fval += (t[i] - 1)*fApB +log(1+exp(fApB))

	for it in range(maxiter):
		#Update Gradient and Hessian (use H' = H + sigma I)
		h11=h22=sigma #Numerically ensures strict PD
		h21=g1=g2=0.0
		for i in range(length):
			fApB = deci[i]*A+B
			if (fApB >= 0):
				p=exp(-fApB)/(1.0+exp(-fApB))
				q=1.0/(1.0+exp(-fApB))
			else:
				p=1.0/(1.0+exp(fApB))
				q=exp(fApB)/(1.0+exp(fApB))
			d2=p*q
			h11+=deci[i]*deci[i]*d2
			h22+=d2
			h21+=deci[i]*d2
			d1=t[i]-p
			g1+=deci[i]*d1
			g2+=d1

		#Stopping Criteria
		if abs(g1)<eps and abs(g2)<eps:
			break

		#Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21
		dA=-(h22*g1 - h21 * g2) / det
		dB=-(-h21*g1+ h11 * g2) / det
		gd=g1*dA+g2*dB

		#Line Search
		stepsize = 1
		while stepsize >= minstep:
			newA = A + stepsize * dA
			newB = B + stepsize * dB

			#New function value
			newf = 0.0
			for i in range(length):
				fApB = deci[i]*newA+newB
				if fApB >= 0:
					newf += t[i]*fApB + log(1+exp(-fApB))
				else:
					newf += (t[i] - 1)*fApB +log(1+exp(fApB))

			#Check sufficient decrease
			if newf < fval + 0.0001 * stepsize * gd:
				A, B, fval = newA, newB, newf
				break
			else:
				stepsize = stepsize / 2.0

		if stepsize < minstep:
			#print("line search fails",A,B,g1,g2,dA,dB,gd)
			return [A,B]

	#if it>=maxiter-1:
		#print("reaching maximal iterations",g1,g2)
	return [A,B]

#reads decision_value and Platt parameter [A,B]
#outputs predicted probability
def SigmoidPredict(deci, AB):
	A, B = AB
	fApB = deci * A + B
	if (fApB >= 0):
		return exp(-fApB)/(1.0+exp(-fApB))
	else:
		return 1.0/(1+exp(fApB))


