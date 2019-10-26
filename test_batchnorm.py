import pytest
import numpy as np
from layers import BatchNorm

def test1():
	bn = BatchNorm(3)
	X = np.zeros((4,3))
	Y = bn.forward(X)
	assert np.allclose(Y, np.zeros((4,3)))

def test2():
	bn = BatchNorm(3)
	bn.beta = np.ones((1,3))
	X = np.zeros((4,3))
	Y = bn.forward(X)
	assert np.allclose(Y, np.ones((4,3)))

def test3():
	bn = BatchNorm(3)
	X = np.zeros((4,3))
	Y = bn.forward(X)
	assert np.allclose(Y, np.zeros((4,3)))

def test4():
	bn = BatchNorm(3)
	X = np.array([[1,2,3], [2,5,-4]])
	Y = bn.forward(X)
	assert np.allclose(Y, np.array([[-.5,-1.5,3.5],[.5,1.5,-3.5]]))

def test5():
	bn = BatchNorm(3)
	bn.beta = np.array([[-1, 0, 1]])
	X = np.array([[1,-2,6], [3,4,2], [2,-5,-2]])
	Y = bn.forward(X, train=True)
	Ytrue = np.array([[-2,-1,5], [0,5,1], [-1,-4,-3]])
	assert np.allclose(Y, Ytrue)
	dY = np.array([[-1,3,0], [2,0,-4], [-2,-1,2]])
	dX, [(_, dbeta)] = bn.backward(dY)
	dbetatrue = np.array([[-1,2,-2]])
	assert np.allclose(dbeta, dbetatrue)
	dXtrue = np.array([[-2/3, 7/3, 2/3], [7/3, -2/3, -10/3], [-5/3, -5/3, 8/3]])
	assert np.allclose(dX, dXtrue)
