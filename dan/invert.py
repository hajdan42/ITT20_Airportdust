
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as scp

method="pseudo"

# "pseudo", 


#load engine location data
y=np.random.rand(5*20)
A=np.random.rand(20*5,298*5)


if(method=="pseudo"):
	x=scp.sparse.linalg.lsqr(A, y)
	print(x)

print("HI")
