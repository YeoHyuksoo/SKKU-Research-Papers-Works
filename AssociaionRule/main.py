import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Market_Basket_Optimisation.csv',header = None)
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])

from apyori import apriori
rules  = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
result_detailed = []

for i in range(len(results)):
    result_detailed.append('Rule:\t' + str(results[i][0]) +
                        '\nSupport:\t' + str(results[i][1]) +
                        '\nConfidence:\t' + str(results[i][2][0][2]) +
                        '\nLift:\t' + str(results[i][2][0][3]) )

