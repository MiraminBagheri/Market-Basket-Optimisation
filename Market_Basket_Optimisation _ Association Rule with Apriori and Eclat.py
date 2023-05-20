# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
n_rows= len(dataset)
n_columns= len(dataset.count(0))
for i in range(0, n_rows):
  transactions.append([str(dataset.values[i,j]) for j in range(0, n_columns)])

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)

# Visualising the results

## Displaying the first results coming directly from the output of the apriori function
results = list(rules)
results

## Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## Displaying the results non sorted (Apriori)
print(resultsinDataFrame)

## Displaying the results sorted by descending lifts (Apriori)
print('\n\n\n Displaying the results sorted by descending "Lifts" (Apriori)')
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))


resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support', 'Confidence', 'Lift'])

## Displaying the results sorted by descending "Supports" (Eclat)
print('\n\n\n Displaying the results sorted by descending "Supports" (Eclat)')
print(resultsinDataFrame.nlargest(n = 10, columns = 'Support'))