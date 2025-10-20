#Import cima.csv and fda.csv and mix them together
#Save the file called cima_fda.csv
import pandas as pd

# Load the data
cima = pd.read_csv('cima.csv')
fda = pd.read_csv('fda.csv')

# Print the column names of both dataframes
print(cima.columns)
print(fda.columns)

#Remove all columns but 'text' from cima
cima = cima[['text']]

#Remove all columns but 'Label Text' from fda
fda = fda[['Label Text']]

#Rename the columns to 'text'
cima.columns = ['text']
fda.columns = ['text']

#Concatenate the two dataframes mixing them
mix = pd.concat([cima, fda], ignore_index=True)

#Shuffle the rows
mix = mix.sample(frac=1).reset_index(drop=True)

#Save the file
mix.to_csv('cima_fda.csv', index=False)