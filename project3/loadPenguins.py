import pandas as pd
import seaborn as sns
from palmerpenguins import load_penguins

data = load_penguins()
print(data.shape)
print(data.columns)
print(data)


#Check for nan elements
print(data.isna().sum())
#Remove rows containing nan elements
data.dropna(inplace=True)
print(data.isna().sum())
print(data.shape)

body = data.loc[:, ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]
print(body.head())

penguins = data["species"]
print(penguins.head())
