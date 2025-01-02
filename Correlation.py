# replace with code to load pandas, matplotlib.pyplot, and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# code to import dataset
df = pd.read_excel("/content/faculty.xlsx")
# code to preview dataset (heading)
df.head()
# code to select only columns that can be included in a correlation analysis OR drop out columns that cannot be included
df = df.drop(["facid","gender","tenure-track"], axis=1)
df.head()
# code to generate histograms on remaining 14 variables/columns. Hint: Change the layout values to accomodate 14 histograms.
df.hist(layout=(5,4), figsize=(16,24), bins=15)
plt.show
#code to create new transformed variables for n4, courserate, m2, and m4. Use the Box-Cox method.
df['n4_bc'] = stats.boxcox(df['n4'])[0]
df['courserate_bc'] = stats.boxcox(df['courserate'])[0]
df['m2_bc'] = stats.boxcox(df['m2'])[0]
df['m4_bc'] = stats.boxcox(df['m4'])[0]
#preview the headers
df.head()
#code to view histograms of transformed variables (it's okay to include the other variables as well, in which case use layout=(5,4))
df.hist(layout=(5,4), figsize=(16,24), bins=15)
plt.show()
#code to create new data subset object that drops n4, courserate, m2, and m4 (keep all transformed vars for now)
df_T = df.drop(['n4','courserate','m2','m4'], axis = 1)
#preview the new data object subset
df_T.head()
#code to create correlation matrix
cor_matrix2 = df_T.corr()
# code to create color map table (cool-warm to show positive and negative correlations)
cor_matrix2.style.background_gradient(cmap='coolwarm', axis=None, vmin=-1, vmax=1)
