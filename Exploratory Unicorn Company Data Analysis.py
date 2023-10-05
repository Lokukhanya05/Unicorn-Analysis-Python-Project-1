#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd

import matplotlib as plt
import seaborn as sns


# In[7]:


df = pd.read_csv(r"C:\Users\lokuk\Downloads\Unicorn_Companies.csv")
df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.shape


# In[10]:


df.info()


# In[8]:


df['Valuation'] = df['Valuation'].str.replace('$','')
df


# In[9]:


df['Funding'] = df['Funding'].str.replace('$','')
df


# In[11]:


df['Valuation'] = df['Valuation'].str.replace('B','')
df


# In[12]:


df['Valuation']=df['Valuation'].astype(float)
df


# In[13]:


df['Funding'] = df['Funding'].str.replace('B','')
df


# In[14]:


df['Funding'] = df['Funding'].str.replace('M','')
df


# In[15]:


df['Funding']=df['Funding'].astype(float)
df


# In[ ]:





# In[16]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Check the data type of values in the "Funding" column
print(df['Funding'].dtype)


# In[17]:


import pandas as pd

df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Convert "year founded" column to datetime data type
df['Year Founded'] = pd.to_datetime(df['Year Founded'], format='%Y')

# Print the updated DataFrame
df


# In[13]:


import pandas as pd

df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Check data type of "year founded" column
print(df['Year Founded'].dtype)



# In[8]:


import pandas as pd

df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Convert "Date Joined" column to datetime data type
df['Date Joined'] = pd.to_datetime(df['Date Joined'])

# Print the updated DataFrame
df


# In[15]:


import pandas as pd

df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Convert "Date Joined" column to datetime data type
df['Date Joined'] = pd.to_datetime(df['Date Joined'])

# Print the updated DataFrame
print(df['Date Joined'].dtype)


# In[16]:


df.info()


# In[11]:


df.isnull().sum()


# In[38]:


# visualising missing values using heatmap
plt.figure(figsize = (10,6))
sns.heatmap(df.isnull(), cbar =True, cmap = 'cool')
plt.title('Visualising missing values')
plt.show()


# In[104]:


df.columns


# ## Wrangling
# 
# - I will replace the missing values on the City and Select Investors with Unknown and Imputation mode respectively.
# 
# - Add a new Column on investigation the total revenue made over the years ( Valuation - Funding) alias Gross Returns
# 
# - Convert DateJoined to a DateTime data type
# 

# In[21]:


numerical_cols = df.select_dtypes(include = ['int64','float64']).columns.tolist()
numerical_cols


# In[23]:


categorical_cols = df.select_dtypes(include = ['object', 'category']).columns.tolist()
categorical_cols


# In[24]:


for column in categorical_cols:
    print(df[column].value_counts())


# In[25]:


for column in numerical_cols:
    print(df[column].value_counts())


# In[31]:


df[['Valuation', 'Funding']].describe()


# In[75]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Calculate the summary statistics for the "Valuation" and "Funding" columns
valuation_summary = df['Valuation'].describe()
funding_summary = df['Industry'].describe()

# Print the summary statistics
print("Valuation Summary:\n", valuation_summary)
print("\nFunding Summary:\n", funding_summary)


# In[51]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Calculate the mode of the "Select Investors" column
mode = df['Select Investors'].mode()[0]

# Fill in the missing values with the mode
df['Select Investors'].fillna(mode, inplace=True)

# Print the updated DataFrame
df


# In[53]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Fill in the missing values with "unknown"
df['City'].fillna('unknown', inplace=True)

# Print the updated DataFrame
df


# In[59]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Calculate the mode of the "Select Investors" column
mode = df['Select Investors'].mode()[0]

# Fill in the missing values with the mode
df['Select Investors'].fillna(mode, inplace=True)

df['City'].fillna('unknown', inplace=True)

# Print the updated DataFrame
df.head()


# In[60]:


df.shape


# In[62]:


df['City'].isnull().sum()


# In[63]:


df['Select Investors'].isnull().sum()


# ## Exploratory Analysis
# 

# In[66]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Generate descriptive statistics for the "Valuation" and "Funding" columns
valuation_stats = df['Valuation'].describe()
funding_stats = df['Funding'].describe()

# Print the descriptive statistics
print("Valuation Statistics:\n", valuation_stats)
print("\nFunding Statistics:\n", funding_stats)


# - Most companies were founded with one billion with a frequency as high as 60 companies over a century. Though funding went up to as far as 946M giving a mean of one billion on observation the funding of many companies multipied over a short period of time (2020 to 2022) as compared to 1919 t0 2020.
# 
# - The founded companies were able to join the Unicorn companies over the years though most of them remained with a valuation of one billion giving a total of 471 companies.
# 

# In[67]:


df['Select Investors'].nunique()


# In[68]:


# top 5 investors with the most number of investments
top5_invest = df['Select Investors'].value_counts().head
top5_invest()


# - Sequoia Capital is the top Investor amongst all investing on a total of 3 companies 

# In[69]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Generate the top 5 investors by count
top5_invest = df['Select Investors'].value_counts().head()

# Create a bar chart of the top 5 investors
plt.figure(figsize=(8,6))
top5_invest.plot.bar()
plt.title('Top 5 Investors')
plt.xlabel('Investor')
plt.ylabel('Count')

# Show the plot
plt.show()


# In[70]:


# the country with the highest number of investments
unique_country = df['Country'].value_counts().sort_values(ascending =False)[:5]
unique_country


# In[71]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Get the top 5 countries by count
unique_country = df['Country'].value_counts().sort_values(ascending=False)[:5]

# Create a bar graph of the top 5 countries
plt.figure(figsize=(8,6))
unique_country.plot.bar()
plt.title('Top 5 Countries')
plt.xlabel('Country')
plt.ylabel('Count')

# Show the plot
plt.show()


# - United Staes of America has the most Investors having San Fransisco and NewYork being the leading cities where the investments are.

# In[72]:


invest_count = df.groupby('Select Investors')['Valuation'].sum().sort_values(ascending=False)[:5]
invest_count


# - on observation all select Investors put in $9B at most on these.

# In[73]:


import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Generate descriptive statistics for the numerical columns
stats = df.describe()

# Print the descriptive statistics
print(stats)


# In[125]:


invest_count = df.groupby('Industry')['Valuation'].sum().sort_values(ascending=False)[:5]
invest_count


# - Cyber security industry has the most Valuation followed by Mobile and telecommunication . Though Fintech Industry has more companies invested on

# # Visualisation

# In[126]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[134]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Convert the "Date Joined" column to datetime format
df['Date Joined'] = pd.to_datetime(df['Date Joined'])

# Group the data by "Date Joined" and count the number of companies that joined in each year
grouped_data = df.groupby(df['Date Joined'].dt.year)['Company'].count().reset_index()

# Create a line graph of "Date Joined" vs. "Companies"
plt.figure(figsize=(12,6))
plt.plot(grouped_data['Date Joined'], grouped_data['Company'], marker='o')
plt.title('Companies Joined by Year')
plt.xlabel('Year')
plt.ylabel('Number of Company')

# Show the plot
plt.show()


# In[136]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Convert the "Date Founded" column to datetime format
df['Year Founded'] = pd.to_datetime(df['Year Founded'])

# Create a scatter plot of "Date Founded" vs. "Valuation"
plt.figure(figsize=(8,6))
plt.scatter(df['Year Founded'], df['Valuation'])
plt.title('Year Founded vs. Valuation')
plt.xlabel('Year Founded')
plt.ylabel('Valuation (in billions)')

# Show the plot
plt.show()


# In[139]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Count the number of occurrences for each industry
industry_counts = df['Industry'].value_counts()

# Select the top 4 industries
top_4_industries = industry_counts.head(4)

# Create a pie chart for the top 4 industries
plt.figure(figsize=(8, 6))
plt.pie(top_4_industries, labels=top_4_industries.index, autopct='%1.1f%%')
plt.title('Top 4 Industries')

# Show the plot
plt.show()


# In[140]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Count the number of occurrences for each company and city
company_counts = df['Company'].value_counts().head(5)
city_counts = df['City'].value_counts().head(5)

# Create two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Plot the top 5 companies as a horizontal bar chart in the first subplot
ax1.barh(company_counts.index, company_counts.values)
ax1.set_title('Top 5 Companies')
ax1.set_xlabel('Number of Occurrences')

# Plot the top 5 cities as a horizontal bar chart in the second subplot
ax2.barh(city_counts.index, city_counts.values)
ax2.set_title('Top 5 Cities')
ax2.set_xlabel('Number of Occurrences')

# Adjust the layout and spacing between subplots
plt.subplots_adjust(wspace=0.4)

# Show the plot
plt.show()


# In[141]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Count the number of occurrences for each company and city
company_counts = df['Company'].value_counts().head(5)
city_counts = df['City'].value_counts().head(5)

# Create a pivot table of the top 5 companies and cities
pivot_table = df.pivot_table(index='Company', columns='City', aggfunc='size', fill_value=0)
pivot_table = pivot_table.loc[company_counts.index, city_counts.index]

# Create a heatmap of the pivot table
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues')
plt.title('Top 5 Companies and Cities')
plt.xlabel('City')
plt.ylabel('Company')

# Show the plot
plt.show()


# In[143]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Count the number of occurrences for each industry
industry_counts = df['Industry'].value_counts().head(10)

# Create a bar chart of the top 10 industries
plt.figure(figsize=(8, 6))
plt.bar(industry_counts.index, industry_counts.values)
plt.title('Top 10 Industries')
plt.xlabel('Industry')
plt.ylabel('Number of Occurrences')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Show the plot
plt.show()


# In[144]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Create a pivot table of the number of companies in each industry and continent
pivot_table = df.pivot_table(index='Industry', columns='Continent', aggfunc='size', fill_value=0)

# Create a heatmap of the pivot table
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, fmt='d', cmap='Blues')
plt.title('Number of Companies by Industry and Continent')
plt.xlabel('Continent')
plt.ylabel('Industry')

# Show the plot
plt.show()


# In[146]:


import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:\\Users\\lokuk\\Downloads\\Unicorn_Companies.csv')

# Count the number of occurrences for each company
company_counts = df['Company'].value_counts().head(5)

# Filter the DataFrame to include only the top 5 companies
df_filtered = df[df['Company'].isin(company_counts.index)]

# Create a pivot table of the number of companies founded in each year for the top 5 companies
pivot_table = df_filtered.pivot_table(index='Year Founded', columns='Company', aggfunc='size', fill_value=0)

# Create a bar chart of the number of companies founded in each year, highlighting the bars for the top 5 companies
plt.figure(figsize=(12, 6))
plt.bar(pivot_table.index, pivot_table.sum(axis=1), color='gray')
for company in company_counts.index:
    plt.bar(pivot_table.index, pivot_table[company], label=company)
plt.title('Number of Companies Founded by Year for Top 5 Companies')
plt.xlabel('Year Founded')
plt.ylabel('Number of Companies')
plt.legend()

# Show the plot
plt.show()


# In[ ]:




