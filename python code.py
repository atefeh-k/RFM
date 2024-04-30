#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install squarify --default-timeout=100')


# In[3]:


pip install pandas --default-timeout=100


# In[4]:


get_ipython().system('pip install geopandas --default-timeout=100')


# In[5]:


# Importing necessary libraries for data manipulation and analysis
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import squarify

# Configuring Pandas display options for numerical output formatting
pd.options.display.float_format = '{:,.2f}'.format


# In[6]:


# Importing necessary libraries for data visulization
import geopandas as gpd
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer
from bokeh.models import HoverTool


# In[7]:


df = pd.read_csv("C:/Users/Mysterious/Desktop/data_analyst_task.csv", low_memory=False)


# In[8]:


# change the name of columns to an standard type
df.columns = [col.lower().replace(' ', '_') for col in df.columns]


# In[9]:


# Function to convert Excel serial date to datetime
def excel_date_to_datetime(excel_serial_date):
   excel_epoch = pd.Timestamp('1899-12-30')
   return excel_epoch + pd.to_timedelta(excel_serial_date, unit='D')


# In[134]:


df.head()


# In[11]:


# Convert 'created_at' column to datetime
def convert_dates(row):
    try:
        # Try to convert the normal date format
        return pd.to_datetime(row)
    except ValueError:
        try:
            # If ValueError is raised, try to convert the Excel serial date
            return excel_date_to_datetime(float(row))
        except (ValueError, TypeError):
            # If another ValueError or TypeError is raised, return the original row
            return row
df['created_at'] = df['created_at'].apply(convert_dates)
# Save the dataframe back to CSV if needed
df.to_csv("C:/Users/Mysterious/Desktop/data_analyst_task.csv", index=False)


# In[12]:


df["created_at"].max()


# In[13]:


# Analysis as of: 2024-02-01 (max order date in the dataset: 2024-01-31)
today = datetime.strptime('2024-02-01', '%Y-%m-%d')


# In[14]:


# final_price columns items * their price - total discount
df ["final_price"] = (df["items"] * df["price"] ) - (df["discount"] + df["voucher_discount"] )


# In[15]:


agg_dict1 = {
    'order_number': 'count', # Count the number of orders per user
    'created_at': 'max', # Find the most recent order date per user
    'final_price': 'sum'# Calculate the total spend per user
}
# Group the original DataFrame by 'user_id' and apply the aggregation functions
df_rfm = df.groupby('user_id').agg(agg_dict1).reset_index()
# Rename the columns of the resulting DataFrame to reflect the RFM analysis terms
df_rfm.columns = ['user_id', 'frequency', 'max_date', 'monetary']
# Calculate the 'recency' by subtracting the most recent purchase date from the current date
# and converting the result to days
df_rfm['recency'] = (today - df_rfm['max_date']).dt.days
# Drop the 'max_date' column as it's no longer needed after calculating 'recency'
df_rfm.drop(['max_date'], axis=1, inplace=True)


# In[44]:


r_labels = range(5, 0, -1)  # Assuming recency will have enough unique values for 5 bins

# Determine the number of unique bins for 'frequency' and 'monetary'
f_unique_bins = pd.qcut(df_rfm['frequency'], q=8, duplicates='drop').cat.categories
m_unique_bins = pd.qcut(df_rfm['monetary'], q=5, duplicates='drop').cat.categories

# Create labels based on the number of unique bins
f_labels = range(1, len(f_unique_bins) + 1)
m_labels = range(1, len(m_unique_bins) + 1)

# Apply qcut with the correct number of labels
df_rfm['r_score'] = pd.qcut(df_rfm['recency'], q=5, labels=r_labels).astype(int)
df_rfm['f_score'] = pd.qcut(df_rfm['frequency'], q=8, labels=f_labels, duplicates='drop').astype(int)
df_rfm['m_score'] = pd.qcut(df_rfm['monetary'], q=5, labels=m_labels, duplicates='drop').astype(int)

# Calculate the RFM score sum
df_rfm['rfm_sum'] = df_rfm['r_score'] + df_rfm['f_score'] + df_rfm['m_score']


# In[116]:


df_rfm["r_score"].value_counts()


# In[46]:


df_rfm["f_score"].value_counts()


# In[47]:


df_rfm["m_score"].value_counts()


# In[48]:


df_rfm


# In[104]:


def assign_label(df, r_rule, fm_rule, label, colname='rfm_label'):
    df.loc[(df['r_score'].between(r_rule[0], r_rule[1]))
            & (df['f_score'].between(fm_rule[0], fm_rule[1])), colname] = label
    return df
df_rfm['rfm_label'] = ''

# Adjusting the RFM segment labels based on a 4-point scale for Frequency
# Champions: Best customers who bought most recently, most often, and are heavy spenders
df_rfm = assign_label(df_rfm, (5,5), (4,4), 'champions')

# Loyal Customers: Customers who buy on a regular basis. Responsive to promotions.
df_rfm = assign_label(df_rfm, (3,4), (4,4), 'loyal customers')

# Potential Loyalist: Recent customers with average frequency.
df_rfm = assign_label(df_rfm, (4,5), (2,3), 'potential loyalist')

# New Customers: Customers who have a high overall RFM score but are not frequent shoppers.
df_rfm = assign_label(df_rfm, (5,5), (1,1), 'new customers')

# Promising: Recent shoppers, but spent a small amount.
df_rfm = assign_label(df_rfm, (4,4), (1,1), 'promising')

# Needing Attention: Above average recency, frequency, and monetary values. May not have shopped recently.
df_rfm = assign_label(df_rfm, (2,3), (2,3), 'needing attention')

# About to Sleep: Below average recency, frequency, and monetary values. Will lose them if not reactivated.
df_rfm = assign_label(df_rfm, (2,3), (1,2), 'about to sleep')

# At Risk: Shopped long ago, bought few, and spent little.
df_rfm = assign_label(df_rfm, (1,2), (2,3), 'at risk')

# Can't Lose Them: Made big purchases, and often, but haven’t returned for a long time.
df_rfm = assign_label(df_rfm, (1,2), (4,4), 'cant lose them')

# lost: Last purchase was long back, low spenders, and low number of orders.
df_rfm = assign_label(df_rfm, (1,2), (1,2), 'lost')


# In[91]:


df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Replace inf/-inf with NaN
df_rfm.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
df_rfm.dropna(inplace=True)


# In[92]:


# Assuming df_rfm is your DataFrame
colnames = ['recency', 'frequency', 'monetary']

# Check for negative values and print them
for col in colnames:
    negative_values = df_rfm[df_rfm[col] < 0]
    if not negative_values.empty:
        print(f"Negative values found in column {col}:")
        print(negative_values)

# Plotting the histograms
for col in colnames:
    fig, ax = plt.subplots(figsize=(12,3))
    sns.histplot(df_rfm[col], kde=True , bins=10, binrange=(0, df_rfm[col].quantile(0.99)))  # Use histplot with KDE
    ax.set_title(f'Distribution of {col}')
    plt.show()


# In[108]:


segments = ['loyal customers', 'lost', 'potential loyalist']
for col in colnames:
    fig, ax = plt.subplots(figsize=(12,3))
    for segment in segments:
        sns.histplot(df_rfm[df_rfm['rfm_label']==segment][col], kde=True, label=segment)
    ax.set_title(f'Distribution of {col}')
    plt.legend()
    plt.show()


# In[109]:


palette = sns.color_palette("Blues_r", n_colors=13)

for rfm_type in ['sum', 'label']:
    fig, ax = plt.subplots(figsize=(12,5))
    sns.countplot(x='rfm_'+rfm_type, data=df_rfm, palette=palette)
    ax.set_title('Number of customers in each RFM cluster (%s)' % rfm_type)
    if rfm_type == 'label':
        plt.xticks(rotation=90)
    plt.show()


# In[110]:


agg_dict2 = {
    'user_id': 'count',
    'recency': 'mean',
    'frequency': 'mean',
    'monetary': 'sum'
}

df_analysis = df_rfm.groupby('rfm_label').agg(agg_dict2).sort_values(by='recency').reset_index()
df_analysis.rename({'rfm_label': 'label', 'user_id': 'count'}, axis=1, inplace=True)
df_analysis['count_share'] = df_analysis['count'] / df_analysis['count'].sum()
df_analysis['monetary_share'] = df_analysis['monetary'] / df_analysis['monetary'].sum()
df_analysis['monetary'] = df_analysis['monetary'] / df_analysis['count']


# In[111]:


colors = ['#37BEB0', '#DBF5F0', '#41729F', '#C3E0E5', '#0C6170', '#5885AF', '#E1C340', '#274472', '#F8EA8C', '#A4E5E0', '#1848A0']

for col in ['count', 'monetary']:
    labels = df_analysis['label'] + df_analysis[col + '_share'].apply(lambda x: ' ({0:.1f}%)'.format(x*100))

    fig, ax = plt.subplots(figsize=(16,6))
    squarify.plot(sizes=df_analysis[col], label=labels, alpha=.8, color=colors)
    ax.set_title('RFM Segments of Customers (%s)' % col)
    plt.axis('off')
    plt.show()


# In[112]:


palette = sns.color_palette("coolwarm", 10)

fig, ax = plt.subplots(figsize=(12,6))
plot = sns.scatterplot(x='recency', y='frequency', data=df_analysis, hue='label', s=300, palette=palette)

for i in range(len(df_analysis)):
     plot.text(df_analysis['recency'][i]+5,
               df_analysis['frequency'][i]-0.5,
               df_analysis['label'][i],
               horizontalalignment='left',
               size='medium', color='black')

ax.set_title('Recency vs frequency of segments')
ax.get_legend().remove()
plt.show()


# In[122]:


lc_items = df.groupby('user_id').count().reset_index()
lc_items=lc_items[["user_id","order_number"]]
least_item=pd.merge(df_rfm,lc_items,on=["user_id"],how="left")


# In[123]:


least_item


# In[129]:


top=least_item[least_item["rfm_label"]=='loyal customers']["order_number"].max()
but=least_item[least_item["rfm_label"]=='loyal customers']["order_number"].min()
avg=least_item[least_item["rfm_label"]=='loyal customers']["order_number"].mean()


# In[130]:


top


# In[131]:


but


# In[132]:


avg


# In[ ]:





# In[139]:


# Filter the DataFrame for 'loyal customers'
loyal_customers = df_rfm[df_rfm['rfm_label'] == 'loyal customers']

# Merge the RFM DataFrame with the original data to get the transaction details
loyal_customers_data = pd.merge(loyal_customers, df, on='user_id')

# Calculate the average discount rate for loyal customers
loyal_customers_data['discount_rate'] = loyal_customers_data['discount'] / loyal_customers_data['price']
average_discount_rate = loyal_customers_data['discount_rate'].mean()

# Calculate the average voucher discount rate for loyal customers
loyal_customers_data['voucher_discount_rate'] = loyal_customers_data['voucher_discount'] / loyal_customers_data['price']
average_voucher_discount_rate = loyal_customers_data['voucher_discount_rate'].mean()

# Calculate the average number of items per order for loyal customers
average_items_per_order = loyal_customers_data['items'].mean()

# Calculate the average final shipping fee for loyal customers
average_final_shipping_fee = loyal_customers_data['final_shipping_fee'].mean()

# Print out the calculated metrics
print(f"Average Discount Rate for Loyal Customers: {average_discount_rate:.2f}")
print(f"Average Voucher Discount Rate for Loyal Customers: {average_voucher_discount_rate:.2f}")
print(f"Average Items per Order for Loyal Customers: {average_items_per_order:.2f}")
print(f"Average Final Shipping Fee for Loyal Customers: {average_final_shipping_fee:.2f}")


# In[140]:


# Calculate Purchase Frequency
purchase_frequency = df.groupby('user_id')['order_number'].count().mean()

# Calculate Category Preferences
category_preferences = df['main_category'].value_counts(normalize=True) * 100

# Calculate City Distribution
city_distribution = df['city'].value_counts(normalize=True) * 100

# Output the results
print(f"Average Purchase Frequency: {purchase_frequency:.2f} orders per customer")
print("\nCategory Preferences (Percentage of Total Orders):")
print(category_preferences)
print("\nCity Distribution (Percentage of Total Orders):")
print(city_distribution)


# In[163]:


# Define a function to assign segments based on RFM scores
def assign_rfm_segment(df, r_score, f_score, segment_name):
    mask = (df['r_score'].between(r_score[0], r_score[1]) &
    df['f_score'].between(f_score[0], f_score[1]))
    df.loc[mask, 'rfm_label'] = segment_name
    return df

# Apply the function to assign 'loyal customers' segment
df_rfm = assign_rfm_segment(df_rfm, (3,4), (4,4), 'loyal customers')

# Filter the DataFrame for 'loyal customers'
loyal_customers_rfm = df_rfm[df_rfm['rfm_label'] == 'loyal customers']

# Merge the RFM DataFrame with the original data to get the transaction details
loyal_customers_data = pd.merge(loyal_customers_rfm, df, on='user_id')

# Calculate Purchase Frequency for 'loyal customers'
purchase_frequency = loyal_customers_data.groupby('user_id')['order_number'].count().mean()

# Calculate Category Preferences for 'loyal customers'
category_preferences = loyal_customers_data['main_category'].value_counts(normalize=True) * 100

# Calculate City Distribution for 'loyal customers'
city_distribution = loyal_customers_data['city'].value_counts(normalize=True) * 100

# Output the results
print(f"Average Purchase Frequency for Loyal Customers: {purchase_frequency:.2f} orders per customer")
print("\nCategory Preferences for Loyal Customers (Percentage of Total Orders):")
print(category_preferences)
print("\nCity Distribution for Loyal Customers (Percentage of Total Orders):")
print(city_distribution)


# In[148]:


la_items = df.groupby('user_id').count().reset_index()
la_items=la_items[["user_id","created_at"]]
last_item=pd.merge(df_rfm,la_items,on=["user_id"],how="left")


# In[149]:


last_item


# In[151]:


top2=last_item[last_item["rfm_label"]=='lost']["created_at"].max()
but2=last_item[last_item["rfm_label"]=='lost']["created_at"].min()
avg2=last_item[last_item["rfm_label"]=='lost']["created_at"].mean()


# In[152]:


top2


# In[153]:


but2


# In[154]:


avg2


# In[164]:


# Filter the DataFrame for 'lost'
lost = df_rfm[df_rfm['rfm_label'] == 'lost']

# Merge the RFM DataFrame with the original data to get the transaction details
lost_data = pd.merge(lost, df, on='user_id')

# Calculate the average discount rate for lost
lost_data['discount_rate'] = lost_data['discount'] / lost_data['price']
average_discount_rate = lost_data['discount_rate'].mean()

# Calculate the average voucher discount rate for lost
lost_data['voucher_discount_rate'] = lost_data['voucher_discount'] / lost_data['price']
average_voucher_discount_rate = lost_data['voucher_discount_rate'].mean()

# Calculate the average number of items per order for lost
average_items_per_order = lost_data['items'].mean()

# Calculate the average final shipping fee for lost
average_final_shipping_fee = lost_data['final_shipping_fee'].mean()

# Print out the calculated metrics
print(f"Average Discount Rate for lost: {average_discount_rate:.2f}")
print(f"Average Voucher Discount Rate for lost: {average_voucher_discount_rate:.2f}")
print(f"Average Items per Order for lost: {average_items_per_order:.2f}")
print(f"Average Final Shipping Fee for lost: {average_final_shipping_fee:.2f}")

# You can also explore other metrics such as purchase frequency, category preferences, and city distribution


# In[165]:


# Define a function to assign segments based on RFM scores
def assign_rfm_segment(df, r_score, f_score, segment_name):
    mask = (df['r_score'].between(r_score[0], r_score[1]) &
    df['f_score'].between(f_score[0], f_score[1]))
    df.loc[mask, 'rfm_label'] = segment_name
    return df

# Apply the function to assign 'lost' segment
df_rfm = assign_rfm_segment(df_rfm, (3,4), (4,4), 'lost')

# Filter the DataFrame for 'lost'
lost_rfm = df_rfm[df_rfm['rfm_label'] == 'lost']

# Merge the RFM DataFrame with the original data to get the transaction details
lost_data = pd.merge(lost_rfm, df, on='user_id')

# Calculate Purchase Frequency for 'lost'
purchase_frequency = lost_data.groupby('user_id')['order_number'].count().mean()

# Calculate Category Preferences for 'lost'
category_preferences = lost_data['main_category'].value_counts(normalize=True) * 100

# Calculate City Distribution for 'lost'
city_distribution = lost_data['city'].value_counts(normalize=True) * 100

# Output the results
print(f"Average Purchase Frequency for lost: {purchase_frequency:.2f} orders per customer")
print("\nCategory Preferences for lost (Percentage of Total Orders):")
print(category_preferences)
print("\nCity Distribution for lost (Percentage of Total Orders):")
print(city_distribution)


# In[166]:


# Calculate the average discount rate for all customers
df['discount_rate'] = df['discount'] / df['price']
average_discount_rate = df['discount_rate'].mean()

# Calculate the average voucher discount rate for all customers
df['voucher_discount_rate'] = df['voucher_discount'] / df['price']
average_voucher_discount_rate = df['voucher_discount_rate'].mean()

# Calculate the average number of items per order for all customers
average_items_per_order = df['items'].mean()

# Calculate the average final shipping fee for all customers
average_final_shipping_fee = df['final_shipping_fee'].mean()

# Print out the calculated metrics for all customers
print(f"Average Discount Rate for All Customers: {average_discount_rate:.2f}")
print(f"Average Voucher Discount Rate for All Customers: {average_voucher_discount_rate:.2f}")
print(f"Average Items per Order for All Customers: {average_items_per_order:.2f}")
print(f"Average Final Shipping Fee for All Customers: {average_final_shipping_fee:.2f}")


# In[ ]:




