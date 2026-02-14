# Search Queries Anomaly Detection

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
The Search Queries Anomaly Detection project is a machine learning-based analytical tool designed to identify "outliers" in search engine performance data. By analyzing metrics such as Clicks, Impressions, Click-Through Rate (CTR), and Search Position, the project identifies queries that behave significantly differently from the norm.

The project utilizes Python and its data science ecosystem (Pandas, Plotly) to perform Exploratory Data Analysis (EDA) and leverages the Isolation Forest algorithm—an unsupervised machine learning model—to detect anomalies without needing pre-labeled data.

### Executive Summary

In the digital marketing and SEO landscape, monitoring search query performance is critical. However, manually scanning thousands of queries to find those underperforming or overperforming is inefficient. This project automates that process.

- The system processes a dataset containing search query metrics, cleans the data (such as converting string-based CTRs to numerical values), and uses statistical modeling to flag anomalies. These anomalies might represent:

- High-potential opportunities: Queries with unexpectedly high CTRs that could be further capitalized on.

- Potential issues: Queries with high impressions but very low CTRs, suggesting a mismatch between user intent and page content.

- Technical glitches: Sudden drops or spikes in performance that may indicate tracking errors or algorithm shifts.

### Goal
The objective of this analysis is to:

1. Automate Outlier Detection: Replace manual monitoring with a machine learning model (Isolation Forest) that can scale across large datasets.

2. Identify Performance Patterns: Understand the correlation between different metrics, such as how Search Position affects Impressions and Clicks.

3. Actionable Insights: Provide a clear list of "Anomalous Queries" so that marketing teams can focus their optimization efforts on specific keywords that deviate from expected performance.

4. Data Visualization: Use interactive visualizations (via Plotly) to help stakeholders see the distribution of word frequencies and the relationship between Clicks and Impressions.

5. Technical Implementation: Demonstrate a clean end-to-end pipeline in Python, from data preprocessing to model evaluation and visualization.

### Data structure and initial checks 

The dataset is present in the files inside the repository kindly look into it.

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| ---------- | ---------- | ---------- |
| Top queries | The specific search terms or keywords users typed into the search engine that led them to the website. | object | 
| Clicks | The total number of times users clicked on the website link after searching for that specific query. | int64   | 
| Impressions | The total number of times the website appeared in search results for that query, regardless of whether it was clicked. | int64   | 
| CTR | Click-Through Rate: Calculated as $(Clicks / Impressions) \times 100$. (Note: In the raw data, this is often a percentage string like "10%" and requires conversion to a float for analysis). | object | 
| Position | The average ranking of the website on the search engine results page (SERP) for that query (e.g., 1.0 is the top spot). | float64 | 


### Tools
- Python: Google Colab - Data Preparation and pre-processing, Exploratory Data Analysis, Descriptive Statistics, inferential Statistics, Data manipulation, Visualization, Feature Engineering, Text cleaning, Correlation Analysis, Anamoly detectioin using unsupervised learning algorithm namely Isolation forest from machine learning.
  
### Analysis
Python
Loading all the ;ibraries
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
import re
from collections import Counter
```
Code to choose the dimension of the output dataframe.
``` python
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
```
Loading the dataset.
```python
df = pd.read_csv("Queries.csv")
df.head()
```
<img width="711" height="224" alt="image" src="https://github.com/user-attachments/assets/dff620dd-cff1-4018-977e-36d17022fbfe" />

Checking the shape of the data
``` python
df.shape
```
<img width="125" height="34" alt="image" src="https://github.com/user-attachments/assets/a691da04-d5f1-43e4-a168-b20fb4d5ed29" />

``` python
df.info()
```
<img width="407" height="284" alt="image" src="https://github.com/user-attachments/assets/a6463fc8-9cf0-42b2-a16c-30246c8422ba" />

Chdeck on missing or nan values
```python
df.isna().sum()
```
<img width="177" height="151" alt="image" src="https://github.com/user-attachments/assets/336a3c32-0744-44c7-ae1e-3148ff1bc063" />

Preprcesssing of a feature
``` python
# Preprocessing
df['CTR'] = df['CTR'].str.rstrip('%').astype('float') / 100
```
Descriptive Statistics
``` python
df.describe()
```
<img width="536" height="324" alt="image" src="https://github.com/user-attachments/assets/4016e12d-56af-4364-95b2-2feb50786b96" />

**Insights**

- Out of 1,000, on an average 172 clicks.
- Out of 1,000, on an average 1,939 impressions.
- Out of 1,000, on an average 0.225 (22.5%) Click-Through Rate.
- The Impressions column shows a massive gap between the average (1,939) and the maximum (73,380), indicating that a few - items are getting significantly more visibility than others.
- The median (50%) for clicks is only 94, which is much lower than the mean of 172, suggesting that the average is being pulled up by a small number of very high-performing entries.
- While the average position is roughly 4, some items have ranked as high as 1 and as low as 28.5.

Text cleaning and to find the top 20 most common words in search queries.
``` python
def clean_and_split(query):
    words = re.findall(r'\b[a-zA-Z]+\b', query.lower())
    return words

word_counts = Counter()
for query in df['Top queries']:
    word_counts.update(clean_and_split(query))

word_freq_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])

fig = px.bar(word_freq_df, x='Word', y='Frequency', title='Top 20 Most Common Words in Search Queries',width=1000, height=800)
fig.show()
```
<img width="984" height="797" alt="image" src="https://github.com/user-attachments/assets/bf2ba2de-ce9a-44c2-8880-64824b3b10bf" />

Top queries by Clicks and Impressions
```python
top_queries_clicks_vis = df.nlargest(10, 'Clicks')[['Top queries', 'Clicks']]
top_queries_impressions_vis = df.nlargest(10, 'Impressions')[['Top queries', 'Impressions']]

fig_clicks = px.bar(top_queries_clicks_vis, x='Top queries', y='Clicks', title='Top Queries by Clicks',width=1000, height=800)
fig_impressions = px.bar(top_queries_impressions_vis, x='Top queries', y='Impressions', title='Top Queries by Impressions',width=1000, height=800)
fig_clicks.show()
fig_impressions.show()
```
<img width="1016" height="806" alt="image" src="https://github.com/user-attachments/assets/39d7e1ab-f27d-4854-8724-66ec19ecb52a" />

<img width="1011" height="802" alt="image" src="https://github.com/user-attachments/assets/5ec677ad-969f-4176-88f0-2d24f8c0c2f1" />

Queries with highest and lowest CTR
```python
top_ctr_vis = df.nlargest(10, 'CTR')[['Top queries', 'CTR']]
bottom_ctr_vis = df.nsmallest(10, 'CTR')[['Top queries', 'CTR']]
fig_top_ctr = px.bar(top_ctr_vis, x='Top queries', y='CTR', title='Top Queries by CTR',width=1000, height=800)
fig_bottom_ctr = px.bar(bottom_ctr_vis, x='Top queries', y='CTR', title='Bottom Queries by CTR',width=1000, height=800)
fig_top_ctr.show()
fig_bottom_ctr.show()
```
<img width="1031" height="810" alt="image" src="https://github.com/user-attachments/assets/a9388264-31e4-4d75-a17a-9d0c29950347" />

<img width="1022" height="808" alt="image" src="https://github.com/user-attachments/assets/b630ee22-37b3-4cdb-b9cd-615c95376e10" />

Correlation Analysis
```python
# Correlation matrix visualization
correlation_matrix = df[['Clicks', 'Impressions', 'CTR', 'Position']].corr()
fig_corr = px.imshow(correlation_matrix, text_auto=True, title='Correlation Matrix')
fig_corr.show()
```
<img width="969" height="403" alt="image" src="https://github.com/user-attachments/assets/d39897f3-4b78-40d3-abf7-3a13166e56bb" />

**Insights**

In this correlation matrix:

- Clicks and Impressions are positively correlated, meaning more Impressions tend to lead to more Clicks.

- Clicks and CTR have a weak positive correlation, implying that more Clicks might slightly increase the Click-Through Rate.

- Clicks and Position are weakly negatively correlated, suggesting that higher ad or page Positions may result in fewer Clicks.

- Impressions and CTR are negatively correlated, indicating that higher 

- Impressions tend to result in a lower Click-Through Rate.

- Impressions and Position are positively correlated, indicating that ads or pages in higher Positions receive more Impressions.

- CTR and Position have a strong negative correlation, meaning that higher Positions result in lower Click-Through Rates.

**Detecting Anomalies in Search Queries**

- Now, let’s detect anomalies in search queries. You can use various techniques 
for anomaly detection. A simple and effective method is the Isolation Forest algorithm, which works well with different data distributions and is efficient with large datasets

```python
from sklearn.ensemble import IsolationForest
features = df[['Clicks', 'Impressions', 'CTR', 'Position']]
iso_forest = IsolationForest(n_estimators=100, contamination=0.01)  
iso_forest.fit(features)
df['anomaly'] = iso_forest.predict(features)
anomalies = df[df['anomaly'] == -1]
```
``` python
print(anomalies[['Top queries', 'Clicks', 'Impressions', 'CTR', 'Position']])
```
<img width="629" height="227" alt="image" src="https://github.com/user-attachments/assets/d1fb5fe3-9fcf-4951-8f38-47c56e250a7e" />

- The anomalies in our search query data are not just outliers. They are indicators of potential areas for growth, optimization, and strategic focus. These anomalies are reflecting emerging trends or areas of growing interest. Staying responsive to these trends will help in maintaining and growing the website’s relevance and user engagement.

- Search Queries Anomaly Detection means identifying queries that are outliers according to their performance metrics. It is valuable for businesses to spot potential issues or opportunities, such as unexpectedly high or low CTRs.

### Insights
The analysis of the search query dataset reveals critical relationships between metrics that drive search engine performance:

- Positive Correlation (Clicks & Impressions): As expected, higher visibility (Impressions) generally leads to more Clicks. However, the strength of this relationship helps identify "missed opportunities" where impressions are high but clicks are lagging.

- Negative Correlation (CTR & Position): There is a strong inverse relationship between Search Position and Click-Through Rate (CTR). Queries in lower positions (e.g., Position 10+) see a dramatic drop-off in CTR, emphasizing the "winner-takes-most" nature of the first few search results.

- The "High Impression/Low CTR" Anomaly: The Isolation Forest model often flags queries with massive impressions but near-zero CTR. This usually indicates that while the page is ranking for a high-volume term, the content or meta-title doesn't align with what users are looking for.

- The "Niche Authority" Anomaly: Conversely, some anomalies include queries with low impressions but extremely high CTR and high search position. These are "niche wins"—specific terms where your content is the perfect match for a small, highly intent-driven audience.

### Recommendations

For "High-Opportunity" Anomalies (High Impressions, Low CTR)
    
  - Optimize Metadata: If a query has thousands of impressions but few clicks, your title tag or meta description is likely not compelling enough. Rewrite them to be more engaging or to better answer the user's intent.

  - Content Alignment Check: Ensure the landing page actually solves the query. If users see your snippet but don't click, they might feel the page is irrelevant to their specific search term.

For "Top-Performing" Anomalies (High CTR, Low Position)

  -  Boost Content Authority: If you have a high CTR despite being in a lower position (e.g., Position 8), it means users prefer your result over those above you. This is a "diamond in the rough."

  - Action: Build internal and external links to this page to move it into the Top 3. Since the CTR is already high, a move to Position 1 could lead to a massive traffic spike.

For "Declining" Anomalies (High Position, Low CTR)
      
   -  Update "Freshness": If you are in Position 1 but your CTR is lower than the site average, users may perceive your content as outdated. Update the "last modified" date and the actual content to regain trust.

- Continuous Monitoring: Anomaly detection shouldn't be a one-time task. Integrate this Python script into a monthly reporting workflow to catch "rising star" queries early.

- Seasonality Awareness: Be aware that some anomalies might be seasonal (e.g., "Python gift ideas" spiking in December). Use the model to separate permanent shifts in user behavior from temporary seasonal trends.

- Automated Alerting: Consider setting up an automated alert (via Email or Slack) when the Isolation Forest flags a high-value query as an anomaly, allowing for real-time SEO pivots.
