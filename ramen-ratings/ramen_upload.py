from matplotlib.ticker import FuncFormatter
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file = 'ramen-ratings.csv'
df = pd.read_csv(file)

def stars(x,pos):
    """The two args are the value and tick position."""
    return r'$\bigstar$' * int(x)

# clean data
df['Stars'].replace({'Unrated': 3}, inplace=True)
df['Stars'] = pd.to_numeric(df['Stars'])
print(df.info())
# get top 10
df_stars = df.groupby(by='Country').agg({'Stars': ['count', 'mean']})
df_stars.sort_values(by=('Stars', 'count'), ascending=False, inplace=True)
print(df_stars.head(10))
top_10 = df_stars.iloc[0:10, :]
print(top_10)


# plot
fig, ax1 = plt.subplots(figsize=(10, 6))
sns.barplot(data=top_10, x=top_10.index, y=('Stars', 'count'), ax=ax1)
ax1.set_xlabel('Country/Area')
ax1.set_ylabel('Count')
ax1.set_ylim([0, 400])
ax2 = ax1.twinx() # share the x axis
print(top_10[('Stars', 'mean')])
sns.lineplot(data=top_10, x=top_10.index, y=top_10[(
    'Stars', 'mean')], marker="*", color='k', markersize=12)
ax2.set_ylabel('Rate')
ax2.yaxis.set_major_formatter(FuncFormatter(stars))
ax2.set_ylim([0, 5])
ax1.set_title('Ramen Quantities & Ratings in Areas')
plt.tight_layout()
plt.show()
