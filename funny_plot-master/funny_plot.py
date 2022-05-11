from plotly.subplots import make_subplots
import nltk  #
import plotly
from PIL import Image
import plotly.graph_objects as go
import pandas as pd
from textblob import TextBlob
import re
from collections import Counter
from emosent import get_emoji_sentiment_rank
import spacy

nltk.download('stopwords')
pd.set_option('display.max_colwidth', 200)

"""
read sources
"""

file = 'data/trump_tweets.csv'
raw_df = pd.read_csv(file, encoding='utf-8-sig')  # read file,including emoji
clock = Image.open('data/trumpclock.jpg')
twitter = Image.open('data/trump_tweets.png')

"""
clean tweets and preprocessing
"""


def remove_links(tweet):
    # initial clean to remove url
    tweet = re.sub(r'http\S+', '', tweet)
    tweet = re.sub(r'bit.ly/\S+', '', tweet)
    tweet = tweet.strip('[link]')
    return tweet


def remove_users(tweet):
    # removes retweet and @user information
    tweet = re.sub(r'(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)
    return tweet


def replace_str(tweet):
    tweet = tweet.replace('U.S.', 'UnitedStates')
    tweet = tweet.replace('Fake News', 'FakeNews')
    tweet = tweet.replace('White House', 'WhiteHouse')
    tweet = tweet.replace('United States', 'UnitedStates')
    return tweet


my_stopwords = nltk.corpus.stopwords.words('english')
my_punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@#'


def clean_tweet(tweet):
    # cleaning function
    tweet = remove_users(tweet)
    tweet = remove_links(tweet)
    tweet = replace_str(tweet)
    tweet = re.sub(
        '[' + my_punctuation + ']+',
        ' ',
        tweet)  # strip punctuation
    tweet = re.sub(r'\s+', ' ', tweet)  # remove double spacing
    tweet = re.sub('([0-9]+)', '', tweet)  # remove numbers
    tweet_token_list = [word for word in tweet.split(' ')
                        if word not in my_stopwords]  # remove stopwords
    tweet = ' '.join(tweet_token_list)
    return tweet


def get_sentiment(cleaned_text):
    # get setiment of cleaned text
    result = TextBlob(cleaned_text)
    if result.sentiment.polarity > 0:
        return 'Positive'
    elif result.sentiment.polarity == 0:
        return 'Neutral'
    else:
        return 'Negative'


raw_df['clean_tweet'] = raw_df['text'].apply(clean_tweet)
raw_df['sentiment'] = raw_df['clean_tweet'].apply(get_sentiment)


"""
work time analysis
"""
raw_df['created_at'] = pd.to_datetime(raw_df['created_at'])
raw_df['hour'] = raw_df['created_at'].dt.hour


time_group = raw_df.groupby('hour').count()
time_group.reset_index(inplace=True)

time_group['APM'] = (time_group['hour'] >= 12).replace(
    {True: 'PM', False: 'AM'})
AM_df = time_group[time_group['APM'] == 'AM']
PM_df = time_group[time_group['APM'] == 'PM'].copy()
PM_df['hour'] = PM_df.loc[:, 'hour'] - 12
categories = [i for i in range(12)]
AM_df["hour"] = pd.Categorical(AM_df["hour"], categories=categories)
PM_df["hour"] = pd.Categorical(PM_df["hour"], categories=categories)


fig = make_subplots(rows=2,
                    cols=1,
                    specs=[[{'type': 'polar'}]] * 2,
                    vertical_spacing=0.0)
fig.add_trace(go.Barpolar(
    r=AM_df['created_at'].values,
    theta=AM_df['hour'].values,
    marker_color=["MidnightBlue"] * 12,
    opacity=0.7, name="上半场", hovertext=AM_df['created_at'].values,
    hoverinfo="text",
), row=1, col=1)
fig.add_trace(go.Barpolar(
    r=PM_df['created_at'].values,
    theta=PM_df['hour'].values,
    marker_color=["DarkRed"] * 12,
    opacity=0.7, name="下半场", hovertext=PM_df['created_at'].values,
    hoverinfo="text",
),
    row=2, col=1)

fig.add_layout_image(
    dict(
        source=clock,
        opacity=0.9,
        xref="paper", yref="paper",
        x=0, y=0.55,
        sizex=1, sizey=0.5,
        xanchor="left", yanchor="bottom",
        layer='below'
    )
)
fig.add_layout_image(
    dict(
        source=clock,
        opacity=0.9,
        xref="paper", yref="paper",
        x=0.0, y=0.05,
        sizex=1, sizey=0.5,
        xanchor="left", yanchor="bottom",
        layer='below'
    )
)
fig.add_layout_image(
    dict(
        source=twitter,
        opacity=0.9,
        xref="paper", yref="paper",
        x=0.85, y=0.75,
        sizex=0.4, sizey=0.2,
        xanchor="left", yanchor="bottom",
        layer='below'
    )
)
fig.add_layout_image(
    dict(
        source=twitter,
        opacity=0.9,
        xref="paper", yref="paper",
        x=0.85, y=0.25,
        sizex=0.4, sizey=0.2,
        xanchor="left", yanchor="bottom",
        layer='below'
    )
)

fig.update_layout(
    template=None,
    polar1=dict(
        radialaxis=dict(
            range=[
                0,
                2800],
            showticklabels=False,
            ticks='',
            showgrid=False,
            showline=False),
        angularaxis=dict(
            direction="clockwise",
            showticklabels=False,
            ticks='',
            period=12,
            type='category',
            categoryorder="array",
            categoryarray=categories,
            showgrid=False,
            showline=False),
        bgcolor='rgba(0,0,0,0)'),
    polar2=dict(
        radialaxis=dict(
            range=[
                0,
                2800],
            showticklabels=False,
            ticks='',
            showgrid=False,
            showline=False),
        angularaxis=dict(
            direction="clockwise",
            showticklabels=False,
            ticks='',
            period=12,
            type='category',
            categoryorder="array",
            categoryarray=categories,
            showgrid=False,
            showline=False),
        bgcolor='rgba(0,0,0,0)'),
    annotations=[
        dict(
            x=1.25,
            y=0.75,
            showarrow=False,
            text="梦里，开推！",
            xref="paper",
            yref="paper", font=dict(
                family="Gravitas One",
                size=22,
                color="MidnightBlue"
            )
        ),
        dict(
            x=1.25,
            y=0.22,
            showarrow=False,
            text="午饭，开推！",
            textangle=0,
            xref="paper",
            yref="paper", font=dict(
                family="Gravitas One",
                size=22,
                color="DarkRed"
            )
        )
    ],
    margin=dict(l=40, r=140, t=40, b=40),
    width=600,
    height=1200,
    plot_bgcolor='rgba(0,0,0,0)',
    showlegend=False
)
plotly.offline.plot(fig)


"""
emoji analysis
"""

emoji_set = "['\U0001F300-\U0001F5FF'|'\U0001F600-\U0001F64F'|'\U0001F680-\U0001F6FF'|'\u2600-\u26FF\u2700-\u27BF']"
r = re.findall(emoji_set, raw_df['clean_tweet'].str.cat(sep=" . "))
emoji_bow = Counter(r)
emoji_df = pd.DataFrame.from_dict(
    emoji_bow,
    orient='index',
    columns=['cnt']).reset_index()
emoji_df.sort_values(by='cnt', ascending=False, inplace=True)


def get_emoji_sentiment(emoji):
    try:
        return get_emoji_sentiment_rank(emoji)['sentiment_score']
    except KeyError:  # handle unknown emoji
        return 0.5


emoji_df['sentiment'] = emoji_df['index'].apply(get_emoji_sentiment)
emoji_df['color'] = pd.cut(emoji_df['sentiment'],
                           bins=[-0.1,
                                 0.4,
                                 0.6,
                                 1.01],
                           labels=['Darkred',
                                   'MidnightBlue',
                                   'DarkOliveGreen'])
fig = go.Figure()
top_k = 15
label = emoji_df['index'].iloc[0:top_k]
fig.add_trace(go.Bar(x=emoji_df['index'].iloc[0:top_k],
                     y=emoji_df['cnt'].iloc[0:top_k],
                     text=label,
                     orientation='v',
                     marker=dict(color=emoji_df['color'].tolist())))
emoji_pic = Image.open('data/trump_dance.png')
fig.add_layout_image(
    dict(
        source=emoji_pic,
        opacity=0.9,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        sizex=0.5, sizey=0.5,
        xanchor="left", yanchor="bottom",
        layer='below'
    )
)


fig.update_traces(textposition='inside', textfont_size=24)
fig.update_layout(
    width=600,
    height=450,
    xaxis=dict(
        visible=False,
        showgrid=False),
    yaxis=dict(
        visible=False,
        showgrid=False),
    plot_bgcolor='white',
    margin=dict(
        l=10,
        r=10,
        t=10,
        b=10),
    annotations=[
        dict(
            x=0.4,
            y=0.75,
            showarrow=False,
            text="潮人就用Emoji！",
            xref="paper",
            yref="paper",
            font=dict(
                family="Gravitas One",
                size=26,
                color="MidnightBlue"))])
plotly.offline.plot(fig)


"""
tweet content analysis
"""
raw_df['year'] = raw_df['created_at'].dt.year

pos_tweets = raw_df[raw_df.sentiment == 'Positive']
neu_tweets = raw_df[raw_df.sentiment == 'Neutral']
neg_tweets = raw_df[raw_df.sentiment == 'Negative']


def extract_propn(series, top_k=10):
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
    nlp.max_length = 2030000
    doc = nlp(series.str.cat(sep=" . "))
    tok_list = [[]]
    for tok in doc:
        tok_list.append([tok.text, tok.dep_, tok.pos_])
    tok_df = pd.DataFrame(tok_list, columns=['text', 'dep', 'pos'])
    pronouns_df = tok_df[tok_df['pos'] == 'PROPN']['text']
    bow = Counter(pronouns_df)
    count_df = pd.DataFrame.from_dict(
        bow, orient='index', columns=['cnt']).reset_index()
    count_df.sort_values(by='cnt', ascending=False, inplace=True)
    if top_k == -1:
        return count_df
    else:
        return count_df.iloc[0:top_k, :]


pos_cnt_df = extract_propn(pos_tweets['clean_tweet'])
neg_cnt_df = extract_propn(neg_tweets['clean_tweet'])
neu_cnt_df = extract_propn(neu_tweets['clean_tweet'])

pos_cnt_df['name'] = '川普我最牛'
neg_cnt_df['name'] = '小心我怼你'
neu_cnt_df['name'] = '还行吧'

cnt_df = pd.concat([pos_cnt_df, neu_cnt_df, neg_cnt_df],
                   axis=0, ignore_index=True)


pos_tweets_count = pos_tweets.groupby(pos_tweets['created_at'].dt.year).count()
pos_tweets_count['cumsum'] = pos_tweets_count['text'].cumsum()

neg_tweets_count = neg_tweets.groupby(neg_tweets['created_at'].dt.year).count()
neg_tweets_count['cumsum'] = neg_tweets_count['text'].cumsum()

neu_tweets_count = neu_tweets.groupby(neu_tweets['created_at'].dt.year).count()
neu_tweets_count['cumsum'] = neu_tweets_count['text'].cumsum()

trump = Image.open('data/warning.png')


def df_to_sankey(df, cols_tuple_list):
    df.rename(
        columns={
            'name': 'source',
            'index': 'target',
            'cnt': 'value'},
        inplace=True)
    label_set = set(df['source'].unique()) | set(df['target'].unique())
    labels = {v: k for k, v in enumerate(label_set)}
    df.replace(labels, inplace=True)
    return df, list(label_set)


t_list = [('name', 'index')]
cnt_df['color'] = cnt_df['name'].replace(
    {'川普我最牛': 'DarkOliveGreen', '还行吧': 'MidnightBlue', '小心我怼你': 'DarkRed'})

s, labels = df_to_sankey(
    cnt_df, t_list)
fig = make_subplots(rows=1, cols=2, specs=[
                    [{'type': 'xy'}, {'type': 'domain'}]],)
fig.add_trace(
    go.Sankey(
        orientation='h',
        node=dict(
            pad=10,
            thickness=1,
            line=dict(
                width=0),
            label=labels,
            groups=[]),
        link=dict(
            source=s['source'].values,
            target=s['target'].values,
            value=s['value'].values,
            color=s['color'])), row=1, col=2)

fig.add_trace(
    go.Funnel(
        orientation='v', name='川普我最牛', y=pos_tweets_count['cumsum'], x=[
            str(i) for i in pos_tweets_count.index], marker=dict(
                color='DarkOliveGreen'), textinfo="value+percent initial"), row=1, col=1)
fig.add_trace(
    go.Funnel(
        orientation='v', name='还行吧', y=neu_tweets_count['cumsum'], x=[
            str(i) for i in pos_tweets_count.index], marker=dict(
                color='MidnightBlue'), textinfo="value+percent initial"), row=1, col=1)

fig.add_trace(
    go.Funnel(
        orientation='v', name='小心我怼你', y=neg_tweets_count['cumsum'], x=[
            str(i) for i in pos_tweets_count.index], marker=dict(
                color='Darkred'), textinfo="value+percent initial"), row=1, col=1)

fig.add_layout_image(
    dict(
        source=trump,
        opacity=1,
        xref="paper", yref="paper",
        x=-0.4, y=0.2,
        sizex=0.5, sizey=0.5,
        xanchor="left", yanchor="bottom"
    )
)


fig.update_layout(margin=dict(l=320, r=20, t=120, b=20),
                  plot_bgcolor='white',
                  width=1200,
                  height=600,

                  annotations=[
    dict(
        x=-0.4,
        y=0.75,
        showarrow=False,
        text="“我拯救了一个国家，应该得诺贝尔奖”",
        xref="paper",
        yref="paper", font=dict(
            family="Gravitas One",
            size=20,
            color="MidnightBlue"
        )
    )],
)
plotly.offline.plot(fig) # done
