from gensim import corpora
import gensim
from matplotlib import pyplot as plt
import seaborn as sns

def topic_analysis(result):
    dictionary = corpora.Dictionary(result)
    corpus = [dictionary.doc2bow(it) for it in result]
    ldaModel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=15)

    topics = []
    for i, row_list in enumerate(ldaModel[corpus]):
        row = row_list[0] if ldaModel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                topics.append(topic_num)
                # topic_details = ldaModel.show_topic(topic_num)
                # topic_keywords = ", ".join([word for word, prop in topic_details])
                # index = str(topic_num)
                # if index in topic_trends:
                #     topic_trends[index]["numberOfTweets"] = topic_trends[index]["numberOfTweets"] + 1
                # else:
                #     topic_trends[index] = {"numberOfTweets": 1, "topic": topic_keywords}
            else:
                break
    model_topics = ldaModel.show_topics(formatted=False)
    plot(topics)
    return model_topics, topics

def plot(list):
    sns.countplot(list, color='gray')
    plt.show()
    plt.savefig("topic.png")