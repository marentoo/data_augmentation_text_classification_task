import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# from scipy.cluster.hierarchy import dendrogram, linkage
# from scipy.spatial.distance import pdist
import nltk
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud
from nltk import ngrams
from nltk.tokenize import word_tokenize




##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
#STEP 1 - LOAD FILES
def load_and_prepare_data(train_path, validation_path, test_path, sentiment_dict_path, emotions_path):
    print("\nData Loading...\n")
    train_df = pd.read_csv(train_path, sep='\t', header=None)
    validation_df = pd.read_csv(validation_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)

    column_names = ['TEXT', 'LABELS', 'ID']
    train_df.columns = column_names;    validation_df.columns = column_names;    test_df.columns = column_names

    # full_df = pd.concat([train_df, validation_df, test_df], axis=0)  ##analysis on all
    full_df = pd.concat([train_df], axis=0)  ##analysis only on train

    total_train = len(train_df)
    total_validation = len(validation_df)
    total_test = len(test_df)
    total_number_df = len(full_df)
    
    print(f"Total Train: {total_train}")
    print(f"Total Validation: {total_validation}")
    print(f"Total Test: {total_test}")
    print(f"Total Combined: {total_number_df}")
    
    print(full_df.columns)
    print(full_df.dtypes)
    
    with open(emotions_path, 'r') as file:
        emotions_str = file.read()
    emotions = [emotion.strip() for emotion in emotions_str.split("\n")]

    # with open("sentiment_mapping.json", 'r') as json_file:
    #     sent_map = json.load(json_file) 

    # with open('ekman_mapping.json','r') as jsonfile:
    #     ekman_map = json.load(jsonfile)

    with open(sentiment_dict_path) as f:
        sentiment_dict = json.load(f)
    
    return train_df, validation_df,  test_df, full_df, sentiment_dict, emotions, total_train, total_validation, total_test, total_number_df, column_names

train_df, validation_df,  test_df, full_df, sentiment_dict, emotions, total_train, total_validation, total_test, total_number_df, column_names= \
    load_and_prepare_data("train.tsv", "dev.tsv", "test.tsv", "sentiment_dict.json", "emotions.txt")
print('\n/////////////////////////////////////////////////////////////////////////////////////////////\n')




##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
#STEP 2 - DATA PREPROCESSING
print("Look at 10 first rows of data"); print(full_df.head(10))
print('\n/////////////////////////////////////////////////////////////////////////////////////////////\n')
print("\nData preprocessing...\n")

##Labels handling 
full_df['LABELS'] = full_df['LABELS'].str.split(',')
train_df['LABELS'] = train_df['LABELS'].str.split(',')
validation_df['LABELS'] = validation_df['LABELS'].str.split(',')
test_df['LABELS'] = test_df['LABELS'].str.split(',')

#drop label 27 - neutral and last 27th, emotion
full_df = full_df[full_df['LABELS'].apply(lambda x: '27' not in x)]
emotions = emotions[:-1]

#let's add another column of emotions mapped from labels - mapping function to create the 'EMOTIONS' column
def map_labels_to_emotions(labels):
    return [emotions[int(label)] for label in labels]
full_df['EMOTIONS'] = full_df['LABELS'].apply(map_labels_to_emotions)
print('\n/////////////////////////////////////////////////////////////////////////////////////////////\n')




##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##STEP 3 - DATA ANALYSIS
print("Look at 10 first rows of data");print(full_df.head(10))
print('\n/////////////////////////////////////////////////////////////////////////////////////////////\n')
print("\nData analysis and plotting...\n")

# stats = full_df.describe()
# print(stats)
# print("")

#Additional statistic and emotions count
print(f'Total_preprocess_length: {len(full_df)}')
def compute_emotion_statistics(df):
    emotions_count = df['EMOTIONS'].explode().value_counts()

    mean_count = emotions_count.mean()
    median_emotions_count = emotions_count.median()
    mode_emotion = emotions_count.idxmax()
    mode_count = emotions_count.max()
    min_count = emotions_count.min()
    max_count = emotions_count.max()
    range_count = max_count - min_count
    print(f'Median:{median_emotions_count}')
    print(f"Mean: {mean_count:.2f}")
    print(f"Mode: {mode_emotion} ({mode_count} counts)")
    print(f"Min: {min_count}")
    print(f"Max: {max_count}")
    print(f"Range: {range_count}\n")
    print(f'Count: {emotions_count}')
    
    return mean_count, median_emotions_count, mode_emotion, mode_count, min_count, max_count, range_count, emotions_count
(mean_count, median_emotions_count, mode_emotion, mode_count, min_count, max_count, range_count, emotions_count) = compute_emotion_statistics(full_df)

def plot_emotion_distribution(df):

    print("\nPlotting distribution of emotions...\n")
    
    emotions_count = df['EMOTIONS'].explode().value_counts()
    
    emotions_count.plot(kind='barh')
    plt.xlabel('Count')
    plt.ylabel('Emotions')
    plt.title('Distribution of Emotions')
    plt.savefig('plots/distribution_count_byemotions.png')
    plt.close()
plot_emotion_distribution(df=full_df)

## Unbalanced emotions! Bottom 5 by count in each dataset
print("Inbalance classes analysis:")
def get_bottom_label_counts(df, dataset_name):
    labels_count_bottom = df["LABELS"].explode().value_counts().tail(5)
    print(f"{dataset_name}\n",labels_count_bottom)
    print("")
    return dataset_name, labels_count_bottom
labels_count_bottom_full = get_bottom_label_counts(full_df, "full")
labels_count_bottom_train = get_bottom_label_counts(train_df, "train")
labels_count_bottom_validation = get_bottom_label_counts(validation_df, "validation")
labels_count_bottom_test = get_bottom_label_counts(test_df, "test")

##count exatly how many rows grief,relief,pride, embarrassment, nervousness !!!! ? but with other emotions too (becouse two possible options - selected emotion is single for given text or one of multiple emotoin labels)
print("")
def count_label_occurrences(df, label_id, label_name):
    count = df[df['LABELS'].apply(lambda x: label_id in x)].shape[0]
    print(f'{label_name} labeled rows count: {count}')
    return count
grief_count = count_label_occurrences(full_df, '16', 'grief')
pride_count = count_label_occurrences(full_df, '21', 'pride')
relief_count = count_label_occurrences(full_df, '23', 'relief')
nervousness_count = count_label_occurrences(full_df, '19', 'nervousness')
embarrassment_count = count_label_occurrences(full_df, '12', 'embarrassment')

print(""); print("Annotation analysis:")
#Annotations number of rated emotions
#filters the DataFrame to include only those cases where the number of annotations (indicated by the length of the list in the "LABELS" column)
print(f'Length (after preprocess) of full dataframe: {len(full_df)}')
def get_annotators_number(df, num_anot):
    anottators_num = df[df["LABELS"].apply(lambda x: len(x)==num_anot)]
    print(f'Cases annotated by {num_anot} annotator(s): {len(anottators_num)}')
    return anottators_num
labels_one = get_annotators_number(full_df,1);labels_two= get_annotators_number(full_df,2);labels_three= get_annotators_number(full_df,3)
labels_four= get_annotators_number(full_df,4);labels_five= get_annotators_number(full_df,5)
#calculates the percentage of cases that received more than one annotation. 

percentage_more_annotations = (len(labels_two) + len(labels_three) + len(labels_four) + len(labels_five))/ len(full_df)
print(f'Percentage of more than one annotation: {100*percentage_more_annotations:.2f}%\n')

##plot pie chart
total_examples = len(full_df)
annotation_counts = [len(labels_one), len(labels_two), len(labels_three), len(labels_four), len(labels_five)]
def plot_annotation_distribution(total_examples, annotation_counts):
    print("Plotting pie chart...")
    
    labls = ["1 annotation", "2 annotations", "3 annotations", "4 annotations", "5 annotations"]
    percentage_annotations = [(count / total_examples) * 100 for count in annotation_counts]
    explode = (0.1, 0.1, 0.1, 0.1, 0.1)

    plt.figure(figsize=(9,9))
    plt.pie(percentage_annotations, labels=labls, autopct='%1.1f%%', explode=explode)
    plt.title("Percentage of Annotations")
    plt.legend(loc="best")
    plt.axis('equal')
    plt.savefig('plots/pie_chart_annotations.png')
    plt.close()
plot_annotation_distribution(total_examples, annotation_counts)

##plot correletion matrix
def plot_emotion_correlation_heatmap(df, sentiment_dict, emotions):
    print("Plotting emotion correlations...")
    ratings = df['EMOTIONS'].apply(lambda x: ','.join(x)).str.get_dummies(sep=',')
    corr = ratings.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sentiment_color_map = {
        "positive": "#BEECAF",
        "negative": "#94bff5",
        "ambiguous": "#FFFC9E"
    }

    sent_colors = {}
    for emotion in emotions:
        if emotion in sentiment_dict["positive"]:
            sent_colors[emotion] = sentiment_color_map["positive"]
        elif emotion in sentiment_dict["negative"]:
            sent_colors[emotion] = sentiment_color_map["negative"]
        else:
            sent_colors[emotion] = sentiment_color_map["ambiguous"]
    
    row_colors = pd.Series(corr.columns, index=corr.columns).map(sent_colors)

    fig, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=.3,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5},
        cbar_ax=ax)
    
    # Add the color labels below the heatmap
    for i, (label, color) in enumerate(zip(row_colors.index, row_colors)):
        heatmap.add_patch(plt.Rectangle((i, -0.7), 1, 1, fc=color, edgecolor='none'))
    
    plt.suptitle('Correlation Heatmap of Emotions', y=0.95, fontsize=16)
    plt.savefig('plots/correlation_heatmap_byemotions.png')

    # plt.show()
plot_emotion_correlation_heatmap(full_df, sentiment_dict, emotions)
print('\n/////////////////////////////////////////////////////////////////////////////////////////////\n')




##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##STEP 4 - DATA ANALYSIS-TEXT ANALYSIS
print("Data analysis - Text analysis..."); print("")
full_df['TEXT_LENGTH'] = full_df['TEXT'].astype(str).apply(lambda x: len(x.split()))
print(print(f'plotting example 10 rows of df:\n {full_df.head(10)}'))

max_text_length = full_df['TEXT_LENGTH'].max()
max_length_rows = full_df[full_df['TEXT_LENGTH'] == max_text_length]
print("max text token length: ", max_length_rows)

#Plot length of text
def plot_text_length_distribution(df, max_text_length):

    print("Plotting text length...\n")

    sns.displot(df['TEXT_LENGTH'])
    plt.xlim([0, max_text_length])
    plt.xlabel('Text Token Length')
    plt.savefig('plots/text_length_bycount.png')

# Plot Box plot for text length
def plot_boxplot_text_length_by_category(df, emotions):

    print("\nPlotting boxplot...\n")

    temp_df = df.explode("EMOTIONS")
    temp_df['TEXT_WORD_COUNT'] = temp_df["TEXT"].apply(lambda x: len(str(x).split()))

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="EMOTIONS", y="TEXT_WORD_COUNT", data=temp_df, order=emotions)
    plt.xlabel('Emotions')
    plt.ylabel('Text Word Count')
    plt.title('Boxplot of Text Word Count by Emotion')
    plt.xticks(rotation=45, ha='right')
    plt.savefig('plots/boxplot_text_length_bycategory.png')
    plt.close()

plot_text_length_distribution(df=full_df, max_text_length=max_text_length)
plot_boxplot_text_length_by_category(df=full_df, emotions=emotions)
print('\n/////////////////////////////////////////////////////////////////////////////////////////////\n')

##word frequency analysis
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))  # Use 'english' or any other language as needed

def generate_word_frequencies_charts(df, column, stop_words, emotions, n_top=10):

    df['TOKENIZED_TEXT'] = df[column].apply(lambda x: [word for word in x.lower().split() if word not in stop_words])
    
    emotion_word_freq = {emotion: Counter() for emotion in emotions}
    for index, row in df.iterrows():
        for emotion in row['EMOTIONS']:
            emotion_word_freq[emotion].update(row['TOKENIZED_TEXT'])
    
    combined_top_words = []
    
    for emotion, freq in emotion_word_freq.items():
        top_words = freq.most_common(n_top)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(10, 5))
        plt.bar(words, counts, color='skyblue')
        plt.title(f'Top {n_top} Words for {emotion.capitalize()} Emotion')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'plots/top_words/{emotion}_top_words.png')
        plt.close()

        for word, count in top_words:
            combined_top_words.append({'Emotion': emotion, 'Word': word, 'Frequency': count})
    
    combined_top_words_df = pd.DataFrame(combined_top_words)
    combined_top_words_df.to_csv('plots/top_words/top_words.csv', index=False)
generate_word_frequencies_charts(df=full_df, column='TEXT', stop_words=stop_words, emotions=emotions, n_top=10)

def generate_word_frequencies_and_clouds(df, column, stop_words, emotions):
    # Tokenization, lowercasing, and stop word removal
    df['TOKENIZED_TEXT'] = df[column].apply(lambda x: [word for word in x.lower().split() if word not in stop_words])

    # Overall word frequency
    overall_word_freq = Counter([word for sublist in df['TOKENIZED_TEXT'] for word in sublist])
    # Word cloud for overall frequency
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(overall_word_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Overall Word Frequency')
    plt.savefig('plots/word_clouds/overall_wordcloud.png')
    # Word frequency by emotion
    emotion_word_freq = {emotion: Counter() for emotion in emotions}
    for index, row in df.iterrows():
        for emotion in row['EMOTIONS']:
            emotion_word_freq[emotion].update(row['TOKENIZED_TEXT'])
    # Word clouds for each emotion
    for emotion, freq in emotion_word_freq.items():
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Frequency for {emotion.capitalize()}')
        plt.savefig(f'plots/word_clouds/{emotion}_wordcloud.png')
generate_word_frequencies_and_clouds(df=full_df, column='TEXT', stop_words=stop_words, emotions=emotions)


##n-grams analysis
nltk.download('punkt')
def generate_ngrams_and_save(df, column, stop_words, emotions, n=2):  # For tri-grams, change `n` to 3.
    df['TOKENIZED_TEXT'] = df[column].apply(lambda x: word_tokenize(x.lower()))
    df['TOKENIZED_TEXT'] = df['TOKENIZED_TEXT'].apply(lambda x: [word for word in x if word not in stop_words])
    
    # Initialize an empty list to collect dataframes
    ngram_dfs = []

    for emotion in emotions:
        emotion_ngrams = []
        
        for text, text_emotions in zip(df['TOKENIZED_TEXT'], df['EMOTIONS']):
            if emotion in text_emotions:
                emotion_ngrams.extend(list(ngrams(text, n)))
                
        # Count the frequency of n-grams
        ngram_counts = Counter(emotion_ngrams)
        ngram_df = pd.DataFrame(ngram_counts.most_common(), columns=['ngram', 'count'])
        ngram_df['ngram'] = ngram_df['ngram'].apply(lambda x: ' '.join(x))
        
        # Add an emotion column to the dataframe
        ngram_df['emotion'] = emotion
        
        # Append the dataframe to the list
        ngram_dfs.append(ngram_df)
        
        # Create and save a bar plot for the top n-grams for the emotion
        plt.figure(figsize=(10, 6))
        sns.barplot(x='count', y='ngram', data=ngram_df.head(10), palette='viridis')
        plt.xlabel('Frequency')
        plt.ylabel('N-gram')
        plt.title(f'Top {n}-grams for {emotion.capitalize()} Emotion')
        plt.tight_layout()
        plt.savefig(f'plots/ngrams/{emotion}_ngrams.png')
        plt.close()

    # Concatenate all dataframes and save to one combined CSV
    combined_ngram_df = pd.concat(ngram_dfs, ignore_index=True)
    combined_ngram_df.to_csv('plots/ngrams/combined_ngrams.csv', index=False)
generate_ngrams_and_save(df=full_df, column='TEXT', stop_words=stop_words, emotions=emotions, n=2)




##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##-------------------------------------------------------------------------------------------
##STEP 5 - SAVING
print('Saving results...\n')
with open("plots/result_of_analysis.txt", "w") as file:
    file.write(f'Number of train {total_train}\n')
    file.write(f'Number of validation {total_validation}\n')
    file.write(f'Number of test {total_test}\n')
    file.write(f'Number of Rows Full data {total_number_df}\n')
    file.write(f'Column names {column_names}\n')
    file.write(f'\nCount\n{emotions_count}\n')
    file.write(f'\nBottom 5 unrepresented classes {labels_count_bottom_full}\n')
    #file.write("\nBasic Statistics\n");# file.write(stats.round(4).to_string(header=False))
    file.write(f'\n\nLength (after preprocess) of full dataframe {len(full_df)}\n')
    file.write(f'Percentage of more than one annotation {100 * percentage_more_annotations:.2f}%\n')
    file.write("\nAdditional Statistics\n")
    file.write(f'Mean {mean_count:.4f}\n')
    file.write(f'Median {median_emotions_count}\n')
    file.write(f'Mode {mode_emotion} ({mode_count} counts)\n')
    file.write(f'Min {min_count}\n')
    file.write(f'Max {max_count}\n')
    file.write(f'Range {range_count}\n')
    file.write(f'Max aprox. Tokens in one sentence {max_text_length}')