import nltk
import numpy as np
from tqdm import tqdm
import pickle as pkl
from itertools import chain
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('brown')


def calculate_initial_probs(tagged_sentences, tag2id):
    
    initial = np.zeros((len(tag2id),))

    first_tags = []
    
    for sent in tagged_sentences:
        first_tags.append(sent[0][1])
    
    fd = nltk.FreqDist(first_tags)
    
    for tag, tagid in tag2id.items():
        initial[tagid] = fd.freq(tag)
        
    return initial


def calculate_emission_probs(tag2id, word2id, train_tagged_words):
    
    emission = np.zeros((len(tag2id), len(word2id)))
    
    for tag, tagid in tqdm(tag2id.items(), desc="Emission probs"):
        tag_list = [pair for pair in train_tagged_words if pair[1]==tag]
        tag_count = len(tag_list)
        for word, wordid in word2id.items():
            word_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
            word_tag_count = len(word_tag_list)
            emission[tagid][wordid] = word_tag_count/tag_count
    return emission


def calculate_transition_probs(tag2id, train_set):
    
    transition = np.zeros((len(tag2id), len(tag2id)))

    bigram_tags = []

    for sent in tqdm(train_set, desc="Transition bigrams"):
        bigrams = list(nltk.bigrams(sent))
        for bigram in bigrams:
            bigram_tags.append((bigram[0][1], bigram[1][1]))

    for tag1, tagid1 in tqdm(tag2id.items(), desc="Transition probs"):
        tag1_list = [pair for pair in bigram_tags if pair[0]==tag1]
        tag1_count = len(tag1_list)
        for tag2, tagid2 in tag2id.items():
            tag1_tag2_list = [pair for pair in tag1_list if pair[1]==tag2]
            tag1_tag2_count = len(tag1_tag2_list)
            transition[tagid1][tagid2] = tag1_tag2_count/tag1_count
    
    return transition


def viterbi(test_set, tag2id, word2id, initial, emission, transition):
    true_pos = []
    predicted_pos = []

    for tagged_sentence in tqdm(test_set, desc="Viterbi"):

        sent = [word for word, pos in tagged_sentence]
        true_pos_sent = [tag2id[pos] for word, pos in tagged_sentence]
        true_pos.append(true_pos_sent)
        
        viterbi = np.zeros((len(tag2id), len(sent)))
        backpointer = np.zeros((len(tag2id), len(sent)), int)
        
        for s in range(len(tag2id)):
            if(sent[0] in word2id):
                viterbi[s][0] = initial[s] * emission[s][word2id[sent[0]]]
            else:
                viterbi[s][0] = 0 
            backpointer[s][0] = -1
        
        for t in range(1, len(sent)):
            for s in range(len(tag2id)):
                temp = np.zeros((len(tag2id),))
                for s_ in range(len(tag2id)):
                    if(sent[t] in word2id):
                        temp[s_] = viterbi[s_][t-1] * transition[s_][s] * emission[s][word2id[sent[t]]]
                    else:
                        temp[s_] = 0
                  
                viterbi[s][t] = np.max(temp)
                backpointer[s][t] = np.argmax(temp)
                
        bestpathprob = np.max(viterbi[:, -1])
        bestpathpointer = np.argmax(viterbi[:, -1])
        
        predicted_pos_sent = []
        
        for i in reversed(range(len(sent))):
            predicted_pos_sent.insert(0, bestpathpointer)
            bestpathpointer = int(backpointer[bestpathpointer][i])
        
        predicted_pos.append(predicted_pos_sent)
        
    true_pos = np.array(list(chain.from_iterable(true_pos)))
    predicted_pos = np.array(list(chain.from_iterable(predicted_pos)))

    return(true_pos, predicted_pos)
    


def save_object(obj, filename):
    
    file = open(filename, "wb")
    pkl.dump(obj, file)
    file.close()
    return

def load_object(filename):
    
    file = open(filename, "rb")
    obj = pkl.load(file)
    file.close()
    return obj


if __name__ == "__main__":
    
    brown_tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')
    
    accuracies = np.zeros(5,)
    true_pos_all = []
    predicted_pos_all = []
    
    kf = KFold(n_splits=5)
    for i, (train_index, test_index) in enumerate(kf.split(brown_tagged_sents)):
        
        train_set = [brown_tagged_sents[i] for i in train_index]
        test_set = [brown_tagged_sents[i] for i in test_index]
        
        train_tagged_words = [ tup for sent in train_set for tup in sent ]
        test_tagged_words = [ tup for sent in test_set for tup in sent ]
        
        tags = {tag for word,tag in train_tagged_words}
        vocab = {word for word,tag in train_tagged_words}
        
        word2id = {w: i for i, w in enumerate(sorted(vocab))}
        tag2id = {t: i for i, t in enumerate(sorted(tags))}
        id2tag = {i: t for i, t in enumerate(sorted(tags))}
        
        initial = calculate_initial_probs(train_set, tag2id)
        save_object(initial, "initial"+str(i)+".pkl")
        
        emission = calculate_emission_probs(tag2id, word2id, train_tagged_words)
        save_object(emission, "emission"+str(i)+".pkl")
        
        transition = calculate_transition_probs(tag2id, train_set)
        save_object(transition, "transition"+str(i)+".pkl")
        
        true_pos, predicted_pos = viterbi(test_set, tag2id, word2id, initial, emission, transition)
        
        true_pos_all.append(true_pos)
        predicted_pos_all.append(predicted_pos)
        
        accuracies[i] = accuracy_score(true_pos, predicted_pos)
        
        print("Fold "+str(i)+" Accuracy: ", accuracies[i])
        
    print("Average accuracy: ", np.mean(accuracies))
    
    save_object(true_pos_all, "true_pos_all.pkl")
    save_object(predicted_pos_all, "predicted_pos_all.pkl")





    
        
        
        
        

