import nltk
import numpy as np
import pickle as pkl


# brown_tagged_sents = nltk.corpus.brown.tagged_sents(tagset='universal')
# train_set,test_set =train_test_split(brown_tagged_sents, train_size=0.80, test_size=0.20, random_state = 111)
# train_tagged_words = [ tup for sent in tqdm(train_set) for tup in sent ]


def load_object(filename):
    
    file = open(filename, "rb")
    obj = pkl.load(file)
    file.close()
    return obj

initial = load_object("initial4.pkl")
emission = load_object("emission4.pkl")
transition = load_object("transition4.pkl")

word2id = load_object("word2id.pkl")
tag2id = load_object("tag2id.pkl")
id2tag = load_object("id2tag.pkl")



def viterbi(text):
    
    sent = nltk.word_tokenize(text)
    
    
    viterbi = np.zeros((len(tag2id), len(sent)))
    backpointer = np.zeros((len(tag2id), len(sent)))
    
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
            
    # bestpathprob = np.max(viterbi[:, -1])
    bestpathpointer = np.argmax(viterbi[:, -1])
    
    predicted_pos_sent = []
    
    for i in reversed(range(len(sent))):
        predicted_pos_sent.insert(0, bestpathpointer)
        bestpathpointer = int(backpointer[bestpathpointer][i])
    
    id_to_pos = [id2tag[i] for i in predicted_pos_sent]
    
    tagged_sent = [i+"_"+j for (i,j) in zip(sent, id_to_pos)]
    
    return(" ".join(tagged_sent))

print(viterbi("Hello world!"))




    


