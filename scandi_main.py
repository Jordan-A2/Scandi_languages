from xml.dom import minidom
import string
import torch as pt
import torch.nn as nn
import random
import time
import math
import numpy as np


###############################################################################################
# Takes all Scandinavian letters and converts to index

all_letters = string.ascii_lowercase + "æøåäö"
n_letters = len(all_letters)


def letterToIndex(letter):
    return all_letters.find(letter)

###############################################################################################


###############################################################################################
# Converts a word into a tensor

def wordToTensor(word):
    tensor = pt.zeros(len(word), 1, n_letters)
    for li, letter in enumerate(word):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

###############################################################################################


###############################################################################################
# Creating the recurrent neural network
# Defining the forward step and hidden layer
# Uses the softmax function as activation function

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = pt.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return pt.zeros(1, self.hidden_size)

###############################################################################################


###############################################################################################
# Choosing a random entry in a list

def randomChoice(w):
    return w[random.randint(0, len(w) - 1)]

###############################################################################################


###############################################################################################
# Counts how long it has taken to train the the current point

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

###############################################################################################


################################################################################################
# Determines which language is the correct choice based on their scores

def categoryFromOutput(output):
    _, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

###############################################################################################


###############################################################################################
# Picks a random word from one of the three languages and returns it along with it's tensor form

def randomTrainingExample():
    category = randomChoice(all_categories)
    word = randomChoice(category_words[category])
    category_tensor = pt.tensor(
        [all_categories.index(category)], dtype=pt.long)
    word_tensor = wordToTensor(word)
    return category, word, category_tensor, word_tensor

###############################################################################################


###############################################################################################
# Trains the neural network using random words from the datasets

def train(category_tensor, word_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

###############################################################################################


###############################################################################################
# Runs a word through the RNN

def evaluate(word_tensor):
    hidden = rnn.initHidden()

    for i in range(word_tensor.size()[0]):
        output, hidden = rnn(word_tensor[i], hidden)

    return output

###############################################################################################


###############################################################################################
# Predicts what language a given word is based on it's evaluation

def predict_word(input_word, n_predictions=3):
    with pt.no_grad():
        output = evaluate(wordToTensor(input_word))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            predictions.append([value, all_categories[category_index]])
        return predictions

###############################################################################################
###############################################################################################


num_files = 99

da_p = []
for k in range(num_files):
    filename = 'da3/da (%d).xml' % (k+1)
    da_1 = minidom.parse(filename)
    da1_text = da_1.getElementsByTagName('s')
    dq = len(da1_text)
    for j in range(dq):
        dp = len(da1_text[j].childNodes)
        for i in range(dp):
            if da1_text[j].childNodes[i].nodeType == 1:
                continue
            elif da1_text[j].childNodes[i].nodeValue == '\n    ':
                continue
            elif da1_text[j].childNodes[i].nodeValue == '\n  ':
                continue
            else:
                da_p.append(da1_text[j].childNodes[i].nodeValue)


sv_p = []
for k in range(num_files):
    filename = 'sv3/sv (%d).xml' % (k+1)
    sv_1 = minidom.parse(filename)
    sv1_text = sv_1.getElementsByTagName('s')
    sq = len(sv1_text)
    for j in range(sq):
        sp = len(sv1_text[j].childNodes)
        for i in range(sp):
            if sv1_text[j].childNodes[i].nodeType == 1:
                continue
            elif sv1_text[j].childNodes[i].nodeValue == '\n    ':
                continue
            elif sv1_text[j].childNodes[i].nodeValue == '\n  ':
                continue
            else:
                sv_p.append(sv1_text[j].childNodes[i].nodeValue)


no_p = []
for k in range(num_files):
    filename = 'no3/no (%d).xml' % (k+1)
    no_1 = minidom.parse(filename)
    no1_text = no_1.getElementsByTagName('s')
    nq = len(no1_text)
    for j in range(nq):
        nop = len(no1_text[j].childNodes)
        for i in range(nop):
            if no1_text[j].childNodes[i].nodeType == 1:
                continue
            elif no1_text[j].childNodes[i].nodeValue == '\n    ':
                continue
            elif no1_text[j].childNodes[i].nodeValue == '\n  ':
                continue
            else:
                no_p.append(no1_text[j].childNodes[i].nodeValue)


da_len = len(da_p)
sv_len = len(sv_p)
no_len = len(no_p)

da_fl = []
sv_fl = []
no_fl = []

for i in range(da_len):
    da_fl.append(da_p[i].split())

for i in range(sv_len):
    sv_fl.append(sv_p[i].split())

for i in range(no_len):
    no_fl.append(no_p[i].split())

da = [item for sublist in da_fl for item in sublist]
sv = [item for sublist in sv_fl for item in sublist]
no = [item for sublist in no_fl for item in sublist]

da_l = len(da)
sv_l = len(sv)
no_l = len(no)

for i in range(da_l):
    if da[i].isalpha() == False:
        da[i] = ""
    else:
        da[i] = da[i].lower()

for i in range(sv_l):
    if sv[i].isalpha() == False:
        sv[i] = ""
    else:
        sv[i] = sv[i].lower()

for i in range(no_l):
    if no[i].isalpha() == False:
        no[i] = ""
    else:
        no[i] = no[i].lower()


filter_object_d = filter(lambda x: x != "", da)
filter_object_s = filter(lambda x: x != "", sv)
filter_object_n = filter(lambda x: x != "", no)
da = list(filter_object_d)
sv = list(filter_object_s)
no = list(filter_object_n)

###############################################################################################

#da = list(dict.fromkeys(da))
#sv = list(dict.fromkeys(sv))
#no = list(dict.fromkeys(no))

#i = set(da).intersection(set(sv))
#j = set(da).intersection(set(no))
#k = set(sv).intersection(set(no))
#ijk = i.intersection(set(no))

#aa = list(set(da).difference(ijk))
#bb = list(set(sv).difference(ijk))
#cc = list(set(no).difference(ijk))

#aaa = list(set(aa).difference(i))
#da = list(set(aaa).difference(j))

#bbb = list(set(bb).difference(i))
#sv = list(set(bbb).difference(k))

#ccc = list(set(cc).difference(j))
#no = list(set(ccc).difference(k))

###############################################################################################

#print(da[:100])
#print(sv[:100])
#print(no[:100])
#print(len(da))
#print(len(sv))
#print(len(no))


n_hidden = 24
n_categories = 3
rnn = RNN(n_letters, n_hidden, n_categories)

category_words = {'Danish': da, 'Swedish': sv, 'Norwegian': no}
all_categories = ['Danish', 'Swedish', 'Norwegian']

criterion = nn.NLLLoss()

learning_rate = 0.01


n_iters = 300000
print_every = 15000


start = time.time()

for iter in range(1, n_iters + 1):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output, loss = train(category_tensor, line_tensor)

    # Print iter number, loss, name and guess
#    if iter % print_every == 0:
#        guess, guess_i = categoryFromOutput(output)
#        correct = '✓' if guess == category else '✗ (%s)' % category
#        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters *
#                                                100, timeSince(start), loss, line, guess, correct))