###############################################################################################
# Predicts the language of the sentence by finding the mean of each word prediction in
# the sentence

import scandi_main as sm
import numpy as np

def predict(sentence):
    print('\n> %s' % sentence)
    lst = sentence.split()
    p = len(lst)
    d = []
    n = []
    s = []
    for i in range(p):
        x = sm.predict_word(lst[i])
        if x[0][1] == 'Danish':
            d.append(x[0][0])
            if x[1][1] == 'Norwegian':
                n.append(x[1][0])
                s.append(x[2][0])
            else:
                n.append(x[2][0])
                s.append(x[1][0])
        elif x[0][1] == 'Norwegian':
            n.append(x[0][0])
            if x[1][1] == 'Danish':
                d.append(x[1][0])
                s.append(x[2][0])
            else:
                d.append(x[2][0])
                s.append(x[1][0])
        else:
            s.append(x[0][0])
            if x[1][1] == 'Danish':
                d.append(x[1][0])
                n.append(x[2][0])
            else:
                d.append(x[2][0])
                n.append(x[1][0])

    d_av = np.mean(d)
    print('Danish: %s' % d_av)
    n_av = np.mean(n)
    print('Norwegian: %s' % n_av)
    s_av = np.mean(s)
    print('Swedish: %s' % s_av)

    if d_av > n_av and d_av > s_av:
        print('Conclusion: Danish')
    elif n_av > d_av and n_av > s_av:
        print('Conclusion: Norwegian')
    else:
        print('Conclusion: Swedish')

predict('jeg købte fem æbler i butikken')
predict('jeg gikk til butikken og kjøpte fem epler')
predict('jag gick till affären och köpte fem äpplen')
predict('i dag er det sophias fødselsdag og hun elsker mig så meget')
predict('på loftet sidder nissen med sin julegrød sin julegrød så god og sød han nikker og han smiler og han er så glade for julegrød er hans bedste mad')
