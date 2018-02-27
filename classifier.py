import os, re, math, sys
import numpy as np

class null_stemmer:
    def __init__(self):
        return
    
    def stem(self, word):
        return word

try:
    import nltk
    STEMMER = nltk.PorterStemmer()
    print("nltk Porter Stemmer set up")
except ImportError, e:
    STEMMER = null_stemmer()
    print("nltk not found, defaulting to no stemming")




def tokenize_into_words(line):
    line = line.lower()
    return re.findall('[a-z|A-Z]+|[0-9]+|[^0-9a-zA-Z]{1,3}', line)

def tokenize_into_trigrams(email_string):
    email = filter(lambda x : x not in ['\n',' '], email_string)
    return [email[i:i+3].lower() for i in range(0,len(email),3)]



def remove_header(email):
    if('\n\n' in email):
        end_of_header = email.index('\n\n')
        return email[end_of_header+2:]
    else:
        return ''

def get_header(email):
    if('\n\n' in email):
        end_of_header = email.index('\n\n')
        return email[:end_of_header]
    else:
        return ''

def separate_email(email):
    if('\n\n' not in email):
        return ''
    
    end_of_header = email.index('\n\n')
    body = email[end_of_header+2:]
    header = email[:end_of_header]
    
    return header, body


def stem_tokens(tokens):
    result = []
    for token in tokens:
        if token.isalnum():
            result += [str(STEMMER.stem(token))]
        else:
            result += [token]
    return result

def tokenize_into_words_and_stem(email):
    split_email = email.split()
    tokens_per_line = [tokenize_into_words(space_separated) for space_separated in split_email]
    if len(tokens_per_line) > 0:
        tokens = reduce(lambda x,y : x+y, tokens_per_line)
        tokens = stem_tokens(tokens)
    else:
        tokens = []
    return tokens

def append_frequencies(tokens, frequencies):
    for token in set(tokens):
        count = sum([token == x for x in tokens])
        if(token in frequencies):
            frequencies[token]  += count
        else:
            frequencies[token] = count
    return




def train(filenames, tokenizer, email_manipulator):
    frequencies = dict()

    i=0
    print("Progress: "),
    for file_name in filenames:
        email = open(file_name).read()
        
        email = email_manipulator(email)
            
        tokens = tokenizer(email)
        
        append_frequencies(tokens, frequencies)
        
        i+=1
        if(i%100 == 0):
            print(i),
            sys.stdout.flush()
            
    print("Done Training")

    return frequencies



def merge_frequencies(table1, table2):
    result = dict()
    
    common_keys = set(table1.keys()) & set(table2.keys())
    
    result.update(table1)
    result.update(table2)
    
    for key in common_keys:
        result[key] = table1[key] + table2[key]
        
    return result



def filter_frequency_table(spam_frequencies, ham_frequencies, factor=1):
    for key in ham_frequencies.keys():
        N_ham = ham_frequencies[key]
        if key in spam_frequencies:
            N_spam = spam_frequencies[key]
        else:
            N_spam = 0

        if N_ham + N_spam < factor:
            ham_frequencies.pop(key)

    for key in spam_frequencies.keys():
        N_spam = spam_frequencies[key]
        if key in ham_frequencies:
            N_ham = ham_frequencies[key]
        else:
            N_ham = 0

        if N_ham + N_spam < factor:
            spam_frequencies.pop(key)
    
    return


def filter_frequency_table_2(spam_frequencies, ham_frequencies):
    for key in ham_frequencies.keys():
        if key in spam_frequencies:
            prob_ham = P(ham_frequencies[key], ham_frequencies)
            prob_spam = P(spam_frequencies[key], spam_frequencies)

            to_compare = math.e**(prob_spam) / float(math.e**(prob_spam) + math.e**(prob_ham))
            if (0.45 <= to_compare) and (to_compare <= 0.55):
                ham_frequencies.pop(key)
                spam_frequencies.pop(key)
    return



def P(word, frequencies, total_words, num_unique):
    
    if(word in frequencies):
        numerator = frequencies[word] + 1./num_unique
    else:
        numerator = 1./num_unique

    denominator = total_words+1
    
    return math.log(numerator) - math.log(denominator)

def predict_tokens_as_spam(tokens, spam_frequencies, ham_frequencies, modified):

    spam_result = 0
    ham_result = 0
    
    num_unique_spam = len(spam_frequencies)
    num_unique_ham = len(ham_frequencies)
    
    num_spam_words = sum(spam_frequencies.values())
    num_ham_words = sum(ham_frequencies.values())
    
    if modified:
        for token in tokens:
            spam_result += P_modified(token, spam_frequencies, num_spam_words, num_unique_spam, len(spam_frequencies))
            ham_result += P_modified(token, ham_frequencies, num_ham_words, num_unique_ham, len(ham_frequencies))
    else:
        for token in tokens:
            spam_result += P(token, spam_frequencies, num_spam_words, num_unique_spam)
            ham_result += P(token, ham_frequencies, num_ham_words, num_unique_ham)
        
    return spam_result > ham_result


def P_modified(word, frequencies, total_words, num_unique, num_tokens):
    if(word in frequencies):
        numerator = frequencies[word] + num_tokens/float(num_unique)/float(total_words)
    else:
        numerator = num_tokens/float(num_unique)/float(total_words)
    
    denominator = total_words+1
    
    return math.log(numerator) - math.log(denominator)




def test_spam(filenames, spam_frequencies, ham_frequencies, tokenizer, email_manipulator, modified):
    spam_results = []

    print("progress: "),
    i=0
    
    for file_name in filenames:
        email = open(file_name).read()
        
        email = email_manipulator(email)
            
        tokens = tokenizer(email)

        predict_spam = predict_tokens_as_spam(tokens, spam_frequencies, ham_frequencies, modified)
        spam_results += [predict_spam]

        i+=1
        if (i%100 == 0):
            print(i),
            sys.stdout.flush()
    
    return spam_results

def test(spam_filenames, ham_filenames, spam_frequencies, ham_frequencies, tokenizer, email_manipulator, show_first=False, modified=False):
    
    print("Testing spam, "),
    spam_results = test_spam(spam_filenames, spam_frequencies, ham_frequencies, tokenizer, email_manipulator, modified)
    if show_first:
        print("\n\nFirst spam email classified as spam:")
        print("-"*300)
        email = open(spam_filenames[spam_results.index(True)]).read()
        print(remove_header(email))
        print("-"*300)
        print("\n\nFirst spam email classified as ham:")
        print("-"*300)
        email = open(spam_filenames[spam_results.index(False)]).read()
        print(remove_header(email))
        print("-"*300)
        print("\n\n")
    print("Done testing spam")
    
    print("Testing ham, "),
    ham_results = test_spam(ham_filenames, spam_frequencies, ham_frequencies, tokenizer, email_manipulator, modified)
    if show_first:
        print("\n\nFirst ham email classified as ham:")
        print("-"*300)
        email = open(ham_filenames[ham_results.index(False)]).read()
        print(remove_header(email))
        print("-"*300)
        print("\n\nFirst ham email classified as spam:")
        print("-"*300)
        email = open(ham_filenames[ham_results.index(True)]).read()
        print(remove_header(email))
        print("-"*300)
        print("\n\n")
    ham_results = map(lambda x : not x, ham_results)
    print("Done testing ham")
    
    return spam_results, ham_results


def print_results(spam_results, ham_results):
    print("\nTest Results.")
    false_negative = 1 - sum(spam_results) / float(len(spam_results))
    false_positive = 1 - sum(ham_results) / float(len(ham_results))
    error = 1- ((sum(spam_results) + sum(ham_results)) / float(len(spam_results) + len(ham_results)))
    
    num_spam_total = len(spam_results)
    num_ham_total = len(ham_results)
    
    num_spam_classified = sum(spam_results)
    num_ham_classified = sum(ham_results)
    
    print("Spam (%d in total)" % num_spam_total)
    print("\t Number correctly classified as spam: %d" % num_spam_classified)
    print("\t Number incorrectly classified as ham: %d" % (num_spam_total - num_spam_classified))
    print("\t False negative: %f%%" % (false_negative*100))
    
    print("Ham (%d in total)" % num_ham_total)
    print("\t Number correctly classified as ham: %d" % num_ham_classified)
    print("\t Number incorrectly classified as spam: %d" % (num_ham_total - num_ham_classified))
    print("\t False positive: %f%%" % (false_positive*100))
    
    print("\n Total Error: %f%%\n" % (error*100))
    
    return



def main():

    training_spam_filenames = ["training/spam/" + filename for filename in os.listdir("training/spam/")]
    training_ham_filenames = ["training/ham/" + filename for filename in os.listdir("training/ham/")]

    testing_spam_filenames = ["testing/spam/" + filename for filename in os.listdir("testing/spam/")]
    testing_ham_filenames = ["testing/ham/" + filename for filename in os.listdir("testing/ham/")]



    # # Words

    # ### No header
    print("Training: Email tokenized into words, ignore header.")
    email_manipulator = lambda email : remove_header(email)


    ham_body_stemmed_frequencies = train(training_ham_filenames, tokenize_into_words_and_stem, email_manipulator)
    spam_body_stemmed_frequencies = train(training_spam_filenames, tokenize_into_words_and_stem, email_manipulator)

    print("Testing")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_body_stemmed_frequencies, ham_body_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator,
                                     True)


    print_results(spam_results, ham_results)


    # ### Header



    email_manipulator = lambda email : get_header(email)


    print("Training: Email tokenized into words, only header.")
    ham_header_stemmed_frequencies = train(training_ham_filenames, tokenize_into_words_and_stem, email_manipulator)
    spam_header_stemmed_frequencies = train(training_spam_filenames, tokenize_into_words_and_stem, email_manipulator)


    print("Testing")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_header_stemmed_frequencies, ham_header_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator)


    print_results(spam_results, ham_results)



    # ### Both


    email_manipulator = lambda email : email

    print("Training: Email tokenized into words, full header and body of email.")
    ham_full_stemmed_frequencies = merge_frequencies(ham_body_stemmed_frequencies, ham_header_stemmed_frequencies)
    spam_full_stemmed_frequencies = merge_frequencies(spam_body_stemmed_frequencies, spam_header_stemmed_frequencies)


    print("Testing.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_full_stemmed_frequencies, ham_full_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator)



    print_results(spam_results, ham_results)



    # # Trigram

    # ### No header


    print("Training: Email tokenized into trigrams, ignore header.")
    email_manipulator = lambda email : remove_header(email)


    ham_body_trigram_frequencies = train(training_ham_filenames, tokenize_into_trigrams, email_manipulator)
    spam_body_trigram_frequencies = train(training_spam_filenames, tokenize_into_trigrams, email_manipulator)


    print("Testing")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_body_trigram_frequencies, ham_body_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator)



    print_results(spam_results, ham_results)


    # ### Header

    email_manipulator = lambda email : get_header(email)

    print("Training: Email tokenized into trigrams, only header.")
    ham_header_trigram_frequencies = train(training_ham_filenames, tokenize_into_trigrams, email_manipulator)
    spam_header_trigram_frequencies = train(training_spam_filenames, tokenize_into_trigrams, email_manipulator)


    print("Testing")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_header_trigram_frequencies, ham_header_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator)


    print_results(spam_results, ham_results)


    # ### Both


    email_manipulator = lambda email : email

    print("Training: Email tokenized into trigrams, both header and body of email.")
    ham_full_trigram_frequencies = merge_frequencies(ham_body_trigram_frequencies, ham_header_trigram_frequencies)
    spam_full_trigram_frequencies = merge_frequencies(spam_body_trigram_frequencies, spam_header_trigram_frequencies)


    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_full_trigram_frequencies, ham_full_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator)



    print_results(spam_results, ham_results)




    # # New Dataset Preliminaries
    """
    location = 'CSDMC2010_SPAM/extracted/'

    filenames = [name for name in os.listdir(location)]

    labels = open('CSDMC2010_SPAM/SPAMTrain.label').read().strip()

    is_spam = dict()

    for entry in labels.split('\n'):
        is_spam[entry.split()[1]] = entry.split()[0] == '0'


    os.mkdir('CSDMC2010_SPAM/extracted/spam')
    os.mkdir('CSDMC2010_SPAM/extracted/ham')

    for f in filenames:
        if f in is_spam:
            if is_spam[f]:
                os.rename(location+f, location+'spam/'+f)
            else:
                os.rename(location+f, location+'ham/'+f)

    location = 'CSDMC2010_SPAM/extracted/' + 'ham/'

    filenames = [name for name in os.listdir(location)]

    os.mkdir(location + 'training')
    os.mkdir(location + 'testing')

    i=0
    for f in filenames:
        if i % 6 == 0:
            os.rename(location+f, location+'testing/'+f)
        else:
            os.rename(location+f, location+'training/'+f)
        i+=1
    """





    # # Testing new dataset


    location = 'CSDMC2010_SPAM/extracted/'

    print("New Corpus: CSDMC2010_SPAM")
    training_spam_filenames = [location + "training/spam/" + filename for filename in os.listdir(location+"training/spam/")]
    training_ham_filenames = [location + "training/ham/" + filename for filename in os.listdir(location+"training/ham/")]

    testing_spam_filenames = [location + "testing/spam/" + filename for filename in os.listdir(location+"testing/spam/")]
    testing_ham_filenames = [location + "testing/ham/" + filename for filename in os.listdir(location+"testing/ham/")]


    # # Words


    email_manipulator = lambda email : email

    print("Training: Email tokenized into words, both header and email body.")
    ham_new_data_stemmed_frequencies = train(training_ham_filenames, tokenize_into_words_and_stem, email_manipulator)
    spam_new_data_stemmed_frequencies = train(training_spam_filenames, tokenize_into_words_and_stem, email_manipulator)


    print("Testing")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_new_data_stemmed_frequencies, ham_new_data_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator)


    print_results(spam_results, ham_results)

    # # Trigram

    print("Training: Email tokenized into trigrams, both header and email body.")
    email_manipulator = lambda email : email

    ham_new_data_trigram_frequencies = train(training_ham_filenames, tokenize_into_trigrams, email_manipulator)
    spam_new_data_trigram_frequencies = train(training_spam_filenames, tokenize_into_trigrams, email_manipulator)

    print("Testing")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_new_data_trigram_frequencies, ham_new_data_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator)


    print_results(spam_results, ham_results)


    # # Modification


    testing_spam_filenames = ["testing/spam/" + filename for filename in os.listdir("testing/spam/")]
    testing_ham_filenames = ["testing/ham/" + filename for filename in os.listdir("testing/ham/")]



    print("Applying modification to consider email size")

    # # Words

    # ### No header


    email_manipulator = lambda email : remove_header(email)

    print("Testing: Email tokenized into words, ignore header.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_body_stemmed_frequencies, ham_body_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator,
                                     False, True)

    print_results(spam_results, ham_results)


    # ### Header


    email_manipulator = lambda email : get_header(email)

    print("Testing: Email tokenized into words, only header.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_header_stemmed_frequencies, ham_header_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator,
                                     False, True)

    print_results(spam_results, ham_results)


    # ### Both



    email_manipulator = lambda email : email

    print("Testing: Email tokenized into words, both header and email body.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_full_stemmed_frequencies, ham_full_stemmed_frequencies,
                                     tokenize_into_words_and_stem, email_manipulator,
                                     False, True)

    print_results(spam_results, ham_results)



    # # Trigram

    # ### No header

    email_manipulator = lambda email : remove_header(email)

    print("Testing: Email tokenized into trigrams, ignore header.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_body_trigram_frequencies, ham_body_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator,
                                     False, True)

    print_results(spam_results, ham_results)



    # ### Header

    email_manipulator = lambda email : get_header(email)

    print("Testing: Email tokenized into trigrams, only header.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_header_trigram_frequencies, ham_header_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator,
                                     False, True)

    print_results(spam_results, ham_results)



    # ### Both

    email_manipulator = lambda email : email

    print("Testing: Email tokenized into trigrams, both header and email body.")
    spam_results, ham_results = test(testing_spam_filenames, testing_ham_filenames,
                                     spam_full_trigram_frequencies, ham_full_trigram_frequencies,
                                     tokenize_into_trigrams, email_manipulator,
                                     False, True)


    print_results(spam_results, ham_results)
    
    print("\n\nEnd of processing.")

    return # end main


main()

