# -*- coding: utf-8 -*-
"""

USC ID: 5860082592
## Hidden Markov Models for Part of Speech Tagging
In this project, we explore hidden markov models for tagging parts of speech. We begin by using the WSJ Penn Treebank corpus, which contains human annotated parts of speech for the sentences.

We begin by parsing the data and creating a vocabulary. For optimization, we select a threshold of 3. We output the word and its frequency within the corpus to `vocab.txt` and as aforementioned, our unknown word **threshold as** **3**. The total size of the vocabulary ends up being **16,920** words. We have **32,537** as the frequency of our unknown word count

We then implement a HMM on our training data. We compute transition probabailities as a function of count(s -> s') / count(s) and emission probabilties as a function of count(s -> x) / count(s). We output these parameters to `hmm.json`. There are **1392 transition parameters** and **50286 emission parameters** in our HMM.

We then move on to decoding our HMM. Greedy decoding selects the most probable tag for a single word at a time, it is not the most optimal since it takes local decisions. The accuracy of the greedy decoder on the dev data is **92.13%.**

Another way to decode the HMM is using the Viterbi algorithm. Viterbi decoding is an algorithm that computes the maximum probability of a state at a given time step given the maximum path at the previous time step with complexity O(mk^2). The accuracy of the Viterbi decoder on the dev data is **93.64%**. 

We write the output of the test data to `greedy.out` and `viterbi.out`, respectively.
"""


import pandas as pd
import json

"""## Vocabulary Creation

Task: Create a vocabulary using the training data in the file train and output the vocabulary into a txt file named vocab.txt. The format of the ocabulary file is that each line contains a 
word type, its index in the vocabulary and its occurrences, separated by the tab symbol ‘\t’. 
The first line should be the special token ‘< unk >’ and the following lines should be sorted by its occurrences in descending

"""

THRESHOLD = 3

from collections import defaultdict

vocab = defaultdict(lambda: 0)

with open("data/train", "r") as file:
  lines = file.readlines()
  for line in lines:
    word_split = line.strip().split("\t")
    word = word_split[1] if len(word_split) >= 2 else None
    if word:
      vocab[word] += 1

new_vocab = defaultdict(lambda: 0)

for word, count in vocab.items():
  if count >= THRESHOLD:
    new_vocab[word] = count
  else:
    new_vocab["<UNK>"] += count

new_vocab = pd.DataFrame.from_dict(new_vocab, orient='index', columns=['count'])
new_vocab = new_vocab.sort_values(by='count', ascending=False)
new_vocab = new_vocab.reset_index(drop=False)
new_vocab = new_vocab.rename(columns={"index": "word"})

# Moving unknown to the top of the data frame:
unk_index = new_vocab[new_vocab['word'] == "<UNK>"].index[0]
unk_count = new_vocab[new_vocab['word'] == "<UNK>"]["count"].values[0]
new_vocab = new_vocab.drop([unk_index], axis=0)
new_vocab.index = range(len(new_vocab))
new_vocab = pd.concat([pd.DataFrame([["<UNK>", unk_count]], columns=["word", "count"]), new_vocab])
new_vocab = new_vocab.reset_index(drop=True)

with open("vocab.txt", "w") as file:
    for i, row in new_vocab.iterrows():
        file.write("{}\t{}\t{}\n".format(row['word'], i, row['count']))

"""We select our unknown word threshold as **3**. 

The total size of the vocabulary ends up being **16,920** words.

We have **32,537** as the frequency of our unknown word count

## Model Learning
"""

# Find all sentences from the training data

from collections import defaultdict

def get_sentences_and_states(file_path):
  sentences = []
  states = []
  with open(file_path, "r") as file:
    lines = file.readlines()
    # singular sentence
    sentence = []
    # singular tag of format (prev tag, curr tag)
    state_array = []
    prev_state = "<START>"
    for line in lines:
      # New sentence
      if len(line.strip()) == 0 and len(sentence) > 0:
        sentences.append([_ for _ in sentence])
        states.append([_ for _ in state_array])
        sentence = []
        state_array = []
        prev_state = "<START>"
        continue
      word_split = line.strip().split("\t")
      word = word_split[1] if len(word_split) >= 2 else None
      curr_state = word_split[2] if len(word_split) >= 3 else None
      sentence.append(word)
      state_array.append((prev_state, curr_state))
      prev_state = curr_state
    sentences.append([_ for _ in sentence])
    states.append([_ for _ in state_array])
    return sentences, states

def hmm(sentences, states):
  transitions = defaultdict(int)
  emissions = defaultdict(int)
  state_count = defaultdict(int)

  sentences_states = zip(sentences, states)


  for sentence, tag_tuples in sentences_states:
    for word, tag_tuple in zip(sentence, tag_tuples):
      state_count[tag_tuple[1]] += 1
      transitions[tag_tuple] += 1
      emissions[(tag_tuple[1], word)] += 1
  
  # START TOKEN
  state_count["<START>"] = len(sentences)
  
  serialized_transitions = {}
  for (prev_state, state), count in transitions.items():
    serialized_transitions[json.dumps((prev_state, state))] = count / state_count[prev_state]
  serialized_emissions = {}
  for (state, word), count in emissions.items():
    serialized_emissions[json.dumps((state, word))] = count / state_count[state] 
  
  return {
      "transitions": dict(serialized_transitions), 
      "emissions": dict(serialized_emissions)
      }

sentences_train, states_train = get_sentences_and_states("data/train")

hmm_model = hmm(sentences_train, states_train)
with open("hmm.json", "w") as file:
    json.dump(hmm_model, file)

transition_params = len(hmm_model["transitions"].keys())
emissions_params = len(hmm_model["emissions"].keys())

print(f'There are {transition_params} transition parameters and {emissions_params} emission parameters in our HMM')

"""## Greedy Decoding with HMM

In greedy decoding, we start from one word and decode a tag one word at a time, making local decisions. For every word, we can compute:

s(j) = argmax t(s[j] | s[j-1])e(x[j]|s[j]) for all js
"""

def get_all_states(states):
    states_set = set()
    states_set.add("<START>")
    for state_sentence in states:
      for state in state_sentence:
        states_set.add(state[1])
    return states_set

unique_states = get_all_states(states_train)

def greedy_decoder(sentences, states, transitions, emissions):
  decoded_tags = []
  for sentence in sentences:
    prev_tag = "<START>"
    tags = [] 
    for word in sentence:
      prob = float("-inf")
      curr_tag = None
      for tag in states:
        t_pob = transitions.get(json.dumps((prev_tag, tag)), 0)
        e_pob = emissions.get(json.dumps((tag, word)), 0)
        curr_prob = t_pob * e_pob
        if curr_prob > prob:
          curr_tag = tag
          prob = curr_prob
      prev_tag = curr_tag
      tags.append(curr_tag)
    decoded_tags.append([_ for _ in tags])
  return decoded_tags

dev_sentences, dev_states = get_sentences_and_states("data/dev")
unique_states_dev = get_all_states(dev_states)

decoded_tags_dev = greedy_decoder(dev_sentences, unique_states_dev, hmm_model["transitions"], hmm_model["emissions"])

def calc_accuracy(predictions, ground_truth):
    total = 0
    correct = 0

    for sentences_state, prediction_sentence in zip(ground_truth, predictions):
      for tag_tuple, prediction in zip(sentences_state[1], prediction_sentence):
          total += 1 
          if tag_tuple[1] == prediction:
            correct += 1

    acc = correct / total
    return acc

greedy_decoder_dev_acc = calc_accuracy(decoded_tags_dev, zip(dev_sentences, dev_states))

print(f'The accuracy of the greedy decoder on the dev data is {greedy_decoder_dev_acc}')

test_sentences, test_states = get_sentences_and_states("data/test")

decoded_tags_test = greedy_decoder(test_sentences, unique_states_dev, hmm_model["transitions"], hmm_model["emissions"])

def write_inference_to_file(decoded_tags, test_sentences, filename):
    with open(filename, "w") as file:
        for sentence, decoded_sentence_tags in zip(test_sentences, decoded_tags):
            for i, (word, tag) in enumerate(zip(sentence, decoded_sentence_tags), 1):
                line = f'{i}\t{word}\t{tag}'
                file.write(line + '\n')
            file.write('\n')

write_inference_to_file(decoded_tags_test, test_sentences, "greedy.out")

"""## Viterbi Decoding

Viterbi decoding is an algorithm that computes the maximum probability of a state at a given time step given the maximum path at the previous time step
"""

def viterbi_decoding(sentences, states, transitions, emissions, epsilon = 0.0000000000001):
    states = list(states)
    decoded_tags = []
    for sentence in sentences:
        prev_tag = "<START>"
        trails = {}
        for idx, word in enumerate(sentence):
            for tag in states:
                # Initial probabilities
                prev_tag_idx = states.index(prev_tag)
                curr_tag_idx = states.index(tag)
                if idx == 0:
                    t_prob = transitions.get(json.dumps((prev_tag, tag)), epsilon)
                    e_prob = emissions.get(json.dumps((tag, word)), epsilon)
                    curr_prob = t_prob * e_prob
                    trails[tag, idx] = curr_prob
                    continue
                else:
                  memo_probs = [(trails[k, idx - 1] * transitions.get(json.dumps((k, tag)), epsilon) * emissions.get(json.dumps((tag, word)), epsilon), k) for k in states]
                  k = sorted(memo_probs)[-1][1]
                  trails[tag, idx] = trails[k, idx - 1] * transitions.get(json.dumps((k, tag)), epsilon) * emissions.get(json.dumps((tag, word)), epsilon)
        
        path = []
        for word in range(len(sentence) - 1, -1, -1):
          k = sorted([(trails[k, word], k) for k in states])[-1][1]
          path.append((sentence[word], k))
        
        path.reverse()
        decoded_tags.append([x[1] for x in path])
    return decoded_tags

decoded_tags_dev_viterbi = viterbi_decoding(dev_sentences, unique_states_dev, hmm_model["transitions"], hmm_model["emissions"])

decoded_tags_test_viterbi = viterbi_decoding(test_sentences, unique_states_dev, hmm_model["transitions"], hmm_model["emissions"])

viterbi_decoder_dev_acc = calc_accuracy(decoded_tags_dev_viterbi, zip(dev_sentences, dev_states))

print(f'The accuracy of the Viterbi decoder on the dev data is {viterbi_decoder_dev_acc}')

write_inference_to_file(decoded_tags_test_viterbi, test_sentences, "viterbi.out")