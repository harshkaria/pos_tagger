In this work, we explore hidden markov models for tagging parts of speech. We begin by using the WSJ Penn Treebank corpus, which contains human annotated parts of speech for the sentences.

We begin by parsing the data and creating a vocabulary. For optimization, we select a threshold of 3. We output the word and its frequency within the corpus to vocab.txt and as aforementioned, our unknown word threshold as 3. The total size of the vocabulary ends up being 16,920 words. We have 32,537 as the frequency of our unknown word count

We then implement a HMM on our training data. We compute transition probabailities as a function of count(s -> s') / count(s) and emission probabilties as a function of count(s -> x) / count(s). We output these parameters to hmm.json . There are 1392 transition parameters and 50285 emission parameters in our HMM.
We then move on to decoding our HMM. Greedy decoding selects the most probable tag for a single word at a time, it is not the most optimal since it takes local decisions. The accuracy of the greedy decoder on the dev data is 92.13%.

Another way to decode the HMM is using the Viterbi algorithm. Viterbi decoding is an algorithm that computes the maximum probability of a state at a given time step given the maximum path at the previous time step with complexity O(mk^2). The accuracy of the Viterbi decoder on the dev data is 93.64%.
We write the output of the test data to greedy.out and viterbi.out , respectively.