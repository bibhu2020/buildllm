https://www.youtube.com/watch?v=00GKzGyWFEs&list=PLo2EIpI_JMQvWfQndUesu0nPBAtZ9gP1o


# Encoder
Is used for extracting meaningful information (Natutal Language Understanding) like Q&A, Masked Input processing, classification, etc...

It is bidirectional in nature. It means, the context vector of a word in it built w.r.t to words in left and right both. It predicts the meaning based on the context vectors of entire sentence containing both words from left and right. 

E.g. BERT, RoBERTa

# Decoder
Is used for text generation tasks. (Natural Language Generation). It differs from Encoder in the way its self-attention mechanism works. The context vector of a word is either built w.r.t. to words in left or right, (not both). It predicts next word based on the context vectors of the previous words.

E.g. GPT-2

# Encoder-Decoder
Is used for sequence-to-sequence tasks like translation, summertization, etc..

E.g. T5, mT5, mBART, BART, MarianMT