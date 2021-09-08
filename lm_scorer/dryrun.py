import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer

# Load model to cpu or cuda
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 1
scorer = LMScorer.from_pretrained("gpt2multi", device=device, batch_size=batch_size)

# Return token probabilities (provide log=True to return log probabilities)


sentence1 = "¿Qué te gusta hacer?"
sentence2 = "¿Qué hacer te gusta?"
print(scorer.sentence_score([sentence1, sentence2],reduce="mean"))

'''

¿Qué te gusta hacer? – “What do you like to do?”
Mi pasatiempo favorito es… – “My favourite pastime is…”
¿Cuáles son tus pasatiempos? – “What are your hobbies?”
¿Qué haces en tu tiempo libre? – “What do you do in your free time?”
Me gusta / No me gusta… – “I like / I don’t like…”
Me encanta… – “I love…”
¿Qué te gusta leer? – “Do you like to read?”
¿Que música te gusta? – “What music do you like?”
Mi favorito es… – “My favourite is…”
Me gusta ir… – “I like going to…”
¿En qué trabajas? – “What’s your job?”
¿Te gusta tu trabajo? – “Do you like your job?”
Trabajo en… – “I work at…”


'''







# scorer.tokens_score("I like this package.")
# # => (scores, ids, tokens)
# # scores = [0.018321, 0.0066431, 0.080633, 0.00060745, 0.27772, 0.0036381]
# # ids    = [40,       588,       428,      5301,       13,      50256]
# # tokens = ["I",      "Ġlike",   "Ġthis",  "Ġpackage", ".",     "<|endoftext|>"]
#
# # Compute sentence score as the product of tokens' probabilities
# scorer.sentence_score("I like this package.", reduce="prod")
# # => 6.0231e-12
#
# # Compute sentence score as the mean of tokens' probabilities
# scorer.sentence_score("I like this package.", reduce="mean")
# # => 0.064593
#
# # Compute sentence score as the geometric mean of tokens' probabilities
# scorer.sentence_score("I like this package.", reduce="gmean")
# # => 0.013489
#
# # Compute sentence score as the harmonic mean of tokens' probabilities
# scorer.sentence_score("I like this package.", reduce="hmean")
# # => 0.0028008
#
# # Get the log of the sentence score.
# #print(scorer.sentence_score("I like this package.", log=True))
# # => -25.835
#
# # Score multiple sentences.
# scorer.sentence_score(["Sentence 1", "Sentence 2"])
# # => [1.1508e-11, 5.6645e-12]

# NB: Computations are done in log space so they should be numerically stable.
