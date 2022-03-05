import torch
from torch.autograd import Variable
import random
import re

# ## Generating training data

templates = [
    ("lights setState $light_name $light_state", [
        "~turn $light_state $light_name",
        "~turn $light_name $light_state",
    ]),
    ("lights setState $light_name $light_amount", [
        "~turn $light_name $light_amount",
    ]),
    ("lights setStates $group_name $light_state", [
        "~turn $light_state $group_name",
        "~turn $group_name $light_state",
    ]),
    ("lights setStates $group_name $light_amount", [
        "~turn $group_name $light_amount",
    ]),
    ("lights getState $light_name", [
        "is $light_name on",
    ]),
    ("lights getStates $group_name", [
        "are $group_name on",
    ]),
    ("music setVolume $volume", [
         "~turn the music $volume",
         "~turn it $volume",
    ]),
    ("time getTime", [
         "time",
         "what time is it",
         "~whatis the time",
    ]),
    ("price getPrice $asset", [
         "$asset",
         "how much is $asset",
         "~price of $asset",
         "~whatis the ~price of $asset",
    ]),
    ("weather getWeather $location", [
         "tell me the weather in $location",
         "~whatis it like in $location",
    ]),
    ("greeting", [
        "hi", "hello", "how are you", "what's up", "hey maia",
    ]),
    ("thanks", [
        "thanks", "thank you", "thank you so much", "thx", "you're great",
    ]),
]

variables = {
    "$light_name": [
        ("office_light", ["the office light", "the light in the office"]),
        ("kitchen_light", ["the kitchen light", "the light in the kitchen"]),
        ("living_room_light", ["the living room light", "the light in the living room", "the light in the den"]),
        ("outside_light", ["the outside light", "the outdoor light", "the light outside", "the porch light"]),
    ],
    "$group_name": [
        ("all_lights", ["all the lights", "everything"]),
        ("bedroom_lights", ["my lights", "the bedroom lights", "the lights", "the lights in my room"]),
    ],
    "$light_state": [
        ("on", ["on"]),
        ("off", ["off", "out"]),
        ("blue", ["blue"]),
        ("green", ["green"]),
        ("red", ["red"]),
        ("purple", ["purple"]),
        ("orange", ["orange"]),
        ("white", ["white", "normal"]),
    ],
    "$light_amount": [
        ("low", ["low", "dim", "dark"]),
        ("high", ["high", "bright"]),
        ("down", ["down", "lower", "darker"]),
        ("up", ["up", "brighter"]),
    ],
    "$volume": [
        ("high", ["high", "loud"]),
        ("low", ["low", "quiet", "down"]),
        ("up", ["up", "louder"]),
        ("down", ["down", "quieter"]),
    ],
    "$location": [
        ("san_francisco", ["sf", "san francisco", "the city"]),
        ("new_hampshire", ["nh", "new hampshire", "the northeast"]),
    ],
    "$asset": [
        ("btc", ["btc", "bitcoin", "bitcoins"]),
        ("eth", ["eth", "ethereum", "etherium", "ether"]),
        ("usd", ["usd", "dollars", "us dollars", "the fed"]),
        ("pesos", ["pesos", "mexican dollars"]),
    ],
}

synonyms = {
    "~turn": ["turn", "set", "make", "put", "change"],
    "~whatis": ["what is", "what's", "whats", "tell me", "tell us", "tell me about", "what about", "how about", "show me"],
    "~price": ["price", "value", "exchange rate", "dollar amount"],
}

prefixes = ["please", "pls", "hey maia", "hi", "could you", "would you", "hey", "yo", "ey", "oy", "excuse me please"]
suffixes = ["thanks", "thank you", "please", "plz", "pls", "plox", "ok", "thank you so much"]

# Choose a random pair of templates (output, input)
def choose_templates():
    output_template, input_templates = random.choice(templates)
    input_template = random.choice(input_templates)
    return output_template, input_template

input_template, output_template = choose_templates()
print('input template =', input_template)
print('output template =', output_template)

# We'll assume that all the variables in the input template are used in the output template.

# Choose variable values to fill a template with (output, input)
def choose_variables(template):
    variable_names = [word for word in template.split(' ') if word[0] == '$']
    input_variables = {}
    output_variables = {}
    for variable_name in variable_names:
        variable = random.choice(variables[variable_name])
        output_variables[variable_name] = variable[0]
        input_variables[variable_name] = random.choice(variable[1])
    return output_variables, input_variables

input_variables, output_variables = choose_variables(input_template)
print('input variables =', input_variables)
print('output variables =', output_variables)

def fill_template(template, template_variables):
    filled = []
    for word in template.split(' '):
        # Choose variable
        if word[0] == '$':
            filled.append(template_variables[word])
        # Choose synonym
        elif word[0] == '~':
            filled.append(random.choice(synonyms[word]))
        # Regular word
        else:
            filled.append(word)
    return ' '.join(filled)

PREFIX_PROB = 0.3
SUFFIX_PROB = 0.3

def add_fixes(sentence):
    if random.random() < PREFIX_PROB:
        sentence = random.choice(prefixes) + ' ' + sentence
    if random.random() < SUFFIX_PROB:
        sentence += ' ' + random.choice(suffixes)
    return sentence

def random_training_pair():
    output_template, input_template = choose_templates()
    output_variables, input_variables = choose_variables(input_template)

    output_string = fill_template(output_template, output_variables)
    input_string = fill_template(input_template, input_variables)
    input_string = add_fixes(input_string)

    return input_string, output_string

for i in range(10):
    print('\n', random_training_pair())

# ## Keeping track of the output vocabulary

SOS_token = 0
EOS_token = 1

def tokenize_sentence(s):
    s = re.sub('[^\w\s]', '', s)
    s = re.sub('\s+', ' ', s)
    return s.strip().split(' ')

class DictionaryLang():
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.size = 2 # Count SOS and EOS

    def __str__(self):
        return "%s(size = %d)" % (self.__class__.__name__, self.size)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.size
            self.word2count[word] = 1
            self.index2word[self.size] = word
            self.size += 1
        else:
            self.word2count[word] += 1

    def get_word(self, word):
        return self.word2index[word]

    def indexes_from_sentence(self, sentence):
        return [self.get_word(word) for word in tokenize_sentence(sentence)]

    def variable_from_sentence(self, sentence):
        indexes = self.indexes_from_sentence(sentence)
        indexes.append(EOS_token)
        return Variable(torch.LongTensor(indexes).view(-1, 1))

# First turn the generated data into Lang for input (english) and output (command) languages

output_lang = DictionaryLang()

def add_words(lang, template):
    for word in template.split(' '):
        if word[0] != '$':
            lang.add_word(word)

# Add words from templates

for output_template, input_templates in templates:
    add_words(output_lang, output_template)

# Add values of variables

for variable_name in variables:
    for output_variable, input_variables in variables[variable_name]:
        add_words(output_lang, output_variable)

print("output lang = %s" % output_lang)

# ## Using word vectors for the input vocabulary

from torchtext.vocab import load_word_vectors

class GloVeLang:
    def __init__(self, size):
        self.size = size
        glove_dict, glove_arr, glove_size = load_word_vectors('data/', 'glove.twitter.27B', size)
        self.glove_dict = glove_dict
        self.glove_arr = glove_arr

    def __str__(self):
        return "%s(size = %d)" % (self.__class__.__name__, self.size)

    def vector_from_word(self, word):
        if word in self.glove_dict:
            return self.glove_arr[self.glove_dict[word]]
        else:
            return torch.zeros(self.size)

    def variable_from_sentence(self, sentence):
        words = tokenize_sentence(sentence.lower())
        tensor = torch.zeros(len(words), 1, self.size)
        for wi in range(len(words)):
            word = words[wi]
            tensor[wi][0] = self.vector_from_word(word)
        return Variable(tensor)

input_lang = GloVeLang(100)
print("input lang = %s" % input_lang)
input_lang.variable_from_sentence('turn on the light').size()

# Now we can use these Lang objects to create tensors from sentences:

def variables_from_pair(pair):
    input_variable = input_lang.variable_from_sentence(pair[0])
    target_variable = output_lang.variable_from_sentence(pair[1])
    return (input_variable, target_variable)

def generate_training_pairs(n_iters):
    pairs = []
    for i in range(n_iters):
        pairs.append(variables_from_pair(random_training_pair()))
    return pairs

