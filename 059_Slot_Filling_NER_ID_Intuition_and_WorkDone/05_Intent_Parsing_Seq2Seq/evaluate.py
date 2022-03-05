from data import *
from model import *

MIN_PROB = -0.1

# # Evaluating the trained model

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    encoder.train(False)
    decoder.train(False)

    input_variable = input_lang.variable_from_sentence(sentence)
    input_length = input_variable.size()[0]

    encoder_outputs, encoder_hidden = encoder(input_variable)

    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_hidden = encoder_hidden

    decoded_words = []
    seq_length = input_variable.size(0)
    decoder_attentions = torch.zeros(max_length, seq_length)

    total_prob = 0

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data[-1]
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            break
        else:
            total_prob += topv[0][0]
            decoded_words.append(output_lang.index2word[ni])
        decoder_input = Variable(torch.LongTensor([[ni]]))

    encoder.train(True)
    decoder.train(True)

    return decoded_words, total_prob, decoder_attentions[:di+1]

test_sentences = [
    'um can you turn on the office light',
    'hey maia please turn off all the lights thanks',
    'how are you today',
    'thank you',
    'please make the music loud',
    'whats the weather in minnesota',
    'whats the weather in sf',
    'are you on',
    'is my light on'
]

def evaluate_tests(encoder, decoder, ):
    for test_sentence in test_sentences:
        command, prob, attn = evaluate(encoder, decoder, test_sentence)
        command_str = ' '.join(command)
        if prob < MIN_PROB:
            command_str += ' (???)'
        print(test_sentence, '\n    %.4f : %s' % (prob, command_str))

if __name__ == '__main__':
    import sys
    input = sys.argv[1]
    print('input', input)

    encoder = torch.load('seq2seq-encoder.pt')
    decoder = torch.load('seq2seq-decoder.pt')

    command, prob, attn = evaluate(encoder, decoder, input)
    if prob > -0.05:
        print(prob, command)
    else:
        print(prob, "UNKNOWN")

