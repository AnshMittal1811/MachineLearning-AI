import somata
from evaluate import *

print("Loading encoder...")
encoder = torch.load('seq2seq-encoder.pt')
print("Loading decoder...")
decoder = torch.load('seq2seq-decoder.pt')
print("Loaded models.")

def parse(body, cb):
    print('[parse]', body)
    parsed, prob, attn = evaluate(encoder, decoder, body)
    print(parsed, prob)
    cb({'parsed': parsed, 'prob': prob})

service = somata.Service('maia:parser', {'parse': parse}, {'bind_port': 7181})
