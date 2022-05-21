import sys
from transformers import AutoModel, GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token
model = None

def setup_gpt2(checkpoint):
    global model
    model = GPT2LMHeadModel.from_pretrained(checkpoint, pad_token_id=tokenizer.eos_token_id)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    model.cuda()

def segment_fill(texts, checkpoint):
    assert isinstance(texts, list)
    if model is None:
        setup_gpt2(checkpoint)

    max_len = 0
    for text in texts:
        max_len = max(len(text), max_len)

    inputs = tokenizer([s + '\t' for s in texts], return_tensors="pt", padding=True)
    batch_output = model.generate(input_ids=inputs["input_ids"].cuda(), attention_mask=inputs["attention_mask"].cuda(), max_length=max_len * 2 + 7)
    batch_out_sentence = tokenizer.batch_decode(batch_output, skip_special_tokens=True)
    batch_out_sentence = [b.split('\t')[1] for b in batch_out_sentence]
    results = []
    for text, segmented_text in zip(texts, batch_out_sentence):
        chars_found = 0
        index = 0
        found = False
        for char in segmented_text:
            if index >= len(text):
                results.append(segmented_text[0:chars_found])
                found = True
                break
            if char.lower() == text[index].lower():
                index += 1
                chars_found += 1
            elif char in ['-', ' ', '\'', ';', ',', '.', '(', ')', '[', ']', '\"', '?', '/', '\\']:
                chars_found += 1
            else:
                results.append(None)
                found = True
                break
        if not found:
            results.append(None)

    # rerun all of the ones where a good candidate wasn't found
    for index, result in enumerate(results):
        #print("got from greedY: ", batch_out_sentence[index])
        if result is None:
            # print('going to beam search for', texts[index])
            input_ids = tokenizer.encode(texts[index] + '\t', return_tensors='pt')
            beam_output = model.generate(input_ids.cuda(), max_length=len(texts[index]) * 2 + 7, num_beams=5, early_stopping=True, num_return_sequences=5)
            candidates = [tokenizer.decode(beam_out, skip_special_tokens=True).split('\t')[1] for beam_out in beam_output]
            for candidate in candidates:
                chars_found = 0
                char_index = 0
                for char in candidate: 
                    if char_index >= len(texts[index]):
                        results[index] = candidate[0:chars_found]
                        break
                    if char.lower() == texts[index][char_index].lower():
                        char_index += 1
                        chars_found += 1
                    elif char in ['-', ' ', '\'', ';', ',', '.', '(', ')', '[', ']', '\"', '?', '/', '\\']:
                        chars_found += 1
                    else:
                        break
                if results[index] is not None:
                    break # break again
            if results[index] is None:
                # print('segemented messed up and failed to find!', texts[index], candidates)
                results[index] = texts[index].lower() # fallback to outputting original text

    assert len(results) == len(texts), f"Got {len(results)}, {len(texts)}"
    return results
