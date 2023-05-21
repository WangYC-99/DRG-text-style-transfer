from transformers import BertForMaskedLM, BertTokenizer

model = "/zhangpai25/wyc/drg/gpt-chinese" 
tokenizer = BertTokenizer.from_pretrained(model, use_fast=True)
# model = BertForMaskedLM.from_pretrained(model)

print(tokenizer.tokenize('<START>'))
print(tokenizer.convert_tokens_to_ids(['<start>', '你', '好', '呀']))
print(tokenizer.convert_tokens_to_ids('<start>'))

print(list('你好呀'))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize('<NEG> <CON_START> 神 天 菩 坑 死 我 ! <START> 神天菩萨,坑死我了! <END>')))


print(tokenizer.decode([749, 783]))
# print(tokenizer.tokenize('hospitalization'))

# new_tokens = ['<POS>', '<NEG>','<CON_START>','<START>','<END>']
# num_added_toks = tokenizer.add_tokens(new_tokens)

# model.resize_token_embeddings(len(tokenizer)) 

# print(tokenizer.tokenize('<START>'))
# # print(tokenizer.tokenize('hospitalization'))

# tokenizer.save_pretrained("/zhangpai25/wyc/drg/gpt-chinese") #还是保存到原来的bert文件夹下，这时候文件夹下多了三个文件