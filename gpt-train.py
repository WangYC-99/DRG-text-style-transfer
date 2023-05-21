import torch, os
import torch.nn as nn
from pytorch_pretrained_bert import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from tqdm import tqdm
import numpy as np
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
# from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from bertviz.bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer
import torch.utils.data as Data
import datetime
import argparse
# from transformers import BertTokenizer

special_tokens = ['<POS>', '<NEG>','<CON_START>','<START>','<END>'] # Set the special tokens
# tokenizer = OpenAIGPTTokenizer.from_pretrained('/zhangpai25/wyc/drg/gpt-chinese', special_tokens=special_tokens)
from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('/zhangpai25/wyc/drg/gpt-chinese', special_tokens=special_tokens)
tokenizer = BertTokenizer.from_pretrained('/zhangpai25/wyc/drg/gpt-chinese', use_fast=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

model = OpenAIGPTLMHeadModel.from_pretrained('/zhangpai25/wyc/drg/gpt-chinese', num_special_tokens=len(special_tokens))

# path = os.path.join("/zhangpai25/wyc/drg/model_saved/dg_gptc_2hlm/pytorch_model_zero_grad_1.bin") ## Model Path
# model_state_dict = torch.load(path, map_location=device)
# model.load_state_dict(model_state_dict)
# model.to(device)
# model.eval()

print('gpt loaded')

def load_data(arg_mode, max_seq_lenth):
    # """用来生成训练、测试数据"""
    # train_df = pd.read_csv("bert_example.csv", header=None)
    # sentences = train_df[0].values
    # targets = train_df[1].values
    # train_inputs, test_inputs, train_targets, test_targets = train_test_split(sentences, targets)
    if arg_mode == 'train':
        # typed_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm/domain-1-typed/train.txt'
        # untyped_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm/domain-2-untyped/train.txt'
        file_path = '/zhangpai25/wyc/drg/drg_data/hlm/processed_files_with_bert_with_best_head/sentiment_train_0.txt'
    elif arg_mode == 'val':
        # typed_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm/domain-1-typed/valid.txt'
        # untyped_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm/domain-2-untyped/valid.txt'
        file_path = '/zhangpai25/wyc/drg/drg_data/hlm/processed_files_with_bert_with_best_head/sentiment_dev_0.txt'
    elif arg_mode == 'test':
        # typed_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm/domain-1-typed/test.txt'
        # untyped_data_dir = '/zhangpai25/wyc/lewis/lewis_wyc/datasets_used/chinese/processed/hlm/domain-2-untyped/test.txt'
        file_path = '/zhangpai25/wyc/drg/drg_data/hlm/processed_files_with_bert_with_best_head/reference_0.txt'

    train_inputs = []
    train_targets = []

    file = open(file_path)
    content = file.read()
    sentences = content.split('\n')
    print('sentence shape: {}'.format(len(sentences)))

    for each in sentences:
        input_ids = []
        output_ids = []
        start_position = 0
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(each))
        if tokenizer.convert_tokens_to_ids('<start>') not in token_ids:
            continue
        tokens_counter = 0
        for each_id in token_ids:
            start_position += 1
            if each_id != tokenizer.convert_tokens_to_ids('<start>'):
                input_ids.append(each_id)
            else:
                break
            tokens_counter += 1
            if tokens_counter == max_seq_lenth:
                break
        tokens_counter = 0
        for iteration in range(len(token_ids)):
            if iteration >= start_position:
                output_ids.append(token_ids[iteration])
            tokens_counter += 1
            if tokens_counter == max_seq_lenth:
                break
        input_ids.append(tokenizer.convert_tokens_to_ids('<start>'))

        for iteration in range(max_seq_lenth + 1):
            # print('iteration: {}, input_len:{}, output_len:{}'.format(iteration, len(input_ids), len(output_ids)))
            if iteration > len(input_ids):
                input_ids.append(tokenizer.convert_tokens_to_ids('[PAD]'))
            if iteration > len(output_ids):
                output_ids.append(tokenizer.convert_tokens_to_ids('[PAD]'))
        
        train_inputs.append(torch.tensor(input_ids))
        train_targets.append(torch.tensor(output_ids))
                
    return train_inputs, train_targets

# # data loader test
# input, target = load_data('train', 128)
# print(len(input))
# print(len(input[1]))
# print(tokenizer.decode(target[0]))

sm = torch.nn.Softmax(dim=-1)

def train(args):
    print('training start! ')
    train_inputs, train_targets = load_data('train', args.max_seq_lenth)
    print('data loaded! ')

    # epochs = args.epoch
    # batch_size = args.batch_size
    # data_dir = args.data_dir
    epochs = args.epoch
    batch_size = args.batch_size

    # print('train_inputs lenth: {}'.format(len(train_inputs[0])))

    train_sentence_loader = Data.DataLoader(
        dataset=train_inputs,
        batch_size=batch_size,  # 每块的大小
    )
    train_label_loader = Data.DataLoader(
        dataset=train_targets,
        batch_size=batch_size,
    )

    print("train_sentence_loader: {}".format(len(train_sentence_loader)))
    for each in train_sentence_loader:
        print(each)
        break

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    iteration = 0
    best_eval = 0
    es_count = 0

    model.train()

    for epoch in range(epochs):
        print('...this is epoch : {}...'.format(epoch))

        for input_ids, target_ids in zip(train_sentence_loader, train_label_loader):

            # input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(inputs))
            # target_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(targets))
            print('input_ids: {}, target_ids: {}'.format(len(input_ids), len(target_ids)))
            current_state = input_ids
            loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

            token_count = 0
            sentence_loss = []
            while token_count < args.max_seq_lenth:
                
                preds = sm(model(current_state))
                # print('preds shape:{} '.format(preds[:, -1, :].size()))
                # print('target shape: {}'.format(target_ids[:, 0]))

                loss = loss_fn(preds[:, -1, :], target_ids[:, token_count])

                optimizer.zero_grad()
                loss.backward()

                print('\r token count : {}, loss : {}'.format(token_count, loss), end = '')
                # writer.add_scalar('token_leval_loss', )
                sentence_loss.append(loss)


                sentence_iterator = 0
                for sentence_iterator in range(args.batch_size):
                    pad_iterator = 0
                    for pad_iterator in range(args.max_seq_lenth):
                        if current_state[sentence_iterator][pad_iterator] == torch.tensor(0):
                            current_state[sentence_iterator][pad_iterator] = target_ids[sentence_iterator, token_count]
                            break
                
                # print(current_state)

                token_count += 1
            print('this is iteration: {} and train loss: {}'.format(iteration, sum(sentence_loss) / len(sentence_loss)))
            writer.add_scalar('sentence_leval_loss', sum(sentence_loss) / len(sentence_loss), iteration)
    torch.save(model.state_dict(), args.model_save)
     
            
        

def inference(ref_text):
    print('inferring start! ')

    model.eval()

    input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ref_text))])

    current_state = input_ids

    token_count = 0

    while token_count < args.max_seq_lenth:
        print('\r token counter: {}'.format(token_count), end = '')
        
        preds = sm(model(current_state))
        # print('preds shape:{} '.format(preds[:, -1, :].size()))
        # print('target shape: {}'.format(target_ids[:, 0]))

        result = torch.argmax(preds[-1, -1, :], dim = -1)
        # print('result_type : {}'.format(type(result)))

        # print('token count : {}, result: {}'.format(token_count, result.unsqueeze(dim = -1).unsqueeze(dim = -1).shape))


        # for pad_iterator in range(args.max_seq_lenth):
        #     if current_state[0][pad_iterator] == torch.tensor(0):
        #         current_state[0][pad_iterator] = result
        #         break
        current_state = torch.cat((current_state, result.unsqueeze(dim = -1).unsqueeze(dim = -1)), dim = 1)
        
        if result.item() == tokenizer.convert_tokens_to_ids('<end>'):
            break
        # print(current_state)

        token_count += 1
    
    final = tokenizer.decode(current_state[-1].tolist())
    print("final: {}" .format(final))
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser() # 初始化

    parser.add_argument('--batch_size', type=int, default=4,
                        help='size of one batch') 
    parser.add_argument('--epoch', type=int, default=100,
                        help='epoch')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='learning_rate')
    parser.add_argument('--val_per_ite', type=int, default=20,
                        help='validation per how many iterations')
    parser.add_argument('--model_save', default='/zhangpai25/wyc/drg/model_saved/gpt_wyc/model_saved/',
                        help='path to ckpt save dir')
    parser.add_argument('--device', default='cpu',
                        help='the device you want to use to train')
    parser.add_argument('--instruction', default='train',
                            help='choose from train / test/ infer')
    parser.add_argument('--ckpt_path', default='', )
    parser.add_argument('--max_seq_lenth', type=int, default='128')

    args, others_list = parser.parse_known_args() # 解析已知参数


    time = datetime.datetime.now()
    time_str = str(time.month) + '-' + str(time.day) + '-' + str(time.hour) + '-' + str(time.minute)

    args.model_save = args.model_save + args.instruction + time_str

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('/zhangpai25/wyc/drg/model_saved/gpt_wyc/logs/' + time_str)

    if args.instruction == 'train':
        train(args)

    elif args.instruction == 'infer':
        inference("你好呀，我")
    # eval(args)
    # test()
    print('done :-) thx4using')