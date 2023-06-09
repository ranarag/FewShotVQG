import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import os
from utils import AverageMeter, accuracy, calculate_caption_lengths, count_parameters

from utils import Vocabulary
from utils import get_glove_embedding, get_bert_embedding
from utils import datasets, collate_fn, newcollate_fn, samplers, NewCategoriesSampler7w
from utils import load_vocab
from utils import process_lengths
from utils import NLGEval
from torch.utils.data import DataLoader
import learn2learn as l2l
import pickle as pkl
import copy
from collections import defaultdict
from models import VQGNetANS, recast_answer_embeds
torch.cuda.set_device(0)

from termcolor import cprint, colored
# from models import Autoencoder, CategoryEncoder, Encoder, Decoder
from tqdm import tqdm
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def calc_loss_and_acc(loss_fn, alpha_c, preds, args, alphas, targets):
    
    decode_lengths = process_lengths(targets)

    preds = pack_padded_sequence(preds, decode_lengths, batch_first=True, enforce_sorted=False)
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # some kind of regularization used by Show Attend and tell repo
    loss = loss_fn(preds.data, targets.data)
    if args.decoder_type == 'lstm':
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
    elif args.decoder_type== "transformer":
        dec_alphas = alphas["dec_enc_attns"]
        decoder_layers = len(dec_alphas)
        n_heads = dec_alphas[0].size(1)
        alpha_trans_c = alpha_c / (n_heads * decoder_layers)
        for layer in range(decoder_layers):  # args.decoder_layers = len(dec_alphas)
            cur_layer_alphas = dec_alphas[layer]  # [batch_size, n_heads, 52, 196]
            for h in range(n_heads):
                cur_head_alpha = cur_layer_alphas[:, h, :, :]
                loss += alpha_trans_c * ((1. - cur_head_alpha.sum(dim=1)) ** 2).mean()


    return loss


def fast_adapt(train_batch, test_batch, learner, alpha_c, loss_fn, adaptation_steps, word_dict, args, nlge):

    img_train, cap_train, ans_train, alen_train = train_batch
    img_test, cap_test, ans_test, alen_test = test_batch
    # Adapt the model>
    train_targets = cap_train[:, 1:]
    eval_targets = cap_test[:, 1:]

    for step in range(adaptation_steps):
        preds, alphas = learner(img_train, ans_train, alen_train, cap_train)
        train_error = calc_loss_and_acc(loss_fn, alpha_c, preds, args, alphas, train_targets)
        train_error /= preds.size(0)
        learner.adapt(train_error, allow_nograd=True, allow_unused=True)

    # Evaluate the adapted model
    preds, alphas = learner(img_test, ans_test, alen_test, cap_test)
    # print(preds)
    valid_error= calc_loss_and_acc(loss_fn, alpha_c, preds, args, alphas, train_targets)
    valid_error /= preds.size(0)


    scores = 0.
    return valid_error, scores

def inference(test_batch, word_dict, learner, args):
    img_test, cap_test, ans_test, alen_test = test_batch
    beam_size = 5
            
    predictions = []
    gts = []
    with torch.no_grad():
        

        if args.decoder_type == 'lstm':
            if args.scaling_shifting:
                n_img_features = learner.get_img_embeds(img_test, ans_test, alen_test)
            else:
                n_img_features, ans_embeds = learner.get_img_embeds_no_SS(img_test, ans_test, alen_test)



            for i in range(img_test.size(0)):
                new_img_features = n_img_features[i].unsqueeze(0)

                if args.scaling_shifting:
                    new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2))
                    outputs, alphas = learner.decoder.caption(new_img_features, word_dict, beam_size)
                else:
                    new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2))
    #                 print(ans_embeds.size())
                    new_ans_embeds = ans_embeds[i].unsqueeze(0)
                    new_ans_embeds = new_ans_embeds.expand(beam_size, new_ans_embeds.size(1))
                    outputs, alphas = learner.decoder.caption(new_img_features, word_dict, beam_size, new_ans_embeds)
                output = word_dict.tokens_to_words(outputs)
                predictions.append(output)


                question = word_dict.tokens_to_words(cap_test[i])
                gts.append(question)

        else:
            ans_embeds = learner.answer_encoder(ans_test, alen_test)
            for i in range(img_test.size(0)):
                new_img_features = img_test[i].unsqueeze(0).repeat(beam_size, 1, 1, 1)
                new_ans_embeds = ans_embeds[i].unsqueeze(0).repeat(beam_size, 1)
    #             new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2), new_img_features.size(3))
                outputs, alphas = learner.decoder.caption(new_img_features, new_ans_embeds, word_dict, beam_size)
                output = word_dict.tokens_to_words(outputs)
                predictions.append(output)


                question = word_dict.tokens_to_words(cap_test[i])
                gts.append(question)


        return predictions, gts

def adapt_and_eval(train_batch, test_batch, word_dict, nlge, learner, alpha_c, loss_fn, args, adaptation_steps, optimizer, last):
    learner.train()
    img_train, cap_train, ans_train, alen_train = train_batch
    img_test, cap_test, ans_test, alen_test = test_batch
    # Adapt the model
    train_targets = cap_train[:, 1:]
    eval_targets = cap_test[:, 1:]

    for step in range(adaptation_steps):
        preds, alphas = learner(img_train, ans_train, alen_train, cap_train)
        train_error = calc_loss_and_acc(loss_fn, alpha_c, preds, args, alphas, train_targets)
#         train_error /= preds.size(0)

        optimizer.zero_grad()
        train_error.backward()
        optimizer.step()
    if last:
        learner.eval()
        predictions, gts = inference(test_batch, word_dict, learner, args)



        eg = "Pred = {pred}; GT = {gt}".format(pred=predictions[0], gt=gts[0])
        scores = nlge.compute_metrics(ref_list=[gts], hyp_list=predictions)
        return scores, predictions, gts, eg



def main(args):


    word_dict = load_vocab(args.vocab_path)
    vocabulary_size = len(word_dict)
    nlge = NLGEval(no_glove=True, no_skipthoughts=True)
    ans_dict = load_vocab(args.vocab_path_ans)
    ans_size = len(ans_dict)
    decoder_model_path = os.path.join('model/meta', 'nbn_decoder', args.decoder_model)
    answer_enc_model_path = os.path.join('model/meta', 'nbn_answer_encoder', args.cat_model)
    lm_encoder = 'nobert'
    embedding=None
    if len(args.bert_embed):
        with open(args.bert_embed, 'rb') as fid:
            embedding = pkl.load(fid)
    ans_embedding=None
    if len(args.bert_ans_embed):
        with open(args.bert_ans_embed, 'rb') as fid:
            ans_embedding = pkl.load(fid)
        lm_encoder = 'bert'
    if not args.scaling_shifting:
        lm_encoder += "_no_SS"
    if args.network == 'resnet':
        encoder_dim = 2048
    else:
        encoder_dim = 1536
    new_transform =  transforms.Compose([
        transforms.ToTensor()])
    if args.dataset_type == '7w':
        trainset = datasets.get_class_obj("Val",args.dataset,
                                          #'data/processed/val_img_id_resnet.hdf5', 
                                          args.qid2data,
                                          dataset_type=args.dataset_type,
                                          max_examples=None)

        train_sampler = NewCategoriesSampler7w('train_cat2qid7w.json',
                                          1000,
                                          args.way,
                                          args.train_query,
                                          args.test_query)

        train_loader = DataLoader(trainset,
                                #   batch_size = args.batch_size,
                                  batch_sampler=train_sampler,
                                  num_workers=8,
                                  collate_fn=newcollate_fn)
    else:
        trainset = datasets.get_class_obj("Train", args.dataset, 
                                 transform=None, 
                                 max_examples=None)


        train_sampler = samplers.get_class_obj("Train", trainset.labeln,
                                          trainset.unique_labels,
                                          args.num_batch,
                                          args.way,
                                          args.train_query,
                                          args.test_query)

        train_loader = DataLoader(trainset,
                                  batch_sampler=train_sampler,
                                  num_workers=8,
                                  collate_fn=collate_fn)
                            #   pin_memory=True)

                         

    valset = datasets.get_class_obj("Val",args.val_dataset, 
                                    args.qid2data,
                                    dataset_type=args.dataset_type,
                                    max_examples=None)

    vqg_net = VQGNetANS(ans_size, 
                    768, 
                    vocabulary_size,
                    encoder_dim=encoder_dim, decoder_dim=768,
                    embedding=embedding, ans_embedding=ans_embedding, scale_shift=args.scaling_shifting)
    
    _ = count_parameters(vqg_net, False)
    start_epoch = 0
    if len(args.model):
        vqg_net.load_state_dict(torch.load(args.model))
        start_epoch += int(args.model.split('/')[-1].split('_')[0].replace('epoch','')) 
    
    if args.mode == 'Train':
        maml = l2l.algorithms.MAML(vqg_net, lr=args.lr, first_order=False).cuda()
            
#         elif args.metalr_type == 'alpha':
#             adapt_transform = l2l.optim.ModuleTransform(l2l.nn.Scale)
#             maml = l2l.algorithms.GBML(
#                     module=vqg_net,
#                     transform=adapt_transform,
#                     lr=args.lr,
#                     adapt_transform=True,
#                     )
        maml.cuda()
        optimizer = optim.Adam(maml.parameters(), args.lr)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
    best_score = 0.
    for epoch in range(start_epoch + 1, args.epochs):
        if args.mode == 'Train':
            meta_train(args.num_runs, nlge, maml, optimizer, cross_entropy_loss, train_loader, word_dict, args.alpha_c, args, None)
            if epoch % 10 == 0:
                torch.save(vqg_net.state_dict(),
                            os.path.join(args.model_path,
                            'epoch{}_{}_{}_{}_{}_new{}shot_ans.pkl'.format(epoch, args.network, args.metalr_type, args.dataset_type, \
                                                                           lm_encoder, args.train_query)))

#         Uncomment for meta-testing after each epoch
        if args.mode == 'Test':
            
            score = meta_test(lm_encoder, nlge, vqg_net, cross_entropy_loss, valset, word_dict, ans_dict, args, None)
            exit()

            

def meta_train(runs, nlge, maml, optimizer, cross_entropy_loss,
                train_loader, word_dict, alpha_c,args, log_interval):

  
    meta_batch_size = args.meta_batch
    tqdm_gen = tqdm(range(runs))

    meta_train_error = AverageMeter()
    K = args.way
    N = args.train_query
    p = K*N
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=runs)
    
    
    for epoch in tqdm_gen:
        bleu_1 = AverageMeter()
        bleu_2 = AverageMeter()
        bleu_3 = AverageMeter()
        bleu_4 = AverageMeter()
        meteor = AverageMeter()
        rouge_l = AverageMeter()
        cider = AverageMeter()
        
        optimizer.zero_grad()
        counter = 0
        for data_batch in train_loader:
            if args.dataset_type == '7w':
                (imgs, captions, answers, cat, qids, _) = data_batch
            else:
                (imgs, captions, _,answers, cat, _, _) = data_batch
            counter += 1
            imgs = imgs.cuda().squeeze()
#             print(imgs.size())
            captions = captions.cuda()
            ans = answers.cuda()
            alengths = torch.Tensor(process_lengths(answers))
            img_train, img_test = imgs[:p], imgs[p:]
            cap_train, cap_test = captions[:p], captions[p:]
            ans_train, ans_test = ans[:p], ans[p:]
            alen_train, alen_test = alengths[:p], alengths[p:]

            train_batch = (img_train, cap_train, ans_train, alen_train)
            test_batch = (img_test, cap_test, ans_test, alen_test)

            # Compute meta-training loss
            learner = maml.clone()
            evaluation_error, scores = fast_adapt(train_batch, 
                                                  test_batch,
                                                  learner, alpha_c,
                                                  cross_entropy_loss,
                                                  1, word_dict, args, nlge)
            evaluation_error.backward()
            meta_train_error.update(evaluation_error.item())
            if counter == meta_batch_size:
                # Average the accumulated gradients and optimize
                break


        tqdm_gen.set_description('Error = {}'.format(meta_train_error.avg))



        # Average the accumulated gradients and optimize
        for name, param in maml.named_parameters():
            if param.requires_grad:
                try:
                    param.grad.data.mul_(1.0 / meta_batch_size)
                except:
                    print(name)
                
        optimizer.step()
        scheduler.step()



def meta_test(lm_encoder, nlge, maml, cross_entropy_loss, valset, word_dict, ans_dict, args, log_interval):
    K = args.way
    N = args.train_query
    p = K*N

    bleu_1 = AverageMeter()
    bleu_2 = AverageMeter()
    bleu_3 = AverageMeter()
    bleu_4 = AverageMeter()
    meteor = AverageMeter()
    rouge_l = AverageMeter()
    cider = AverageMeter()
    
    num_iter = args.num_iter
    model_name = args.network
    if not args.scaling_shifting:
        model_name += "_no_SS"
    os.makedirs('{}_{}_{}_ans_res{}shot'.format(model_name, args.dataset_type, lm_encoder, N), exist_ok=True)
    print(num_iter)
    for epoch in range(1,11):
        if args.dataset_type == '7w':
            val_sampler = samplers.get_class_obj("Val", '{}way_{}shot_cat_qids/cat2qid_testing_run{}.json'.format(K, N,epoch),
                                            -1,
                                            args.test_query)
        elif args.dataset_type == 'vqg':
            val_sampler = samplers.get_class_obj("Val", '{}way_{}shot_cat_qids/cats_qids_for_testing_run{}.json'.format(K, N,epoch),
                                            -1,
                                            args.test_query)

        else:
            val_sampler = samplers.get_class_obj("Val", '{}way{}shot/proposed_run{}.json'.format(K, N,epoch),
                                            -1,
                                            args.test_query)
        val_loader = DataLoader(valset,
                                batch_sampler=val_sampler,
                                num_workers=8,
                                collate_fn=newcollate_fn)
        preds_gts_dict = defaultdict(list)
        cat_qid_dict_list = []
        tqdm_gen = tqdm(val_loader)
        for idx,(imgs, captions, answer, cat, qids, img_ids) in enumerate(tqdm_gen):

            imgs = imgs.cuda().squeeze()
            
            captions = captions.cuda()
            ans = answer.cuda()
            alengths = torch.Tensor(process_lengths(answer))
            img_train, img_test = imgs[:p], imgs[p:]
            cap_train, cap_test = captions[:p], captions[p:]
            alen_train, alen_test = alengths[:p], alengths[p:]
            ans_train, ans_test = ans[:p], ans[p:]
            _, qids_test = qids[:p], qids[p:]

            img_ids_test = img_ids[p:]
            learner = copy.deepcopy(maml)
            learner.train()
            optimizer = optim.Adam(learner.parameters(), args.lr)

            best_val = 0.0
            best_score = None
            best_preds = None
            best_gts = None
            best_qids = None
            best_ans = None
            gts_pred_list = []
            
            for ix in range(1, num_iter + 1):
                train_batch = (img_train, cap_train, ans_train, alen_train)
                test_batch = (img_test, cap_test, ans_test, alen_test)
                if ix % args.eval_every == 0:
                    scores, preds, gts, eg = adapt_and_eval(train_batch, test_batch,
                                        word_dict, nlge,
                                        learner, args.alpha_c,
                                        cross_entropy_loss, args,1,
                                        optimizer, True)
                
                    if best_score is None:
                        best_score = scores
                        best_val = scores['CIDEr']
                        best_preds = preds
                        best_gts = gts
                        eg = colored(eg, 'green')
                        best_qids = copy.deepcopy(qids_test)
                        best_ans = copy.deepcopy(ans_test)
                    elif best_val < scores['CIDEr']:
                        best_score = scores
                        best_val = scores['CIDEr']
                        best_preds = preds
                        best_gts = gts
                        best_qids = copy.deepcopy(qids_test)
                        best_ans = copy.deepcopy(ans_test)
                        eg = colored(eg, 'green')
                    print(eg, "ANS = ", ans_dict.tokens_to_words(ans_test[0]), " IMG_ID = ", img_ids_test[0])
                    
                else:
                    adapt_and_eval(train_batch, test_batch,
                                        word_dict, nlge,
                                        learner, args.alpha_c,
                                        cross_entropy_loss, args,1,
                                        optimizer, False)

            for qid, npred, ngts, answ, img_test in zip(best_qids, best_preds, best_gts, best_ans, img_ids_test):

                ean =  ans_dict.tokens_to_words(answ)

                preds_gts_dict[idx].append([qid, npred, ngts, ean, img_test])
            bleu_1.update(best_score['Bleu_1'])
            bleu_2.update(best_score['Bleu_2'])
            bleu_3.update(best_score['Bleu_3'])
            bleu_4.update(best_score['Bleu_4'])
            meteor.update(best_score['METEOR'])
            rouge_l.update(best_score['ROUGE_L'])
            cider.update(best_score['CIDEr'])

            tqdm_gen.set_description('RUN {},'
                                     ' Bleu 1={:.4f}({:.4f}) Bleu 2={:.4f}'
                                     ' Bleu 3={:.4f} Bleu 4={:.4f}'
                                     ' METEOR={:.4f} ROUGE_L={:.4f}'
                                     ' CIDEr={:.4f}'.format(epoch, 
                                                            bleu_1.avg, bleu_1.stddev, bleu_2.avg,
                                                            bleu_3.avg, bleu_4.avg,
                                                            meteor.avg, rouge_l.avg,
                                                           cider.avg))
        with open('{}_{}_{}_ans_res{}shot/preds_gts_dict_run{}.json'.format(model_name, args.dataset_type, lm_encoder, N, epoch), 'w') as fid:
            fid.write(json.dumps(preds_gts_dict))    
    print('Bleu 1={:.4f}({:.4f}) Bleu 2={:.4f}({:.4f})'
          ' Bleu 3={:.4f}({:.4f}) Bleu 4={:.4f}({:.4f})'
          ' METEOR={:.4f}({:.4f}) ROUGE_L={:.4f}({:.4f})'
          ' CIDEr={:.4f}({:.4f})'.format(bleu_1.avg,bleu_1.stddev, 
                                         bleu_2.avg, bleu_2.stddev,
                                         bleu_3.avg, bleu_3.stddev,
                                         bleu_4.avg, bleu_4.stddev,
                                         meteor.avg, meteor.stddev,
                                         rouge_l.avg, rouge_l.stddev,
                                         cider.avg, cider.stddev))
    return cider.avg





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta-train of ProposedFewShotVQG')
    parser.add_argument('--num-batch', type=int, default=32, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--num-runs', type=int, default=30, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--num-iter', type=int, default=40, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--meta-batch', type=int, default=64, metavar='N',
                        help='batch size for meta update (default: 13)')
    parser.add_argument('--epochs', type=int, default=1000, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')
    parser.add_argument('--eval-every', type=int, default=5, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--dataset-type', choices=['7w', 'vqg', 'proposed'], default='vqg',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--network', choices=['effnet', 'resnet'], default='resnet',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--decoder-model', type=str, 
                        default='',
                        help='path to model')
    parser.add_argument('--decoder-type', choices=['lstm', 'transformer'], 
                        default='lstm',
                        help='path to model')
    parser.add_argument('--metalr-type', choices=['maml', 'alpha'], 
                        default='alpha',
                        help='path to model')
    parser.add_argument('--cat-model', type=str,
                        default='',
                        help='path to model')
    parser.add_argument('--scaling-shifting', type=bool, default=False)
    parser.add_argument('--mode', choices=['Train', 'Test'], default='Test')
    parser.add_argument('--bert-embed', type=str, default="")
    parser.add_argument('--bert-ans-embed', type=str, default="")

    parser.add_argument('--tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')

    
    parser.add_argument('--cat2qid', type=str, default='{}.json',
                        help='batch size for training (default: 64)')   
    # Data parameters.
    parser.add_argument('--qid2indx', type=str, default='new_qid2data.pkl',
                        help='batch size for training (default: 64)')
    parser.add_argument('--qid2data', type=str, default='new_qid2data.pkl',
                        help='batch size for training (default: 64)')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--vocab-path-ans', type=str,
                        default='data/processed/vocab_iq_ans.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='latest_train_resnet_no_SS_iq_dataset_new.hdf5',
                        help='Path for train annotation file.')
    parser.add_argument('--val-dataset', type=str,
                        default='val_img_id_no_SS_resnet.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-categories', type=int, default=16,
                        help='Number of answer types we use.')

#     parser.add_argument('--model', type=str, default='',
#                         help='path to model')
    parser.add_argument('--model', type=str, default='model/meta/epoch44_resnet_nobert_no_SS_new10shot_ans.pkl',
                        help='path to model')
    parser.add_argument('--model-path', type=str, default='model/meta/',
                        help='path to model')
    # parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--way', type=int, default=3) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=10) # (Shot) The number of meta train samples for each class in a task
    parser.add_argument('--test_query', type=int, default=50) # The number of meta test samples for each class in a task
    main(parser.parse_args())
