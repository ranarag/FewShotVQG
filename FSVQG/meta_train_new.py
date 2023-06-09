import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
# from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import os
from utils import AverageMeter, accuracy, calculate_caption_lengths

from utils import Vocabulary
from utils import get_glove_embedding, get_bert_embedding
from utils import DatasetLoader, CategoriesSampler, collate_fn
from utils import load_vocab
from utils import process_lengths
from utils import NLGEval
from torch.utils.data import DataLoader
import learn2learn as l2l
import pickle as pkl
import copy
# from tensordash.torchdash import Torchdash

from models import VQGNet, recast_category_embeds
torch.cuda.set_device(0)


from models import Autoencoder, CategoryEncoder, Encoder, Decoder, DecoderWithAttention
from tqdm import tqdm
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, targets):
    
    decode_lengths = process_lengths(targets)

    preds = pack_padded_sequence(preds, decode_lengths, batch_first=True, enforce_sorted=False)
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # some kind of regularization used by Show Attend and tell repo
    att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
        
        # loss
    loss = loss_fn(preds.data, targets.data)
    loss += att_regularization
    return loss


def fast_adapt(train_batch, test_batch, learner, alpha_c, loss_fn, adaptation_steps, word_dict, nlge):

    img_train, cap_train, cat_train = train_batch
    img_test, cap_test, cat_test = test_batch
    # Adapt the model>
    train_targets = cap_train[:, 1:]
    eval_targets = cap_test[:, 1:]

    for step in range(adaptation_steps):
        preds, alphas = learner(img_train, cat_train, cap_train)
        
        train_error = calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, train_targets)
        train_error /= preds.size(0)
        learner.adapt(train_error, allow_nograd=True)

    # Evaluate the adapted model
    preds, alphas = learner(img_test, cat_test, cap_test)
    # print(preds)
    valid_error= calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, eval_targets)
    valid_error /= preds.size(0)

    scores = 0.
    return valid_error, scores



def adapt_and_eval(train_batch, test_batch, word_dict, nlge, learner, alpha_c, loss_fn, adaptation_steps, optimizer, last):
    img_train, cap_train, cat_train = train_batch
    img_test, cap_test, cat_test = test_batch
    # Adapt the model
    train_targets = cap_train[:, 1:]
    eval_targets = cap_test[:, 1:]
    
#    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    for step in range(adaptation_steps):
        preds, alphas = learner(img_train, cat_train, cap_train)
        train_error = calc_loss_and_acc(loss_fn, alpha_c, preds, alphas, train_targets)
        train_error /= preds.size(0)
#         print(train_error.item())
        optimizer.zero_grad()
        train_error.backward()
        optimizer.step()
#        scheduler.step()
#         learner.adapt(train_error, allow_nograd=True,allow_unused=True)
#     if last:

    cat_embeds, cat_bias = learner.category_encoder(cat_test)
    recast_embeds, recast_bias = recast_category_embeds(cat_embeds, cat_bias, img_test.size(2))
#     best_score = None
#     best_val = 0.0
    n_img_features = img_test.mul(recast_embeds) + recast_bias
    n_img_features = n_img_features.permute(0, 2, 1)
    beam_size = 5
    predictions = []
    gts = []
    
    for i in range(img_test.size(0)):
        new_img_features = n_img_features[i].unsqueeze(0)
        new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2))
        outputs, alphas = learner.decoder.caption(new_img_features, word_dict, beam_size)
        output = word_dict.tokens_to_words(outputs)                    
        predictions.append(output)


        question = word_dict.tokens_to_words(cap_test[i])
        gts.append(question)
        
    

    scores = nlge.compute_metrics(ref_list=[gts], hyp_list=predictions)
    return scores, preds, gts



def main(args):
    writer = SummaryWriter()

    word_dict = load_vocab(args.vocab_path)
    vocabulary_size = len(word_dict)
    nlge = NLGEval(no_glove=True, no_skipthoughts=True)

#     embedding = get_bert_embedding('840B',
#                                         768,
#                                         word_dict)
#     print("DONE fetching")
    with open('bert_embedding.pkl', 'rb') as fid:
        embedding = pkl.load(fid)
#     embedding=None
    new_transform =  transforms.Compose([
        transforms.ToTensor()])

    trainset = DatasetLoader(args.dataset, 
                             transform=new_transform, 
                             max_examples=None)
    
    train_sampler = CategoriesSampler(trainset.labeln,
                                      trainset.unique_labels,
                                      args.num_batch,
                                      args.way,
                                      args.train_query,
                                      args.test_query)
    
    train_loader = DataLoader(trainset,
                            #   batch_size = args.batch_size,
                              batch_sampler=train_sampler,
                              num_workers=8,
                              collate_fn=collate_fn)
                            #   pin_memory=True)


    
 
    vqg_net = VQGNet(args.num_categories, 
                    args.hidden_size, 
                    vocabulary_size,
                    cat_model=args.cat_model,
                    decoder_model=args.decoder_model,
                    embedding=embedding)
 
    # vqg_net.retain_grad()
    maml = l2l.algorithms.MAML(vqg_net, lr=args.lr, first_order=False)
 

    




    optimizer = optim.Adam(maml.parameters(), args.lr)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
    best_score = 0.
    for epoch in range(args.epochs):
        bleu_1, bleu_2, bleu_3, bleu_4 , meteor, rouge_l, cider = meta_train(100, nlge, maml, optimizer, cross_entropy_loss, 
                                train_loader, word_dict, args.alpha_c, args.way, args.train_query, None)
#         scheduler.step()
        print('EPOCH {},'
              ' Bleu 1={:.4f} Bleu 2={:.4f}'
              ' Bleu 3={:.4f} Bleu 4={:.4f}'
              ' METEOR={:.4f} ROUGE_L={:.4f}'
              ' CIDEr={:.4f}'.format(epoch,
                                     bleu_1.avg, bleu_2.avg,
                                     bleu_3.avg, bleu_4.avg,
                                     meteor.avg, rouge_l.avg,
                                     cider.avg))

        #Uncomment for meta-testing after each epoch
        score = meta_test(100,nlge, vqg_net, cross_entropy_loss, val_loader, word_dict, args.alpha_c, args.way, args.train_query, None)
        if best_score < score:
            best_score = score
            torch.save(vqg_net.decoder.state_dict(),
                        os.path.join('devi_model', args.model_path, 'meta/','nbn_decoder/',
                        'best_bert_effnet_new10shot.pkl'))
            torch.save(vqg_net.category_encoder.state_dict(),
                        os.path.join('devi_model', args.model_path, 'meta/','nbn_category_encoder/',
                        'best_bert_effnet_new10shot.pkl'))
        #Uncomment for meta-testing after each epoch
            
        # meta_test(10,nlge, maml, cross_entropy_loss, val_loader, word_dict, args.alpha_c, args.way, args.train_query, None)

def meta_train(runs, nlge, maml, optimizer, cross_entropy_loss,
                train_loader, word_dict, alpha_c, way, shot, log_interval):


    meta_batch_size = train_loader.batch_sampler.n_batch  
    tqdm_gen = tqdm(range(runs))
    meta_loss = AverageMeter()
    tot_bleu_1 = AverageMeter()
    tot_bleu_2 = AverageMeter()
    tot_bleu_3 = AverageMeter()
    tot_bleu_4 = AverageMeter()
    tot_meteor = AverageMeter()
    tot_rouge_l = AverageMeter()
    tot_cider = AverageMeter()
    meta_train_error = AverageMeter()
    K = way
    N = shot
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
        for  (imgs, captions, _,  _, cat, _, _) in train_loader:
            imgs = imgs.cuda().squeeze()

            captions = captions.cuda()
            cat = cat.cuda()
            img_train, img_test = imgs[:p], imgs[p:]
            cap_train, cap_test = captions[:p], captions[p:]
            cat_train, cat_test = cat[:p], cat[p:]
            
            rand_idx = torch.randperm(cat_train.size(0))
            img_train = img_train[rand_idx]
            cap_train = cap_train[rand_idx]
            cat_train = cat_train[rand_idx]
            
            rand_idx = torch.randperm(cat_test.size(0))
            img_test = img_test[rand_idx]
            cap_test = cap_test[rand_idx]
            cat_test = cat_test[rand_idx]
            train_batch = (img_train, cap_train, cat_train)
            test_batch = (img_test, cap_test, cat_test)

            # Compute meta-training loss
            learner = maml.clone()
            # batch = tasksets.train.sample()
            evaluation_error, scores = fast_adapt(train_batch, 
                                                  test_batch,
                                                  learner, alpha_c,
                                                  cross_entropy_loss,
                                                  2, word_dict, nlge)

            evaluation_error.backward()
            meta_train_error.update(evaluation_error.item())

        tqdm_gen.set_description('Error = {}'.format(meta_train_error.avg))


        # Average the accumulated gradients and optimize
        for _, param in maml.named_parameters():
            if param.requires_grad:
                param.grad.data.mul_(1.0 / meta_batch_size)
        optimizer.step()
        scheduler.step()
        tot_bleu_1.update(bleu_1.avg)
        tot_bleu_2.update(bleu_2.avg)
        tot_bleu_3.update(bleu_3.avg)
        tot_bleu_4.update(bleu_4.avg)
        tot_meteor.update(meteor.avg)
        tot_rouge_l.update(rouge_l.avg)
        tot_cider.update(cider.avg)

    return tot_bleu_1, tot_bleu_2, tot_bleu_3, tot_bleu_4, tot_meteor, tot_rouge_l, tot_cider


def meta_test(runs, nlge, maml, cross_entropy_loss, val_loader, word_dict, alpha_c, way, train_query, log_interval):
    K = way
    N = train_query
    p = K*N
    # meta_batch_size = 32
    bleu_1 = AverageMeter()
    bleu_2 = AverageMeter()
    bleu_3 = AverageMeter()
    bleu_4 = AverageMeter()
    meteor = AverageMeter()
    rouge_l = AverageMeter()
    cider = AverageMeter()
    
    num_iter = 2
    for epoch in range(1):
        tqdm_gen = tqdm(val_loader)
        for (imgs, captions, _,  _, cat, _, qids) in tqdm_gen:

            
            
            imgs = imgs.cuda().squeeze()

            
            captions = captions.cuda()
            cat = cat.cuda()
            img_train, img_test = imgs[:p], imgs[p:]
            cap_train, cap_test = captions[:p], captions[p:]
            cat_train, cat_test = cat[:p], cat[p:]
            
            rand_idx = torch.randperm(cat_train.size(0))
            img_train = img_train[rand_idx]
            cap_train = cap_train[rand_idx]
            cat_train = cat_train[rand_idx]
            
            rand_idx = torch.randperm(cat_test.size(0))
            img_test = img_test[rand_idx]
            cap_test = cap_test[rand_idx]
            cat_test = cat_test[rand_idx]
            train_batch = (img_train, cap_train, cat_train)
            test_batch = (img_test, cap_test, cat_test)
            train_batch = (img_train, cap_train, cat_train)
            test_batch = (img_test, cap_test, cat_test)
            
            
            learner = copy.deepcopy(maml)
            learner.train()
            optimizer = optim.Adam(learner.parameters(), 0.001)
            #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter + 1)
            best_val = 0.0
            best_score = None

            for ix in range(num_iter + 1):
                
                scores, _, _ = adapt_and_eval(train_batch, test_batch,
                                        word_dict, nlge,
                                        learner, alpha_c,
                                        cross_entropy_loss,2,
                                        optimizer, ix // num_iter)
                if best_score is None:
                    best_score = scores
                    best_val = scores['CIDEr']
                elif best_val < scores['CIDEr']:
                    best_score = scores
                    best_val = scores['CIDEr']

            bleu_1.update(best_score['Bleu_1'])
            bleu_2.update(best_score['Bleu_2'])
            bleu_3.update(best_score['Bleu_3'])
            bleu_4.update(best_score['Bleu_4'])
            meteor.update(best_score['METEOR'])
            rouge_l.update(best_score['ROUGE_L'])
            cider.update(best_score['CIDEr'])

            tqdm_gen.set_description('RUN {},'
                                     ' Bleu 1={:.4f} Bleu 2={:.4f}'
                                     ' Bleu 3={:.4f} Bleu 4={:.4f}'
                                     ' METEOR={:.4f} ROUGE_L={:.4f}'
                                     ' CIDEr={:.4f}'.format(epoch, 
                                                            bleu_1.avg, bleu_2.avg,
                                                            bleu_3.avg, bleu_4.avg,
                                                            meteor.avg, rouge_l.avg,
                                                           cider.avg))


    return bleu_4.avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Meta-train of ProposedFewShotVQG')
    parser.add_argument('--num-batch', type=int, default=32, metavar='N',
                        help='batch size for training (default: 64)')
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
    parser.add_argument('--data', type=str, default='data/coco',
                        help='path to data images (default: data/coco)')
    parser.add_argument('--network', choices=['vgg19', 'resnet152', 'densenet161'], default='resnet152',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--decoder-model', type=str, 
                        default='devi_model/model/meta/nbn_decoder/best_bert_effnet_new10shot.pkl',
                        help='path to model')
    parser.add_argument('--cat-model', type=str,
                        default='devi_model/model/meta/nbn_category_encoder/best_bert_effnet_new10shot.pkl', 
                        help='path to model')
#     parser.add_argument('--cat-model', type=str, help='path to model')#, default='model/meta/nbn_category_encoder/best_imgnet_new.pkl')
#     parser.add_argument('--decoder-model', type=str, help='path to model')#, default='model/meta/nbn_decoder/best_imgnet_new.pkl')
    parser.add_argument('--tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')

    
    
    # Data parameters.
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='latest_train_effecientnet_iq_dataset.hdf5',
                        help='Path for train annotation file.')
    parser.add_argument('--val-dataset', type=str,
                        default='latest_val_ft_img_iq_dataset.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--train-dataset-weights', type=str,
                        default='data/processed/iq_train_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--val-dataset-weights', type=str,
                        default='data/processed/iq_val_dataset_weights.json',
                        help='Location of sampling weights for training set.')
    parser.add_argument('--cat2name', type=str,
                        default='data/processed/cat2name.json',
                        help='Location of mapping from category to type name.')
    parser.add_argument('--load-model', type=str, default=None,
                        help='Location of where the model weights are.')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='Size for randomly cropping images')

    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-examples', type=int, default=1024,
                        help='For debugging. Limit examples in database.')
    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-categories', type=int, default=16,
                        help='Number of answer types we use.')
    
    parser.add_argument('--model-path', type=str, default='model',
                        help='path to model')
    # parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--way', type=int, default=3) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=10) # (Shot) The number of meta train samples for each class in a task
    parser.add_argument('--test_query', type=int, default=10) # The number of meta test samples for each class in a task
    main(parser.parse_args())
