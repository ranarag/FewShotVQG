import argparse, json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
import os
from utils import AverageMeter, accuracy, calculate_caption_lengths, count_parameters

from utils import Vocabulary
from utils import get_glove_embedding, get_bert_embedding
from utils import datasets, collate_fn, newcollate_fn, samplers, NewDatasetLoader, NewCategoriesSampler7w, UnsupSampler7w
from utils import load_vocab
from utils import process_lengths
from utils import NLGEval
from torch.utils.data import DataLoader
import learn2learn as l2l
import pickle as pkl
import copy
from collections import defaultdict
from models import VQGNetANSCAT, recast_answer_embeds
# from torchsummary import summary
from models.Autoencoder import IEncoder
torch.cuda.set_device(0)
import bf3s.algorithms as algorithms
import bf3s.utils as utils

from termcolor import cprint, colored
from efficientnet_pytorch import EfficientNet
# from models import Autoencoder, CategoryEncoder, Encoder, Decoder
from tqdm import tqdm

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def calc_loss_and_acc(alpha_c, preds, alphas, targets):
    
    decode_lengths = process_lengths(targets)

    preds = pack_padded_sequence(preds, decode_lengths, batch_first=True, enforce_sorted=False)
    targets = pack_padded_sequence(targets, decode_lengths, batch_first=True, enforce_sorted=False)

        # some kind of regularization used by Show Attend and tell repo
    att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()
    loss = F.cross_entropy(preds.data, targets.data)
    loss += att_regularization
    return loss

########################################################################
def create_rotations_labels(batch_size, device):
    """Creates the rotation labels."""
    labels_rot = torch.arange(4, device=device).view(4, 1)

    labels_rot = labels_rot.repeat(1, batch_size).view(-1)
    return labels_rot


def create_4rotations_images(images, stack_dim=None):
    """Rotates each image in the batch by 0, 90, 180, and 270 degrees."""
    images_4rot = []
    for r in range(4):
        images_4rot.append(utils.apply_2d_rotation(images, rotation=r * 90))

    if stack_dim is None:
        images_4rot = torch.cat(images_4rot, dim=0)
    else:
        images_4rot = torch.stack(images_4rot, dim=stack_dim)

    return images_4rot


def rotation_task(rotation_classifier, features, labels_rotation):
    """Applies the rotation prediction head to the given features."""
    scores = rotation_classifier(features)
    assert scores.size(1) == 4

    
    loss = F.cross_entropy(scores, labels_rotation)

    return scores, loss


#########################################################################

def inference(test_batch, word_dict, learner, args):
    learner.eval()
    img_test, cap_test, cat_test, ans_test, alen_test = test_batch
    beam_size = 5
            
    predictions = []
    gts = []
    with torch.no_grad():
        
#         recast_embeds, recast_bias = recast_answer_embeds(cat_embeds, cat_bias, img_test.size(2))

#         n_img_features = img_test.mul(recast_embeds) + recast_bias
#         n_img_features = n_img_features.permute(0, 2, 1)
        if args.decoder_type == 'lstm':
            if args.scaling_shifting:
                n_img_features = learner.get_img_embeds(img_test, cat_test, ans_test, alen_test)
            else:
                n_img_features, tot_embeds = learner.get_img_embeds_no_SS(img_test, cat_test, ans_test, alen_test)



            for i in range(img_test.size(0)):
                new_img_features = n_img_features[i].unsqueeze(0)
#                 print(new_img_features.size())

                #if args.decoder_type == 'lstm':
                if args.scaling_shifting:
                    new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2))
                    outputs, alphas = learner.decoder.caption(new_img_features, word_dict, beam_size)
                else:
                    new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2))
    #                 print(cat_embeds.size())
                    new_tot_embeds = tot_embeds[i].unsqueeze(0)
                    new_tot_embeds = new_tot_embeds.expand(beam_size, new_tot_embeds.size(1))
                    outputs, alphas = learner.decoder.caption(new_img_features, word_dict, beam_size, new_tot_embeds)
                output = word_dict.tokens_to_words(outputs)
                predictions.append(output)


                question = word_dict.tokens_to_words(cap_test[i])
                gts.append(question)

        else: 
            cat_embeds = learner.category_encoder(cat_test)
            ans_embeds = learner.answer_encoder(ans_test, alen_test)
            tot_embeds = torch.cat([ans_embeds, cat_embeds], dim=1)
            for i in range(img_test.size(0)):
                new_img_features = img_test[i].unsqueeze(0).repeat(beam_size, 1, 1, 1)
                new_tot_embeds = tot_embeds[i].unsqueeze(0).repeat(beam_size, 1)
    #             new_img_features = new_img_features.expand(beam_size, new_img_features.size(1), new_img_features.size(2), new_img_features.size(3))
                outputs, alphas = learner.decoder.caption(new_img_features, new_tot_embeds, word_dict, beam_size)
                output = word_dict.tokens_to_words(outputs)
                predictions.append(output)


                question = word_dict.tokens_to_words(cap_test[i])
                gts.append(question)


        return predictions, gts

def adapt_and_eval(train_batch, test_loader, word_dict, nlge, learner, img_encoder, alpha_c, loss_fn, args, adaptation_steps, optimizer, last):
    learner.train()
    img_train, cap_train, cat_train, ans_train, alen_train = train_batch
#     img_test, cap_test, cat_test, ans_test, alen_test = test_batch
    # Adapt the model
    train_targets = cap_train[:, 1:]
#     eval_targets = cap_test[:, 1:]

    for step in range(adaptation_steps):
        if args.network == 'resnet':
            img_train = img_encoder(img_train)
        else:
            img_train = img_encoder.extract_features(img_train)
        preds, alphas = learner(img_train, cat_train, ans_train, alen_train, cap_train)
        train_error = calc_loss_and_acc(alpha_c, preds, alphas, train_targets)
#         train_error /= preds.size(0)

        optimizer.zero_grad()
        train_error.backward()
        optimizer.step()
    if last:
        preds, gts_test, img_ids_test, qids_test, ans_test_list = [], [], [], [], []
        with torch.no_grad():            
            for idx,(img_test, cap_test, answer, cat_test, qids, img_ids) in enumerate(test_loader):
                cap_test = cap_test.cuda()
                
                ans_test = answer.cuda()
                cat_test = cat_test.cuda()
                img_test = img_test.cuda().squeeze()
                ans_test_list += answer
                alen_test = torch.Tensor(process_lengths(answer))
                qids_test+= qids
                img_ids_test += img_ids
                if args.network == 'resnet':
                    img_test = img_encoder(img_test)
                else:
                    img_test = img_encoder.extract_features(img_test)
#                 print(img_test.size())
                test_batch = img_test, cap_test, cat_test, ans_test, alen_test
                predictions, gts = inference(test_batch, word_dict, learner, args)
                preds += predictions
                
                gts_test += gts

        eg = "Pred = {pred}; GT = {gt}; IMG_ID = {img_id}".format(pred=preds[0], gt=gts_test[0], img_id=img_ids_test[0])



        scores = nlge.compute_metrics(ref_list=[gts_test], hyp_list=preds)
        return scores, preds, gts_test, eg, img_ids_test, qids_test, ans_test_list


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
    if args.network == 'resnet':
        encoder_dim = 2048
    else:
        encoder_dim = 1536
    if not args.scaling_shifting:
        lm_encoder += "_no_SS"
    new_transform =  transforms.Compose([
        transforms.ToTensor()])
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224,
                                     scale=(1.00, 1.2),
                                     ratio=(0.75, 1.3333333333333333)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    if args.mode == 'Train':
        if args.dataset_type == '7w':
            trainset = datasets.get_class_obj("Val", args.dataset,
                                              args.qid2data,
                                              dataset_type=args.dataset_type,
                                              transform=transform,
                                              max_examples=None)


            train_sampler = NewCategoriesSampler7w('train_cat2qid7w.json',
                                              1000,
                                              args.way,
                                              args.train_query,
                                              args.test_query)
            train_sampler2 = UnsupSampler7w(args.qid2data, 4)

            train_loader = DataLoader(trainset,
                                      batch_sampler=train_sampler,
                                      num_workers=8,
                                      collate_fn=newcollate_fn)

            train_loader2 = DataLoader(trainset,
                                      batch_sampler=train_sampler2,
                                      num_workers=8,
                                      collate_fn=newcollate_fn)

        else:
            trainset = datasets.get_class_obj("Train", args.dataset, 
                                     transform=transform, 
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
            train_loader2 = DataLoader(trainset,
                                      batch_size=4,
                                      num_workers=8,
                                      collate_fn=collate_fn)
    else:


        valset = NewDatasetLoader(args.val_dataset, #'data/processed/val_img_id_resnet.hdf5', 
                                  args.qid2data,
                                  dataset_type=args.dataset_type,
                                  transform=transform,      
                                  max_examples=None)
        val_loader = DataLoader(valset,                                
                                num_workers=8,
                                collate_fn=newcollate_fn)
    vqg_net = VQGNetANSCAT(args.num_categories, args.hidden_size,
                           ans_size, 768, 
                           vocabulary_size,
                           encoder_dim=encoder_dim, decoder_dim=768,
                           embedding=embedding, ans_embedding=ans_embedding, scale_shift=args.scaling_shifting)

    if args.network == 'effnet':
        img_encoder = EfficientNet.from_pretrained('efficientnet-b3', advprop=True).cuda()
    else:
        img_encoder = IEncoder().cuda()
    if args.mode == 'Train':
        rot_classifier = nn.Sequential(
                            nn.Conv2d(encoder_dim, 64, 1),
                            nn.Flatten(),
                            nn.Linear(3136, 4, bias=True)).cuda()

        img_enc_params = list(img_encoder.parameters()) + list(rot_classifier.parameters())
        img_enc_optim = optim.Adam(img_enc_params, args.lr)

    _ = count_parameters(vqg_net, False)
    _ = count_parameters(img_encoder, False)
    start_epoch = 0
    if len(args.model):
        vqg_net.load_state_dict(torch.load(args.model))
        img_encoder.load_state_dict(torch.load(args.img_encoder))
        start_epoch += int(args.model.split('/')[-1].split('_')[0].replace('epoch','')) 

    
    if args.mode == 'Train':
        net_params = list(vqg_net.parameters()) + list(img_encoder.parameters())
        optimizer = optim.Adam(net_params, args.lr)
    else:
        optimizer = optim.Adam(vqg_net.parameters(), args.lr)
    cross_entropy_loss = nn.CrossEntropyLoss(reduction='mean').cuda()
    best_score = 0.
    print("OKAY")
    for epoch in range(start_epoch + 1, args.epochs+1):
        if args.mode == 'Train':
            train(args.num_runs, nlge, vqg_net, img_encoder, rot_classifier, optimizer, img_enc_optim, cross_entropy_loss, train_loader, train_loader2, word_dict, args.alpha_c, args, None)
            if epoch % 10 == 0:
                torch.save(vqg_net.state_dict(),
                            os.path.join(args.model_path,
                            'epoch{}_{}_{}_{}_new{}shot_non_meta_ans_cats.pkl'.format(epoch, args.network, args.dataset_type, lm_encoder, args.train_query)))

                torch.save(img_encoder.state_dict(),
                            os.path.join(args.model_path,
                            'img_encoder/epoch{}_{}_{}_{}_new{}shot_non_meta_ans_cats.pkl'.format(epoch, args.network, args.dataset_type,lm_encoder, args.train_query)))
    
#         Uncomment for meta-testing after each epoch
        
        if args.mode == 'Test':
            print("TESTING")
            score = test(lm_encoder, nlge, vqg_net, img_encoder, cross_entropy_loss, valset, word_dict, ans_dict, args, None)
            exit()

            

def train(runs, nlge, model, img_encoder, rot_classifier, optimizer, img_enc_optim, cross_entropy_loss,
                train_loader, train_loader2, word_dict, alpha_c, args, log_interval):

  
    meta_batch_size = args.meta_batch
    tqdm_gen = tqdm(range(len(train_loader)))

    train_error = AverageMeter()
    rot_error = AverageMeter()
    K = args.way
    N = args.train_query
    p = K*N
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=runs)
#     scheduler2 = optim.lr_scheduler.CosineAnnealingLR(im, T_max=runs)
    train1_iter = iter(train_loader)
    train2_iter = iter(train_loader2)
    for _ in tqdm_gen:
#             index += 1
#             print(index)

        try:
            data_batch1= train1_iter.next()
        except:
            train1_iter = iter(train_loader)
            data_batch1 = train1_iter.next()

        if args.dataset_type == '7w':
            (imgs, captions, answers, cat, qids, _) = data_batch1
        else:
            (imgs, captions, _,answers, cat, _, _) = data_batch1

        try:
            data_batch2 = train2_iter.next()
        except:
            train2_iter = iter(train_loader2)
            data_batch2 = train2_iter.next()

        if args.dataset_type == '7w':
            (imgs2, _, _, _, _, _) = data_batch2
        else:
            (imgs2, _, _,_, _, _, _) = data_batch2

        
    
#     for  (imgs, captions, _,answers, cat, _, _) in tqdm_gen:
#         counter += 1
        imgs = imgs.cuda().squeeze()
        train_batch_size = imgs.size(0)
        imgs2 = imgs2.cuda().squeeze()

        images = create_4rotations_images(imgs2)
        labels_rotation = create_rotations_labels(imgs2.size(0), imgs2.device)
        tot_imgs = torch.cat([imgs, images], dim=0)
        # Extract features from the train and test images.

        if args.network == 'effnet':
            tot_features = img_encoder.extract_features(tot_imgs)
        else:
            tot_features = img_encoder(tot_imgs)

        features, features2 = tot_features[train_batch_size:], tot_features[:train_batch_size]

        scores_rotation, loss_rotation = rotation_task(
            rot_classifier, features, labels_rotation
        )


#######################################################################
        alengths = torch.Tensor(process_lengths(answers))
        captions = captions.cuda()
        ans = answers.cuda()
        cat = cat.cuda()

#         img_train = imgs
        train_targets = captions[:, 1:]
        preds, alphas = model(features2, cat, ans, alengths, captions)
        train_loss = calc_loss_and_acc(alpha_c, preds, alphas, train_targets)
#         train_loss /= preds.size(0)

        # Compute meta-training loss
        tot_loss = train_loss + loss_rotation
        optimizer.zero_grad()
        img_enc_optim.zero_grad()
        tot_loss.backward()
        

        optimizer.step()
        img_enc_optim.step()
        rot_error.update(loss_rotation.item())
        train_error.update(train_loss.item())

        tqdm_gen.set_description('Train Error = {}; Rot Error = {}'.format(train_error.avg, rot_error.avg))



 


def test(lm_encoder, nlge, model, img_encoder, cross_entropy_loss, valset, word_dict, ans_dict, args, log_interval):
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
    os.makedirs('{}_{}_{}_ans_cat_res{}shot'.format(model_name, args.dataset_type, lm_encoder, N), exist_ok=True)
    print(num_iter)
    if args.dataset_type == '7w':
        sampler_file = '{}way_{}shot_cat_qids/cat2qid_testing_run{}.json'
    elif args.dataset_type == 'vqg':
        sampler_file = '{}way_{}shot_cat_qids/cats_qids_for_testing_run{}.json'
    else:
        sampler_file = '{}way{}shot/proposed_run{}.json'
    os.makedirs('{}_{}_{}_cat_res{}shot'.format(model_name, args.dataset_type, lm_encoder, N), exist_ok=True)
    for epoch in range(1,11):        
        val_sampler = samplers.get_class_obj("Val", sampler_file.format(K, N,epoch),
                                            p,
                                            args.test_query)
        val_loader = DataLoader(valset,
                                batch_sampler=val_sampler,
                                num_workers=8,
                                collate_fn=newcollate_fn)
        
        preds_gts_dict = defaultdict(list)
        cat_qid_dict_list = []
        tqdm_gen = tqdm(val_loader)
        for idx,(imgs, captions, answer, cat, qids, img_ids) in enumerate(tqdm_gen):

            
            test_sampler = samplers.get_class_obj("ValVQG", sampler_file.format(K, N,epoch),
                                                 idx,
                                                 p,
                                                10)
            test_loader = DataLoader(valset,
                                batch_sampler=test_sampler,
                                num_workers=8,
                                collate_fn=newcollate_fn)

            img_train = imgs.cuda().squeeze()
            cap_train = captions.cuda()
            ans_train = answer.cuda()
            cat_train = cat.cuda()
            alen_train = torch.Tensor(process_lengths(answer))
            qid_train = qids
            
            learner = copy.deepcopy(model)
            n_img_encoder = copy.deepcopy(img_encoder)
            n_img_encoder.train()
            learner.train()
            tot_params = list(learner.parameters()) + list(n_img_encoder.parameters())
            optimizer = optim.Adam(tot_params, 0.002)

            best_val = 0.0
            best_score = None
            best_preds = None
            best_gts = None
            best_qids = None
            best_ans = None
            gts_pred_list = []
            
            for ix in range(num_iter + 1):
                train_batch = (img_train, cap_train, cat_train, ans_train, alen_train)
#                 test_batch = (img_test, cap_test, cat_test, ans_test, alen_test)
                if ix % 5 == 0:
                    scores, preds, gts, eg, img_ids_test, qids_test, ans_test = adapt_and_eval(train_batch, test_loader,
                                        word_dict, nlge,
                                        learner, n_img_encoder, args.alpha_c,
                                        cross_entropy_loss, args, 1,
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
                    print(eg, "ANS  = ", ans_dict.tokens_to_words(ans_test[0]))
                    
                else:
                    adapt_and_eval(train_batch, test_loader,
                                        word_dict, nlge,
                                        learner, n_img_encoder, args.alpha_c,
                                        cross_entropy_loss, args,1,
                                        optimizer, False)
#                 scheduler.step()
                

            for qid, npred, ngts, answ, img_test in zip(best_qids, best_preds, best_gts, ans_test, img_ids_test):

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
        
        with open('{}_{}_{}_ans_cat_res{}shot/preds_gts_dict_run{}'.format(model_name, args.dataset_type, lm_encoder, N, epoch), 'w') as fid:
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
    parser.add_argument('--num-iter', type=int, default=20, metavar='N',
                        help='batch size for training (default: 64)')
    parser.add_argument('--meta-batch', type=int, default=64, metavar='N',
                        help='batch size for meta update (default: 13)')
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='number of epochs to train for (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-3, metavar='LR',
                        help='learning rate of the decoder (default: 1e-4)')
    parser.add_argument('--step-size', type=int, default=5,
                        help='step size for learning rate annealing (default: 5)')
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A',
                        help='regularization constant (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L',
                        help='number of batches to wait before logging training stats (default: 100)')

    parser.add_argument('--network', choices=['effnet', 'resnet'], default='resnet',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--dataset-type', choices=['7w', 'vqg', 'metavqg'], default='vqg',
                        help='Network to use in the encoder (default: vgg19)')
    parser.add_argument('--decoder-model', type=str, 
                        default='',
                        help='path to model')
    parser.add_argument('--cat-model', type=str,
                        default='',
                        help='path to model')
    parser.add_argument('--scaling-shifting', type=bool, default=False)
    parser.add_argument('--mode', choices=['Train', 'Test'], default='Train')
    parser.add_argument('--decoder-type', choices=['lstm', 'transformer'], default='lstm')
    parser.add_argument('--bert-embed', type=str, default="bert_embedding.pkl")
    parser.add_argument('--bert-ans-embed', type=str, default="bert_embedding_ans.pkl")
#     parser.add_argument('--bert-embed', type=str, default="")
#     parser.add_argument('--bert-ans-embed', type=str, default="")
    parser.add_argument('--tf', action='store_true', default=False,
                        help='Use teacher forcing when training LSTM (default: False)')

    
    
    # Data parameters.
#     parser.add_argument('--qid2data', type=str, default='new_qid2data.pkl',
#                         help='batch size for training (default: 64)')
    parser.add_argument('--qid2data', type=str, default='new_qid2data.pkl',
                        help='batch size for training (default: 64)')
    parser.add_argument('--vocab-path', type=str,
                        default='data/processed/vocab_iq.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--vocab-path-ans', type=str,
                        default='data/processed/vocab_iq_ans.json',
                        help='Path for vocabulary wrapper.')
    parser.add_argument('--dataset', type=str,
                        default='latest_train_iq_dataset_new.hdf5',
                        help='Path for train annotation file.')
    parser.add_argument('--val-dataset', type=str,
                        default='val_img_id_raw.hdf5',
                        help='Path for train annotation json file.')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--hidden-size', type=int, default=768,
                        help='Dimension of lstm hidden states.')
    parser.add_argument('--num-categories', type=int, default=16,
                        help='Number of answer types we use.')
    
#     parser.add_argument('--model', type=str, default='model/meta/epoch87_effnet_bert_new1shot_ans_cats.pkl',
#                         help='path to model')
    parser.add_argument('--model', type=str, default='',
                        help='path to model')
    parser.add_argument('--img-encoder', type=str, default='',
                        help='path to model')
    parser.add_argument('--model-path', type=str, default='model/meta/',
                        help='path to model')
    # parser.add_argument('--num_batch', type=int, default=100) # The number for different tasks used for meta-train
    parser.add_argument('--way', type=int, default=3) # Way number, how many classes in a task
    parser.add_argument('--train_query', type=int, default=10) # (Shot) The number of meta train samples for each class in a task
    parser.add_argument('--test_query', type=int, default=50) # The number of meta test samples for each class in a task
    main(parser.parse_args())
