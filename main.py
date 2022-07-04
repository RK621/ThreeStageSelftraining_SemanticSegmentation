import click
import sys, time
import numpy as np
import os
import math
import itertools
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F


import para1, cf2
from script_lib import mk_pl_loss_meanT
from tools import job_helper


# benchmark mode in cudnn
torch.backends.cudnn.benchmark = True

# wraping 
def wrap_ind(lab, indx, indy, msk):
    # lab: batch times H times W; 
    # indx,... can be list or np array 
    for i in range(len(lab)): new = lab[i][indx[i], indy[i]]; new[msk[i]<0.5] = 255; lab[i] = new
    return lab
    
@job_helper.job('train_seg', enumerate_job_names=False)
def train_seg(submit_config: job_helper.SubmitConfig, dataset, model, arch, freeze_bn,
                              opt_type, sgd_momentum, sgd_nesterov, sgd_weight_decay,
                              learning_rate, lr_sched, lr_step_epochs, lr_step_gamma, lr_poly_power,
                              bin_fill_holes, crop_size, aug_hflip, aug_vflip, 
                              aug_hvflip, aug_scale_hung, aug_max_scale, aug_scale_non_uniform, aug_rot_mag,  
                              rampup, num_epochs, 
                              iters_per_epoch, batch_size, n_sup, n_unsup, n_val, split_seed, split_path, 
                              val_seed, save_preds, save_model, num_workers, preds_all):
    settings = locals().copy()
    del settings['submit_config']

    from architectures import network_architectures
    import torch.utils.data
    from datapipe import datasets
    from datapipe import seg_data, seg_transforms, seg_transforms_cv
    import evaluation
    from tools import optim_weight_ema, lr_schedules


    # set subbatch sizes 
    if cf2.use_unlabelleddata: 
        batch_size = cf2.pl_batchsize1 + cf2.pl_batchsize2
        print('note: batch_size is changed! fixed gt/nogt ratio: %d/%d'%(cf2.pl_batchsize1, cf2.pl_batchsize2) )



    if crop_size == '':
        crop_size = None
    else:
        crop_size = [int(x.strip()) for x in crop_size.split(',')]

    torch_device = torch.device('cuda:0')
    s_torch_device = torch_device
    assert s_torch_device == torch_device

    assert not( bin_fill_holes )


    #
    # Load data sets
    #
    ds_dict = datasets.load_dataset(dataset, n_val, val_seed, n_sup, n_unsup, split_seed, split_path, pl_path=cf2.pl_path)

    ds_src = ds_dict['ds_src']
    ds_tgt = ds_dict['ds_tgt'] # = ds_src  
    tgt_val_ndx = ds_dict['val_ndx_tgt']
    src_val_ndx = ds_dict['val_ndx_src'] if ds_src is not ds_tgt else None
    test_ndx = ds_dict['test_ndx_tgt'] # = None 
    sup_ndx = ds_dict['sup_ndx']
    unsup_ndx = ds_dict['unsup_ndx']

    n_classes = ds_src.num_classes
    root_n_classes = math.sqrt(n_classes)

    print('Loaded data')



    # Build network
    NetClass = network_architectures.seg.get(arch)

    if cf2.use_unlabelleddata: 
        student_net = NetClass(ds_src.num_classes, doublehead=cf2.g_branch_op).to(s_torch_device)
        print('\nnote: student net use \hat(g)')
    else: 
        student_net = NetClass(ds_src.num_classes).to(torch_device)

    if opt_type == 'adam':
        student_optim = torch.optim.Adam([
            dict(params=student_net.pretrained_parameters(), lr=learning_rate * 0.1),
            dict(params=student_net.new_parameters(), lr=learning_rate)])
    elif opt_type == 'sgd':
        student_optim = torch.optim.SGD([
            dict(params=student_net.pretrained_parameters(), lr=learning_rate * 0.1),
            dict(params=student_net.new_parameters(), lr=learning_rate)],
            momentum=sgd_momentum, nesterov=sgd_nesterov, weight_decay=sgd_weight_decay)
    else:
        raise ValueError('Unknown opt_type {}'.format(opt_type))

    if cf2.use_unlabelleddata:  # setting up mean teacher
        teacher_net = NetClass(ds_src.num_classes, doublehead=cf2.g_branch_op).to(torch_device)
        for p in teacher_net.parameters(): p.requires_grad = False
        teacher_optim = optim_weight_ema.EMAWeightOptimizer(teacher_net, student_net, cf2.meanT_alpha)
        eval_net = teacher_net
    elif True:  # not using teacher student model
        teacher_net = student_net
        teacher_optim = None
        eval_net = student_net

    BLOCK_SIZE = student_net.BLOCK_SIZE
    NET_MEAN, NET_STD = seg_transforms.get_mean_std(ds_tgt, student_net)

    if freeze_bn:
        if not hasattr(student_net, 'freeze_batchnorm'):
            raise ValueError('Network {} does not support batchnorm freezing'.format(arch))

    clf_crossent_loss = nn.CrossEntropyLoss(ignore_index=255)

    print('network built')


    if iters_per_epoch == -1:
        iters_per_epoch = len(unsup_ndx) // batch_size
    total_iters = iters_per_epoch * num_epochs

    lr_epoch_scheduler, lr_iter_scheduler = lr_schedules.make_lr_schedulers(
        optimizer=student_optim, total_iters=total_iters, schedule_type=lr_sched,
        step_epochs=lr_step_epochs, step_gamma=lr_step_gamma, poly_power=lr_poly_power
    )


    train_transforms = []
    eval_transforms = []

    if crop_size is not None:
        if aug_scale_hung:
            train_transforms.append(seg_transforms_cv.SegCVTransformRandomCropScaleHung(crop_size, (0, 0), uniform_scale=not aug_scale_non_uniform))
        elif aug_max_scale != 1.0 or aug_rot_mag != 0.0:
            assert 0==1 # not implemnted for pl_arr
            train_transforms.append(seg_transforms_cv.SegCVTransformRandomCropRotateScale(
                crop_size, (0, 0), rot_mag=aug_rot_mag, max_scale=aug_max_scale, uniform_scale=not aug_scale_non_uniform, constrain_rot_scale=True))
        else:
            train_transforms.append(seg_transforms_cv.SegCVTransformRandomCrop(crop_size, (0, 0)))
    else:
        if aug_scale_hung:
            raise NotImplementedError('aug_scale_hung requires a crop_size')

    if aug_hflip or aug_vflip or aug_hvflip:
        train_transforms.append(
            seg_transforms_cv.SegCVTransformRandomFlip(aug_hflip, aug_vflip, aug_hvflip))
    train_transforms.append(seg_transforms_cv.SegCVTransformNormalizeToTensor(NET_MEAN, NET_STD))
    eval_transforms.append(seg_transforms_cv.SegCVTransformNormalizeToTensor(NET_MEAN, NET_STD))


    trans_ = train_transforms
    train_sup_ds = ds_src.dataset(labels=True, mask=False, xf=False, pair=False,
                                  transforms=seg_transforms.SegTransformCompose(trans_),
                                  pipeline_type='cv')

    # additonal transform for unlabeled data if needed 
    if cf2.meanT_addAugPost:  # RandomAug
        unsup_transform = train_transforms + [seg_transforms_cv.RandAugmentTransformPost()]
    else: unsup_transform = train_transforms

    train_unsup_ds = ds_src.dataset(labels=False, mask=False, xf=False, pair=False,
                                    transforms=seg_transforms.SegTransformCompose(unsup_transform),
                                    pipeline_type='cv')
    eval_ds = ds_src.dataset(labels=True, mask=False, xf=False, pair=False, transforms=seg_transforms.SegTransformCompose(eval_transforms), pipeline_type='cv')
    pl_gen_ds = ds_tgt.dataset(labels=False, mask=False, xf=False, pair=False, transforms=eval_transforms[0], pipeline_type='cv', include_indices=True) # new

    collate_fn = seg_data.SegCollate(BLOCK_SIZE)


    # Train data pipeline: data loaders
    if cf2.use_unlabelleddata:  # semisupervised framework (including mean teacher)
        train_unsup_loader_0 = None
        train_unsup_loader_1 = None
        #
        sup_sampler = seg_data.RepeatSampler(torch.utils.data.SubsetRandomSampler(sup_ndx))
        train_sup_loader = torch.utils.data.DataLoader(train_sup_ds, cf2.pl_batchsize1, sampler=sup_sampler, collate_fn=collate_fn, num_workers=num_workers)
        #
        unsup_sampler = seg_data.RepeatSampler(torch.utils.data.SubsetRandomSampler(unsup_ndx))

        train_unsup_loader_whole = torch.utils.data.DataLoader(train_unsup_ds, cf2.pl_batchsize2, sampler=unsup_sampler, collate_fn=collate_fn, num_workers=num_workers)
        pl_gen_loader = torch.utils.data.DataLoader(pl_gen_ds, batch_size, collate_fn=collate_fn, num_workers=num_workers)

    else:  # supervised piplie
        sup_sampler = seg_data.RepeatSampler(torch.utils.data.SubsetRandomSampler(sup_ndx))
        train_sup_loader = torch.utils.data.DataLoader(train_sup_ds, batch_size, sampler=sup_sampler,
                                                   collate_fn=collate_fn, num_workers=num_workers)
        train_unsup_loader_0 = None
        train_unsup_loader_1 = None

    # Eval pipeline
    src_val_loader, tgt_val_loader, test_loader = datasets.eval_data_pipeline( ds_src, ds_tgt, src_val_ndx, tgt_val_ndx, test_ndx, batch_size, collate_fn, NET_MEAN, NET_STD, num_workers)

    # Report setttings
    print('Settings:')
    print(', '.join(['{}={}'.format(key, settings[key]) for key in sorted(list(settings.keys()))]))

    # Report dataset size
    print('Dataset:')
    print('len(sup_ndx)={}'.format(len(sup_ndx)))
    print('len(unsup_ndx)={}'.format(len(unsup_ndx)))
    if ds_src is not ds_tgt:
        print('len(src_val_ndx)={}'.format(len(tgt_val_ndx)))
        print('len(tgt_val_ndx)={}'.format(len(tgt_val_ndx)))
    else:
        print('len(val_ndx)={}'.format(len(tgt_val_ndx)))
    if test_ndx is not None:
        print('len(test_ndx)={}'.format(len(test_ndx)))

    if n_sup != -1:
        print('sup_ndx={}'.format(sup_ndx.tolist()))


    best_tgt_miou = -1
    best_epoch = 0

    eval_net_state = {key: value.detach().cpu().numpy() for key, value in eval_net.state_dict().items()}

    # Create iterators
    train_sup_iter = iter(train_sup_loader)
    train_unsup_iter_0 = iter(train_unsup_loader_0) if train_unsup_loader_0 is not None else None
    train_unsup_iter_1 = iter(train_unsup_loader_1) if train_unsup_loader_1 is not None else None
    train_unsup_iter_whole = iter(train_unsup_loader_whole) if (cf2.use_unlabelleddata) else None # whole set (w and w/o gt) 



    iter_i = 0
    print('Training...')
    for epoch_i in range(num_epochs):
        if lr_epoch_scheduler is not None:
            lr_epoch_scheduler.step(epoch_i)

        if (teacher_optim is not None) and ('ema_alpha_list' in dir(cf2)):
            teacher_optim.ema_alpha = cf2.ema_alpha_list[epoch_i]

        t1 = time.time()

        if rampup > 0:
            ramp_val = network_architectures.sigmoid_rampup(epoch_i, rampup)
        else:
            ramp_val = 1.0

        student_net.train()
        if teacher_net is not student_net:
            teacher_net.train()

        if freeze_bn:
            student_net.freeze_batchnorm()
            if teacher_net is not student_net:
                teacher_net.freeze_batchnorm()

        sup_loss_acc = 0.0
        consistency_loss_acc = 0.0
        conf_rate_acc = 0.0
        n_sup_batches = 0
        n_unsup_batches = 0
        pl_loss_acc = 0.0


        src_val_iter = iter(src_val_loader) if src_val_loader is not None else None
        tgt_val_iter = iter(tgt_val_loader) if tgt_val_loader is not None else None


        for sup_batch in itertools.islice(train_sup_iter, iters_per_epoch):

            if lr_iter_scheduler is not None:
                lr_iter_scheduler.step(iter_i)
            student_optim.zero_grad()


            #
            # Supervised branch
            #

            if cf2.use_unlabelleddata:  # varibles: sup_loss, pl_loss 
                pl_batch = next( train_unsup_iter_whole ) # sup_batch['image'].dtype: torch.float32; sup_batch['labels'].dtype: torch.int64;  sup_batch['pl'].dtype: torch.int64
                tmp = torch.cat((sup_batch['image'], pl_batch['image']), 0)
                #
                t_batch_x = tmp.to(torch_device) # teacher first 

                t_logits_sup, _ = teacher_net(t_batch_x)

                if cf2.meanT_addAugPost: # additional noise for student at pl_batch['image']
                    batch_x = torch.cat((sup_batch['image'], pl_batch['image_post']), 0).to(s_torch_device) # student input
                else:
                    batch_x = tmp.to(s_torch_device) # student second 

                logits_sup, logits_pl = student_net(batch_x)
                sup_y = sup_batch['labels']
                batch_y = torch.cat((sup_y, torch.zeros(pl_batch['image'].shape[:1]+sup_y.shape[1:], dtype=sup_y.dtype)+255), 0).to(s_torch_device)
                sup_loss = clf_crossent_loss(logits_sup, batch_y[:,0,:,:])
                #

                t_pred_y = torch.argmax(t_logits_sup, dim=1).detach().cpu().numpy()

                new_pred_y = t_pred_y

                if cf2.meanT_addAugPost: # adjusting the predicted label 
                    new_pred_y = t_pred_y
                    new_pred_y[sup_batch['image'].shape[0]:] = wrap_ind( new_pred_y[sup_batch['image'].shape[0]:], pl_batch['indx'][:,0,...].numpy(), pl_batch['indy'][:,0,...].numpy(), pl_batch['indmsk'][:,0,...].numpy())
                    if 'pl' in pl_batch.keys(): pl_batch['pl'] = pl_batch['pl_post']

                assert cf2.meanT_losstype == 'crossentropy'
                meanT_loss = clf_crossent_loss 

                ramp_up = None
                pl_loss = mk_pl_loss_meanT(new_pred_y, sup_batch, pl_batch, logits_sup, logits_pl, cf2, meanT_loss, s_torch_device, dataset=dataset, ramp_up=ramp_up)

                if cf2.meanT_supweight is None: 
                    suppl_loss = sup_loss + cf2.meanT_conweight * pl_loss
                else: 
                    suppl_loss = cf2.meanT_supweight * sup_loss + cf2.meanT_conweight * pl_loss
                suppl_loss.backward()

            else: # supervised based line
                batch_x = sup_batch['image'].to(torch_device)
                batch_y = sup_batch['labels'].to(torch_device)

                logits_sup = student_net(batch_x)
                sup_loss = clf_crossent_loss(logits_sup, batch_y[:,0,:,:])
                sup_loss.backward()



            student_optim.step()
            if teacher_optim is not None:
                teacher_optim.step()

            sup_loss_val = float(sup_loss.detach())
            if np.isnan(sup_loss_val):
                print('NaN detected; network dead, bailing.')
                return

            sup_loss_acc += sup_loss_val
            n_sup_batches += 1
            iter_i += 1

            if cf2.use_unlabelleddata:
                pl_loss_acc += float(pl_loss.detach())
            

        sup_loss_acc /= n_sup_batches
        if n_unsup_batches > 0:
            consistency_loss_acc /= n_unsup_batches
            conf_rate_acc /= n_unsup_batches
        if cf2.use_unlabelleddata:
            pl_loss_acc /= n_sup_batches

        eval_net.eval()

        if ds_src is not ds_tgt:
            src_iou_eval = evaluation.EvaluatorIoU(ds_src.num_classes, bin_fill_holes)
            with torch.no_grad():
                for batch in src_val_iter:
                    batch_x = batch['image'].to(torch_device)
                    batch_y = batch['labels'].numpy()

                    logits = eval_net(batch_x)
                    pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                    for sample_i in range(len(batch_y)):
                        src_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)

            src_iou = src_iou_eval.score()
            src_miou = src_iou.mean()
        else:
            src_iou_eval = src_iou = src_miou = None

        tgt_iou_eval = evaluation.EvaluatorIoU(ds_tgt.num_classes, bin_fill_holes)
        if cf2.use_unlabelleddata: 
            tgt_iou_eval_pl = evaluation.EvaluatorIoU(ds_tgt.num_classes+1, bin_fill_holes)
        with torch.no_grad():
            for batch in tgt_val_iter:
                batch_x = batch['image'].to(torch_device)
                batch_y = batch['labels'].numpy()

                if cf2.use_unlabelleddata: 
                    logits_sup, logits_pl = eval_net(batch_x)
                    pred_y = torch.argmax(logits_sup, dim=1).detach().cpu().numpy()
                    pred_y_pl = torch.argmax(logits_pl, dim=1).detach().cpu().numpy()
                    for sample_i in range(len(batch_y)):
                        tgt_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)
                        tgt_iou_eval_pl.sample(batch_y[sample_i, 0], pred_y_pl[sample_i], ignore_value=255)
                else: 
                    logits = eval_net(batch_x)
                    pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()
                    for sample_i in range(len(batch_y)):
                        tgt_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)

        tgt_iou = tgt_iou_eval.score()
        tgt_miou = tgt_iou.mean()
        if cf2.use_unlabelleddata: 
            tgt_iou_pl = tgt_iou_eval_pl.score()
            tgt_miou_pl = tgt_iou_pl.mean() * (ds_tgt.num_classes+1)/(ds_tgt.num_classes) # adjusting for the background classes 

        t2 = time.time()

        if best_tgt_miou < tgt_miou: # best model
            model_path = os.path.join(submit_config.run_dir, "model_best.pth")
            torch.save(eval_net.state_dict(), model_path)
            best_tgt_miou = tgt_miou


        if ds_src is not ds_tgt:
            print('\nEpoch: {} took: {:.3f} TRAIN_clf_loss: {:.6f} consistency_loss: {:.6f} conf_rate: {:.6f} '
                  'SRC_VAL_mIoU: {:.6f} TGT_VAL_mIoU: {:.6f} best_val_mIoU: {:.6f}'.format(
                epoch_i + 1, t2 - t1, sup_loss_acc, consistency_loss_acc, conf_rate_acc, src_miou, tgt_miou, best_tgt_miou))
        else:
            if cf2.use_unlabelleddata: 
                print('Epoch: {} took: {:.3f} TRAIN_clf_loss: {:.6f} pl_loss: {:.6f}  VAL_mIoU: {:.6f} VAL_pl_mIoU: {:.6f} Best_mIoU: {:.6f}'.format(
                    epoch_i + 1, t2 - t1, sup_loss_acc, pl_loss_acc, tgt_miou, tgt_miou_pl, best_tgt_miou))

            else: 
                print('Epoch: {} took: {:.3f} TRAIN_clf_loss: {:.6f} consistency_loss: {:.6f} conf_rate: {:.6f} VAL_mIoU: {:.6f} Best_mIoU: {:.6f}'.format(
                    epoch_i + 1, t2 - t1, sup_loss_acc, consistency_loss_acc, conf_rate_acc, tgt_miou, best_tgt_miou))

    if save_model:
        model_path = os.path.join(submit_config.run_dir, "model.pth")
        torch.save(eval_net, model_path)


    # make predictions on all images 
    if preds_all:
        outdir = os.path.join(submit_config.run_dir, 'preds_all')
        from script_lib import make_preds_for_alldata
        if dataset == 'cityscapes': 
            make_preds_for_alldata(eval_net, outdir, torch_device, pl_gen_ds, iscityscapes=True)
        elif dataset == 'pascal_aug':
            make_preds_for_alldata(eval_net, outdir, torch_device, pl_gen_ds) 


    if save_preds and not(cf2.use_unlabelleddata):
        # 
        bestcpt = torch.load(os.path.join(submit_config.run_dir, "model_best.pth"))
        eval_net.load_state_dict(bestcpt)
        #
        out_dir = os.path.join(submit_config.run_dir, 'preds')
        os.makedirs(out_dir, exist_ok=True)
        with torch.no_grad():
            for batch in tgt_val_loader:
                batch_x = batch['image'].to(torch_device)
                batch_ndx = batch['index'].numpy()

                logits = eval_net(batch_x)
                pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                for sample_i, sample_ndx in enumerate(batch_ndx):
                    ds_tgt.save_prediction_by_index(out_dir, pred_y[sample_i].astype(np.uint32), sample_ndx)
    else:
        out_dir = None



    if test_loader is not None:
        test_iou_eval = evaluation.EvaluatorIoU(ds_tgt.num_classes, bin_fill_holes)
        with torch.no_grad():
            for batch in test_loader:
                batch_x = batch['image'].to(torch_device)
                batch_ndx = batch['index'].numpy()

                logits = eval_net(batch_x)
                pred_y = torch.argmax(logits, dim=1).detach().cpu().numpy()

                for sample_i, sample_ndx in enumerate(batch_ndx):
                    if save_preds:
                        ds_tgt.save_prediction_by_index(out_dir, pred_y[sample_i].astype(np.uint32), sample_ndx)
                    test_iou_eval.sample(batch_y[sample_i, 0], pred_y[sample_i], ignore_value=255)

        test_iou = test_iou_eval.score()
        test_miou = test_iou.mean()

        print('FINAL TEST: mIoU={:.3%}'.format(test_miou))
        print('-- TEST {}'.format(', '.join(['{:.3%}'.format(x) for x in test_iou])))




# --- parameter control --- 

para1.job_desc = ''
para1.dataset = 'cityscapes' # ['camvid', , 'pascal', 'pascal_aug', 'isic2017']
para1.model = 'mean_teacher' # ['mean_teacher', 'pi']
para1.arch = 'resnet101_deeplab_imagenet'
para1.freeze_bn = False
para1.opt_type = 'adam' # 'adam', 'sgd'
para1.sgd_momentum = 0.9 
para1.sgd_nesterov = False
para1.sgd_weight_decay = 5e-4
para1.learning_rate = 1e-4 
para1.lr_sched = 'none' # ['none', 'stepped', 'cosine', 'poly']
para1.lr_step_epochs = ''
para1.lr_step_gamma = 0.1
para1.lr_poly_power = 0.9 
para1.bin_fill_holes = False
para1.crop_size = '321,321' 
para1.aug_hflip = False 
para1.aug_vflip = False 
para1.aug_hvflip = False 
para1.aug_scale_hung = False 
para1.aug_max_scale = 1.0 
para1.aug_scale_non_uniform = False 
para1.aug_rot_mag = 0.0 
para1.rampup = -1 
para1.num_epochs = 300 
para1.iters_per_epoch = -1 
para1.batch_size = 10 
para1.n_sup = 100 
para1.n_unsup = - 1
para1.n_val = -1
para1.split_seed = 12345
para1.split_path = None 
para1.val_seed = 131 
para1.save_preds = False 
para1.save_model = False 
para1.num_workers = 4 
para1.preds_all = True

# these are parameters in cityscapes 
def set_cityscapes():
    para1.dataset='cityscapes'
    para1.arch='resnet101_deeplab_coco' 
    para1.freeze_bn = True
    para1.batch_size=4 
    para1.learning_rate=3e-5
    para1.iters_per_epoch=1000 
    para1.num_epochs=20 
    para1.split_path = 'data/cityscapes/split_0.pkl' 
    para1.save_preds = True
    para1.crop_size='256,512' 
    para1.aug_hflip = True
    para1.n_sup=100;


# there are parameters in pascal with coco pretrained 
def set_pascal():
    para1.dataset='pascal_aug'
    para1.arch='resnet101_deeplab_coco'  # deeplabv2 COCO 
    para1.freeze_bn = True
    para1.batch_size=10 # different from cityscapes 
    para1.learning_rate=3e-5
    para1.iters_per_epoch=1000 
    para1.num_epochs=10 
    para1.split_path = 'data/splits/pascal_aug/split_0.pkl' 
    para1.save_preds = True
    para1.crop_size='321,321' # different from cityscapes 
    para1.aug_hflip = True
    para1.n_sup=212
    para1.aug_scale_hung = True # different from cityscapes 



cf2.pl_path = None # this controls whether PL is loaded
cf2.pl_data_option = 'pure' 
cf2.pl_batchsize1 = 1 # with ground truth
cf2.pl_batchsize2 = 4 # without ground truth  
#
cf2.use_unlabelleddata = False
cf2.meanT_alpha = 0.99
cf2.meanT_conweight = 0.5
cf2.meanT_supweight = None # for Pascal
cf2.meanT_losstype = 'crossentropy' # or MSE
cf2.g_branch_op = 1 # 1: stage 2 g_hat; 2: stage 3 g_tilde
cf2.meanT_addAugPost = False
cf2.ema_alpha_list = [0.99]*20 + [0.995]*20 + [0.9975]*10 + [0.999]*100


if len(sys.argv) > 1: modid=np.double(sys.argv[1]); print('note: input modelid=%.16f'%modid)
else: raise AssertionError('must input options') # error  

# prediction are saved at results/train_seg/...
def get_PL_path(s): return 'results/train_seg/%s/preds_all.zip'%s


# cityscapes fully supervised baseline (imagenet) 
if modid==1:
    set_cityscapes()
    para1.arch = 'resnet101_deeplab_imagenet'; # imagenet 
    para1.job_desc='cityscapes_deeplab2i_100_sup' 

# cityscapes stage 2
if modid==2:
    set_cityscapes()
    cf2.use_unlabelleddata = True; para1.num_epochs=60; para1.arch = 'resnet101_deeplab_imagenet'; # imagenet 
    cf2.pl_batchsize1 = 1; cf2.pl_batchsize2 = 4; cf2.meanT_addAugPost = True;

    cf2.pl_path = get_PL_path( 'cityscapes_deeplab2i_100_sup' )
    cf2.g_branch_op = 1;  cf2.meanT_supweight = 0.5; para1.job_desc='cityscapes_deeplab2i_100_stage2';


# cityscapes stage 3
if modid==3:
    set_cityscapes()
    cf2.use_unlabelleddata = True; para1.num_epochs=60; para1.arch = 'resnet101_deeplab_imagenet'; # imagenet 
    cf2.pl_batchsize1 = 1; cf2.pl_batchsize2 = 4; cf2.meanT_addAugPost = True;

    cf2.pl_path = get_PL_path( 'cityscapes_deeplab2i_100_stage2' )
    cf2.g_branch_op = 2; cf2.meanT_supweight = 0.5; para1.n_sup=100; para1.job_desc='cityscapes_deeplab2i_100_stage3'; 


# pascal fully supervised baseline (coco)
if modid==4: 
    set_pascal(); 
    para1.job_desc='pascalaug_deeplab2coco_212_sup'


# pascal stage 2
if modid==5: 
    set_pascal(); cf2.use_unlabelleddata = True; para1.num_epochs=60; 
    cf2.pl_batchsize1 = 1; cf2.pl_batchsize2 = 9; 
    cf2.meanT_addAugPost = True; cf2.meanT_supweight = 1;

    cf2.pl_path = get_PL_path('pascalaug_deeplab2coco_212_sup')
    cf2.g_branch_op = 1; para1.job_desc='pascalaug_deeplab2coco_212_stage2'; 


# pascal stage 3
if modid==6: 
    set_pascal(); cf2.use_unlabelleddata = True; para1.num_epochs=60; 
    cf2.pl_batchsize1 = 1; cf2.pl_batchsize2 = 9; 
    cf2.meanT_addAugPost = True; cf2.meanT_supweight = 1;

    cf2.pl_path = get_PL_path('pascalaug_deeplab2coco_212_stage2')
    cf2.g_branch_op = 2; para1.job_desc='pascalaug_deeplab2coco_212_stage3'; 


params = {}
for i in dir(para1): 
    if not(i.startswith('__')): 
        params[i] = getattr(para1, i)
train_seg.submit(**params)







