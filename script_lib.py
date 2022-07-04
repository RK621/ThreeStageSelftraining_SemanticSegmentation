
import sys
import torch 
import numpy as np


def make_preds_for_alldata(eval_net, outdir, torch_device, pl_gen_ds, rate=None, iscityscapes=False):

    import zipfile
    import cv2, os
    import time

    torch.backends.cudnn.benchmark = False # faster for different sizes 
    eval_net.eval()

    print('\n\nstart predicting...') 
    os.makedirs(outdir, exist_ok=True)
    time1 = time.time()
    nn = len(pl_gen_ds.ds.sample_names)
    with torch.no_grad():
        for i in range(0,nn,10):
            if i%1000==0: print('processing no. %d'%i); sys.stdout.flush() 
            lgts, names = [], []
            for j in range(i,min(nn,i+10)): 
                tmp = pl_gen_ds.__getitem__(j)

                batch_x = torch.from_numpy(tmp['image'][np.newaxis,...]).to(torch_device)
                tmplog = eval_net(batch_x)
                if type(tmplog) is tuple: tmplog = tmplog[0]
                lgts.append(tmplog) 
                names.append(pl_gen_ds.ds.sample_names[tmp['index']]) 
            for j in range(len(lgts)): 
                predj = torch.argmax(lgts[j], dim=1).detach().cpu().numpy()[0] # shape [1,?,?]
                lgts[j] = None

                cv2.imwrite(os.path.join(outdir, names[j]+'.png') , predj) 
                if iscityscapes:
                    cv2.imwrite(os.path.join(outdir, names[j].split('/')[-1]+ '_leftImg8bit_org.png') , predj)  # names[j] ends with numbers


    print('time spent = ', time.time()-time1) 

    d = outdir if outdir[-1]!='/' else outdir[:-1]
    print('note: writting all files to .zip', d)
    with zipfile.ZipFile(d + '.zip','w') as zipf: 
        for i in os.walk(outdir):
            for n in i[2]:
                zipf.write(''.join((i[0],'/',n)), n) # the second parameter specifies the file name;
    # remove all png files 
    import shutil
    print('note: removing dir ', outdir)
    shutil.rmtree(outdir)

    torch.backends.cudnn.benchmark = True
    return 


def mk_pl_loss_meanT(new_pred_y, sup_batch, pl_batch, logits_sup, logits_pl, cf2, clf_crossent_loss, s_torch_device, dataset='cityscapes', ramp_up=None):
    
    if dataset == 'cityscapes': bdidx = 19
    elif dataset == 'pascal_aug': bdidx = 21
    else: assert 0==1 # not implemented error 


    if cf2.pl_data_option in ['confirm4_bd', 'bd', 'pure']: # 
        new_pred_y[new_pred_y==bdidx]=255 # no boundaries 
        unsup_y = torch.from_numpy(new_pred_y).to(s_torch_device) # student net 
        pl_loss_1 = clf_crossent_loss(logits_sup, unsup_y) # meanT for 1st branch  
        tmppla, tmpplb = sup_batch['pl'], pl_batch['pl']
        tmppl =torch.cat((tmppla, tmpplb), 0).to(s_torch_device)
        pl_loss_2 = clf_crossent_loss(logits_pl, tmppl[:,0,...])
        if ramp_up is not None:
            rate_up = np.exp(-5*(1-ramp_up)**2)
            return  (pl_loss_1*rate_up + pl_loss_2)*0.5
        else:  # task 1 regularization; task 2 PL  
            return (pl_loss_1 + pl_loss_2)*0.5

    if cf2.pl_path is not None:
        tmppla, tmpplb = sup_batch['pl'], pl_batch['pl']
        tmppl =torch.cat((tmppla, tmpplb), 0).detach().cpu().numpy()
        if cf2.pl_data_option in ['ego_only', 'bd_only']: new_pred_y[tmppl[:,0,...]==255] = 255
        if cf2.pl_data_option in ['bd_only']: new_pred_y[tmppl[:,0,...]==bdidx]=bdidx # for cityscape

    if cf2.meanT_singleBranch: new_pred_y[new_pred_y==bdidx]=255 # no boundaries 
        
    unsup_y = torch.from_numpy(new_pred_y).to(s_torch_device) # student net 

    if cf2.meanT_singleBranch: pl_loss = clf_crossent_loss(logits_sup, unsup_y)
    else: pl_loss = clf_crossent_loss(logits_pl, unsup_y)

    if ramp_up is not None: 
        rate_up = np.exp(-5*(1-ramp_up)**2)*0.5 # to be consistent with the above 0.5 
        return pl_loss*rate_up
    else: 
        return pl_loss  # no 0.5 factor here




