% clear all; close all;
addpath('./utilities');
run('D:\AcademicSoftware\matconvnet-1.0-beta25\matlab\vl_setupnn.m') ;

total_seq_num = 6;
total_frame_num = 17; % to be tuned
GOP_size = 8; % to be tuned

key_subrate = 0.5;
subrate = 0.1;  % uniform Sampling Rate

block_size = 32; % Block Size for BCS

im = zeros(288,352,17,'single');

imgPSNR  = zeros(total_seq_num, total_frame_num);
imgSSIM  = zeros(total_seq_num, total_frame_num);

savet = 1;
 
 net = load(['./models/net' num2str(subrate) '.mat']);
 net = dagnn.DagNN.loadobj(net.net);

showResult  = 0;
useGPU      = 0;
pauseTime   = 0;
if useGPU
    net.move('gpu');
end
 
for kk = 1:6   % for each sequence   modified 1:6
    switch kk
         case 1
            sequence_name = 'akiyo_cif';
        case 2
            sequence_name = 'coastguard_cif';
        case 3
            sequence_name = 'Foreman_cif';
        case 4
            sequence_name = 'Mother_daughter_cif';
        case 5
            sequence_name = 'Paris_cif';
        case 6
            sequence_name = 'Silent_cif';   
    end
        
    Opts.sequence_name = sequence_name;
    
    for i = 1 : total_frame_num    % for each frame of this sequence
        filename = ['./Sequences/' sequence_name '/' sequence_name '_' num2str(i) '.png'];
        im(:,:,i) = im2single(imread(filename));
    end
    
    if useGPU
        im = gpuArray(im);
    end
        
    net.eval({'input',im(:,:,1:9)});
    Recon1 = zeros(288,352,8,'like',im);
    
    Recon1(:,:,1) = net.vars(net.getVarIndex('keyFdr_pred')).value;    
    Recon1(:,:,2) = net.vars(net.getVarIndex('nkeyF1dr_pred')).value;
    Recon1(:,:,3) = net.vars(net.getVarIndex('nkeyF2dr_pred')).value;
    Recon1(:,:,4) = net.vars(net.getVarIndex('nkeyF3dr_pred')).value;
    Recon1(:,:,5) = net.vars(net.getVarIndex('nkeyF4dr_pred')).value;
    Recon1(:,:,6) = net.vars(net.getVarIndex('nkeyF5dr_pred')).value;
    Recon1(:,:,7) = net.vars(net.getVarIndex('nkeyF6dr_pred')).value;
    Recon1(:,:,8) = net.vars(net.getVarIndex('nkeyF7dr_pred')).value;
    
    if useGPU
        Recon1 = gather(Recon1);
    end
    
    for j=1:8
        [psnr,ssim] = Cal_PSNRSSIM(im2uint8(im(:,:,j)),im2uint8(Recon1(:,:,j)),0,0);
        imgPSNR(kk,j) = psnr;
        imgSSIM(kk,j) = ssim;
        if savet
            if j == 1
                imwrite(im2uint8(Recon1(:,:,j)),['./VCSNet-gop8-doubleRef-results/' num2str(subrate),'/' sequence_name, '/' sequence_name,'_',num2str(j),'_rate_',num2str(key_subrate),...
                    '_PSNR_',num2str(psnr),'_SSIM_',num2str(ssim),'.png']);
            else
                imwrite(im2uint8(Recon1(:,:,j)),['./VCSNet-gop8-doubleRef-results/' num2str(subrate),'/' sequence_name, '/' sequence_name,'_',num2str(j),'_rate_',num2str(subrate),...
                    '_PSNR_',num2str(psnr),'_SSIM_',num2str(ssim),'.png']);            
            end
        end
    end    
    
    net.eval({'input',im(:,:,9:17)});
    Recon2 = zeros(288,352,9,'like',im);
    
    Recon2(:,:,1) = net.vars(net.getVarIndex('keyFdr_pred')).value;    
    Recon2(:,:,2) = net.vars(net.getVarIndex('nkeyF1dr_pred')).value;
    Recon2(:,:,3) = net.vars(net.getVarIndex('nkeyF2dr_pred')).value;
    Recon2(:,:,4) = net.vars(net.getVarIndex('nkeyF3dr_pred')).value;
    Recon2(:,:,5) = net.vars(net.getVarIndex('nkeyF4dr_pred')).value;
    Recon2(:,:,6) = net.vars(net.getVarIndex('nkeyF5dr_pred')).value;
    Recon2(:,:,7) = net.vars(net.getVarIndex('nkeyF6dr_pred')).value;
    Recon2(:,:,8) = net.vars(net.getVarIndex('nkeyF7dr_pred')).value;
    Recon2(:,:,9) = net.vars(net.getVarIndex('keyF9dr_pred')).value;
    
    if useGPU
        Recon2 = gather(Recon2);
    end
    
    for j=9:17
        [psnr,ssim] = Cal_PSNRSSIM(im2uint8(im(:,:,j)),im2uint8(Recon2(:,:,j-8)),0,0);
        imgPSNR(kk,j) = psnr;
        imgSSIM(kk,j) = ssim;
        if savet
            if j == 9
                imwrite(im2uint8(Recon2(:,:,j-8)),['./VCSNet-gop8-doubleRef-results/' num2str(subrate),'/' sequence_name, '/' sequence_name,'_',num2str(j),'_rate_',num2str(key_subrate),...
                    '_PSNR_',num2str(psnr),'_SSIM_',num2str(ssim),'.png']);
            else
                imwrite(im2uint8(Recon2(:,:,j-8)),['./VCSNet-gop8-doubleRef-results/' num2str(subrate),'/' sequence_name, '/' sequence_name,'_',num2str(j),'_rate_',num2str(subrate),...
                    '_PSNR_',num2str(psnr),'_SSIM_',num2str(ssim),'.png']);            
            end
        end
    end
    

    disp([mean(imgPSNR(kk,1:16)),mean(imgSSIM(kk,1:16))])

end


if savet
    xlswrite(['./VCSNet-gop8-doubleRef-results/' num2str(subrate) '/Rate_' num2str(subrate) '_PSNR_results_All_sequences.xlsx'],imgPSNR);
    xlswrite(['./VCSNet-gop8-doubleRef-results/' num2str(subrate) '/Rate_' num2str(subrate) '_SSIM_results_All_sequences.xlsx'],imgSSIM);
end