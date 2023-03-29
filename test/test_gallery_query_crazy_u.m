% In this file, we densely extract the feature
% We extract feature from 256,256 image and mirrored image.
% -----Please change the file path in this script. ----
function ff=getFeatures(data,progress)

addpath ..;
if progress == 1
    netStruct = load(data);
else
    netStruct = load(['./progress/data/pal_iter_',int2str(progress-1),'/net-epoch-50.mat']);
end
%--------add L2-norm
net = dagnn.DagNN.loadobj(netStruct.net);
net.addLayer('lrn_test',dagnn.LRN('param',[4096,0,1,0.5]),{'pool5'},{'pool5n'},{});
clear netStruct;
net.mode = 'test';
net.move('gpu') ;
net.conserveMemory = true;
im_mean = net.meta(1).normalization.averageImage;
p = dir('/home/datasets/VeRi/image_train/*jpg');
ff = [];
%------------------------------
for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('/home/datasets/VeRi/image_train/',p(i+j-1).name);
        imt = imresize(imread(str),[224,224]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end

save(['./progress/test/train_plm_veri_iter_',int2str(progress),'.mat'],'ff','-v7.3');
end
