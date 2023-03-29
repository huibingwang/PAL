
function train_id_net_res_pal_unsupervised(progress,varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the data
% -------------------------------------------------------------------------
global weights;
% Load character dataset
imdb = load(['progress/url_data_cycle_unsupervised_',int2str(progress),'.mat']) ;
imdb = imdb.imdb;
% 
% global weights;
weights = load(['progress/weights_',int2str(progress),'.mat']);
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_base(max(imdb.images.label));
net.conserveMemory = true;net.meta.normalization.averageImage = reshape([105.6920,99.1345,97.9152],1,1,3);

% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 32;
opts.train.continue = false;
opts.train.gpus = 1;
opts.train.prefetch = false ;
opts.train.nesterovUpdate = true ;
opts.train.expDir = ['./progress/data/pal_iter_',int2str(progress)];
opts.train.derOutputs = {'objective', 1} ;
%opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
%opts.train.constraint = 100;
opts.train.learningRate = [0.1*ones(1,30),0.01*ones(1,10),0.001*ones(1,10)] ;
opts.train.weightDecay = 0.0001;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;
% Call training function in MatConvNet
[net,info] = cnn_train_dag(net, imdb,@getBatch,opts) ;

% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
global weights;
% --------------------------------------------------------------------
[M,N] = size(weights.newWeight);

if max(batch)>M
    max(batch);
end

weight = weights.newWeight(batch,:);
% --------------------------------------------------------------------
im_url = imdb.images.data(batch) ;
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.85,1],...
    'Interpolation', 'bicubic','NumThreads',8);
labels = imdb.images.label(batch);
oim = bsxfun(@minus,im{1},opts.averageImage);
flag = imdb.images.flag(batch);
inputs = {'data',gpuArray(oim),'label',labels,'weight',weight','flag',flag};
