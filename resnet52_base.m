function net = resnet52_market_gan_base(number)
netStruct = load('./data/imagenet-resnet-50-dag.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
net.removeLayer('fc1000');
net.removeLayer('prob');
%---------setting1
for i = 1:numel(net.params)
    if(mod(i,2)==0)
        net.params(i).learningRate=0.02;
    else net.params(i).learningRate=0.001;
    end
    name = net.params(i).name;
    if(name(1)=='b')
        net.params(i).weightDecay=0;
    end
end
%---

net.params(1).learningRate = 0.0001;
dropoutBlock = dagnn.DropOut('rate',0.9);
net.addLayer('dropout',dropoutBlock,{'pool5'},{'pool5d'},{});
fc751Block = dagnn.Conv('size',[1 1 2048 number],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc751',fc751Block,{'pool5d'},{'prediction'},{'fc751f','fc751b'});
%The proposed LSR
net.addLayer('weightedlabelsmooth',dagnn.Loss('loss','weightedlabelsmooth'),{'prediction','label','weight','flag'},'objective');

net.addLayer('top1err', dagnn.Loss('loss', 'classerror'), ...
    {'prediction','label','weight','flag'}, 'top1err') ;
net.addLayer('top5err', dagnn.Loss('loss', 'topkerror', ...
    'opts', {'topK',5}), ...
    {'prediction','label','weight','flag'}, 'top5err') ;

net.initParams();
end

