%递进式训练模型%
addpath('./test');
addpath('./dbscan-master')
addpath('./progress')

iteration = 6;

for progress = 1: iteration 
    disp(['**********************the ',progress,'th training**********************']);
    
    if progress == 1
        ff = test_gallery_query_crazy_u('./data/pretrained_model.mat',progress);%现有模型%
    else
        ff = test_gallery_query_crazy_u('',progress);
    end
    
    disp('**********************  t-sne begin**********************');
    %t-sne降维%
    [M,N] = size(ff');
    
    train_X = ff(1:N,:);
    no_dims = 2;
   
    initial_dims = 20;
    perplexity = 60;
    mappedX = tsne(train_X, [], no_dims, initial_dims, perplexity);
    
    disp('**********************  t-sne end**********************');   
    disp('**********************  dbscan begin**********************');
    
    %DBSCAN%
    A = mappedX;
    numData=size(A,1);
    Kdist=zeros(numData,1);
    [IDX,Dist]=knnsearch(A(2:numData,:),A(1,:));
    Kdist(1)=Dist;
    for i=2:size(A,1)
        [IDX,Dist] = knnsearch(A([1:i-1,i+1:numData],:),A(i,:));
        Kdist(i)=Dist;
    end
    [sortKdist,sortKdistIdx]=sort(Kdist,'descend');
    distX=[1:numData]';
    plot(distX,sortKdist,'r+-','LineWidth',2);
    set(gcf,'position',[1000 340 350 350]);
    grid on;
    
    limitDist = 5;
    if progress < 4
        limitDist = 2;
    end
    
    eps = sortKdist(1)/limitDist; 
    [IDX, isnoise] = DBSCAN2(A,1,10);
    
    PlotClusterinResult(A,IDX);
    
    idmax = max(IDX');
    centers = zeros(int32(idmax),2);
    [M,N] = size(A);
    selectDatasets = zeros(M,N);
    
    IDX = IDX';
    
    for i = 1:idmax
        index = find(IDX==i);
        cluster = A(index,:);
        [outK,center] = kmeans(cluster, 1);
        centers(i,:) = center(1,:);
        
        [M,N] = size(index);
        
        limitRange=4/5;
        
        for j = 1:N
            tempIndex = index(j);
            value = A(tempIndex,:);
            if pdist2(value,center)<=(sortKdist(1)*limitRange)
                selectDatasets(tempIndex,:) = value;
            end
        end
    end

    [M,N] = size(A);
    weights = zeros(M,idmax);
    
    for i = 1:M
        if IDX(i) == 0 || selectDatasets(i,1) == 0
            weights(i,:)=0;
        else
            value = A(i,:);
            for j = 1:idmax
                weight = pdist2(value,centers(j,:));
                weights(i,j) = weight(1);
            end
        end
    end
    
    %%����������
    newDatasets = find(weights(:,1)~=0);
    flagMatrix = zeros(M,1);
    
    %%挑选数据集并置label
    for i = 1:M
        if IDX(i) == 0 || selectDatasets(i,1) == 0
            flagMatrix(i,:)=0;
        else
            flagMatrix(i,:) = IDX(i);
        end
    end
    
    [weightW,weightH]=size(weights);
    [NEWL,NEWW]=size(newDatasets);
    weightsMid=zeros(NEWL,weightH);
    weights3=zeros(NEWL,weightH);
    k=1;
    
    for i = 1:weightW
        for j=1:weightH
            if weights(i,j)==0
                continue;
            end
            [sortWdist,sortWdistIdx]=sort(weights(i,:),'descend');
            WK = find(sortWdistIdx==j);
            weightsMid(k,j)=(1-weights(i,j)/max(weights(i,:)))*WK;
            
            if weightsMid(k,j)<100
                weights3(k,j)=0;
            else
                weights3(k,j)=weightsMid(k,j);
            end
            
            if j == weightH
                k=k+1;
            end
        end 
    end
    save('progress/weights.mat','weights3');    
    save('progress/newLabel_Unsupervised.mat','flagMatrix','-v7.3');  
    save (['progress/params/cycle_',int2str(progress),'.mat'])
    
    disp('**********************  dbscan end**********************');
    disp('**********************  create dataset begin**********************');
    %生成数据集%
    datasets=prepare_data_u('progress/newLabel_Unsupervised.mat',progress);
    disp('**********************  create dataset end**********************');
    
    currentSourceImagesCount = 9867;
    %生成权重%
    weights = prepare_weight_u('progress/weights.mat',progress,currentSourceImagesCount);
    disp('**********************  create weight end**********************');
    
    disp('**********************  training begin**********************');
    %使用新数据集训练%
    train_id_net_res_pal_unsupervised(progress);
    disp('**********************  training end**********************');
end
