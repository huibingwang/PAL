function newWeight=prepare_weight_u(weight,progress,number)

load(weight);
load(['progress/url_data_cycle_unsupervised_',int2str(progress)]);
maxCount = max(imdb.images.label);
count = numel(weights3(:,1));
[M,N]=size(weights3);
newWeight=zeros(M,maxCount);

for i=1:number
    for j=1:N
        newWeight(i,j)=0;
    end
end

for i=1:count
    newWeight(i+number,maxCount-N+1:maxCount)=weights3(i,:);
end

save(['progress/weights_',int2str(progress),'.mat'],'newWeight');

end
