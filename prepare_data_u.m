function imdb=prepare_data_u(newLabel,progress)
load('cycle_train_part.mat');
pp = '/home/datasets/VeRi/image_train/';
pw = '/home/datasets/VeRi/image_train_256/';
newLabel=importdata(newLabel);
imdb.meta.sets=['train','test'];
file = dir(pp);
num = numel(imdb.images.data);
%num=0;
labelNum =  max(imdb.images.label);
%labelNum = 0;
counter_data=num+1;
counter_last = 0;
class = 0;
c_last = '';
cc = [];

for i=1:length(file)
    if file(i).name == "."
        continue
    end
    if file(i).name == ".."
        continue;
    end
    if file(i).name == "'Thumbs.db'" 
        continue;
    end
    if newLabel(i-2) == 0
        continue;
    end
    url = strcat(pp,file(i).name);
    im = imread(url);
    im = imresize(im,[256,256]); % save 256*256 image in advance for faster IO
    url256 = strcat(pw,file(i).name);
    imwrite(im,url256);
    imdb.images.data(counter_data) = cellstr(url256); 
    imdb.images.label(:,counter_data) = newLabel(i)+labelNum;
    imdb.images.flag(:,counter_data) = 0;
    counter_data = counter_data + 1;
end
s = counter_data-1;
imdb.images.set = ones(1,s);
imdb.images.set(:,randi(s,[round(0.1*s),1])) = 2;

%no validation for small class 
list = max(newLabel);
for i=1:list
    [mm,nn] = find(imdb.images.label==i);
    [KK,CC] = size(mm);
    if  CC<10
        imdb.images.set(nn)=1;
    end
end

save(['progress/url_data_cycle_unsupervised_',int2str(progress),'.mat'],'imdb','-v7.3');
end
