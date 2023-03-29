clear;
p = '/home/datasets/VehicleID_V1.0/VehicleID_V1.0/train_test_split/train_list.txt';
pp = '/home/datasets/cycle/bounding_box_train/';
pw = '/home/datasets/cycle/image_train_Cycle_256/';
label =importdata( 'data/train.txt');
imdb.meta.sets=['train','test'];
file = dir(pp);
counter_data=1;
counter_last = 1;
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
    url = strcat(file(i).folder,'/',file(i).name);
    im = imread(url);
    im = imresize(im,[256,256]); % save 256*256 image in advance for faster IO
    url256 = strcat(pw,file(i).name);
    imwrite(im,url256);
    imdb.images.data(counter_data) = cellstr(url256); 
    c = strsplit(file(i).name,'_');
    d = strsplit(c{4},'.');
    if(~isequal(str2num(c{2}),c_last))
        class = class + 1;
        fprintf('%d::%d\n',class,counter_data-counter_last);
        cc=[cc;counter_data-counter_last];
        c_last = str2num(c{2});
        counter_last = counter_data;
    end
    imdb.images.label(:,counter_data) = class;
    imdb.images.flag(:,counter_data) = 1;
    counter_data = counter_data + 1;
end
s = counter_data-1;
imdb.images.set = ones(1,s);
imdb.images.set(:,randi(s,[round(0.1*s),1])) = 2;

%no validation for small class 
cc = [cc;9];
cc = cc(2:end);
list = find(imdb.images.set==2);
for i=1:numel(list)
    if cc(imdb.images.label(list(i)))<3
        imdb.images.set(list(i))=1;
    end
end

save('cycle_train_part.mat','imdb','-v7.3');
