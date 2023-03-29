load 'querySet.mat'
cycleQuerySet = querySet{1};
cycleQuerySet = cycleQuerySet';
save('cycleQuerySet.mat','cycleQuerySet');

load 'testSet.mat'
cycleTestSet = testSet{1};
cycleTestSet = cycleTestSet';
save('cycleTestSet.mat','cycleTestSet');

load 'querySet.mat'
originalQuerySet = querySet{2};
originalQuerySet = originalQuerySet';
save('originalQuerySet.mat','originalQuerySet');

load 'testSet.mat'
originalTestSet = testSet{2};
originalTestSet = originalTestSet';
save('originalTestSet.mat','originalTestSet');