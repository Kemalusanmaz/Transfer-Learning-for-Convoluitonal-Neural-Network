
trafficSignDataFolder=fullfile("C:\Users\kemal\Desktop\Tez_Trafik İşaretleri Tanıma_Kod\gtsrb-preprocessed\train_increased\");


imds = imageDatastore(trafficSignDataFolder, ...
    'LabelSource','foldernames','IncludeSubfolders',true);

tableLabels=countEachLabel(imds);
minLabels=min(tableLabels{:,2});
% imds=splitEachLabel(imds,1410,'randomized');

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');

numTrainImages = numel(imdsTrain.Labels);
numValidationImages = numel(imdsValidation.Labels);
numTotalImages = numTrainImages + numValidationImages;
numImages=imdsTrain.countEachLabel{:,2};
numImages=sum(numImages,"all");
% idx = randperm(numTrainImages,16);

% I = imtile(imds, 'Frames', idx);
% 
% figure
% imshow(I)


net = resnet50;

inputSize = net.Layers(1).InputSize;

lgraph = layerGraph(net); 

[learnableLayer,classLayer] = findLayersToReplace(lgraph);


numClasses = numel(categories(imdsTrain.Labels));

newConvLayer =  convolution2dLayer([1, 1],numClasses,...
    'WeightLearnRateFactor',20,'BiasLearnRateFactor',20,...
    "Name",'new_conv');

lgraph = replaceLayer(lgraph,'fc1000',newConvLayer);

newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'ClassificationLayer_fc1000',newClassificatonLayer);

% imageAugmenter = imageDataAugmenter("RandRotation",[-30 30],"RandXReflection",...
%     true,"RandYTranslation",[-2 2]);
% 
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);

options = trainingOptions(...
   'sgdm', ...
   'MiniBatchSize',32, ...
   'MaxEpochs',5, ...
   'InitialLearnRate',1e-3, ...
   'Shuffle','every-epoch', ...
   'ValidationData',augimdsValidation, ...
   'ValidationFrequency',50, ...
   'Verbose',true, ...
   'Plots','training-progress', ...
   'OutputNetwork','best-validation-loss',...
   'ResetInputNormalization',true,...
   'L2Regularization',0.0005,...
   'VerboseFrequency',50);

netTransfer = trainNetwork(augimdsTrain,lgraph,options);

idx = randperm(numel(augimdsValidation.Files),4);
[YPred,scores] = classify(netTransfer,augimdsValidation);

% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdsValidation,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label));
% end

YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

figure
cm=confusionchart(YValidation,YPred,"ColumnSummary","absolute",...
    "RowSummary","absolute","Normalization","absolute");

save('RN_VK_D3.mat');