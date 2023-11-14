reset(gpuDevice);
gpuDevice(1);

% VERİ SETİNİ OLUŞTURMA

trafficSignDataFolder = fullfile("C:\Users\kemal\Desktop\Tez_Trafik İşaretleri Tanıma\gtsrb-preprocessed\Augment Dataset\train_balanced_augment");

trafficSignDataFolderTest = fullfile("C:\Users\kemal\Desktop\Tez_Trafik İşaretleri Tanıma\gtsrb-preprocessed\Augment Dataset\train_validation");

imds = imageDatastore(trafficSignDataFolder, ...
    'LabelSource','foldernames','IncludeSubfolders',true);

imdsTest = imageDatastore(trafficSignDataFolderTest, ...
    'LabelSource','foldernames','IncludeSubfolders',true);

%-------------------------------------------------------------------------%

%VERİ SETİ İLE İLGİLİ GENEL BİLGİLER

% Veri setinde kaç adet etiket bulunduğu yazdırılır.
tableLabels=countEachLabel(imds);

%Veri setinde bulunan en düşük veri
minLabels=min(tableLabels{:,2});

%Veri setini %80 oranında train-validate olarak böl
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,0.2,'randomized');

%Veri setindeki sınıf sayısı tanımlanır.
numClasses = numel(categories(imds.Labels));

%Eğitim Veri Sayısı
numTrainImages = numel(imdsTrain.Labels);

%Doğrulama Veri Sayısı
numValidationImages = numel(imdsValidation.Labels);

%Test Veri Sayısı
numTestImages = numel(imdsTest.Labels);

%Toplam Veri Sayısı
numTotalImages = numTrainImages + numValidationImages+ numTestImages;
numImages=imds.countEachLabel{:,2};
numImages=sum(numImages,"all");

classNames = categories(imds.Labels);
numClasses = numel(categories(imds.Labels));

%-------------------------------------------------------------------------%

%ViT OLUŞTURMA


net = visionTransformer("tiny-16-imagenet-384");
inputSize = net.Layers(1).InputSize; % getting image size info

net = freezeNetwork(net,LayersToIgnore="SelfAttentionLayer");

newFClayer = fullyConnectedLayer(numClasses,'Name','newHead','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
net = replaceLayer(net,"head",newFClayer);

%-------------------------------------------------------------------------%

%DATA AUGMENTATION

augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain,"ColorPreprocessing","gray2rgb");
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation,"ColorPreprocessing","gray2rgb");
augimdsTest = augmentedImageDatastore(inputSize,imdsTest,"ColorPreprocessing","gray2rgb");

%-------------------------------------------------------------------------%

%EĞİTİM ÖZELLİKLERİNİN AYARLANMASI

miniBatchSize = 8;

numObservationsTrain = numel(augimdsTrain.Files);
numIterationsPerEpoch = floor(numObservationsTrain/miniBatchSize);

options = trainingOptions("adam", ...
    MaxEpochs=2, ...
    InitialLearnRate=0.0001, ...
    MiniBatchSize=miniBatchSize, ...
    ValidationData=augimdsValidation, ...
    ValidationFrequency=numIterationsPerEpoch, ...
    OutputNetwork="best-validation-loss", ...
    Plots="training-progress", ...
    Metrics="accuracy", ...
    Verbose=true);

netTransfer = trainnet(augimdsTrain,net,"crossentropy",options);

%-------------------------------------------------------------------------%

%ACCURACY HESAPLAMA

mbq = minibatchqueue(augimdsTest,1, ...
    MiniBatchFormat="SSCB");

YTest = [];

% Loop over mini-batches.
while hasdata(mbq)
    
    % Read mini-batch of data.
    X = next(mbq);
       
    % Make predictions using the predict function.
    Y = predict(netTransfer,X);
   
    % Convert scores to classes.
    predBatch = onehotdecode(Y,classNames,1);
    YTest = [YTest; predBatch'];
end

figure
TTest = imdsTest.Labels;
cmTest = confusionchart(TTest,YTest,"ColumnSummary","absolute",...
    "RowSummary","absolute","Normalization","absolute");

accuracy = mean(YTest == TTest);

save('visionTransformer_bgDataset.mat');