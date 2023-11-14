% VERİ SETİNİ OLUŞTURMA

% YSA' ya sokulacak datanın dosya dizini girilir.
trafficSignDataFolder=fullfile(...
    "C:\Users\kemal\Desktop\Tez_Trafik İşaretleri Tanıma_Kod\gtsrb-preprocessed\train_balanced\");

% İhtiyaç varsa GPU'nun YSA'yı çalıştır çalıştırmayacağı çıktısını verir.
% Sonuç 1 ise problem yok.

% % try
% %     nnet.internal.cnngpu.reluForward(1);
% % catch ME
% % end

% matlab image dataset oluşturulur. 1. argüman => Dosya Dizini, 
% 2. argüman => Veri setinin etiket kaynağı yani dosya isimleri
% 3. argüman => Alt dosyalar dahil edilsin
imds = imageDatastore(trafficSignDataFolder, ...
                      'LabelSource','foldernames',...
                      'IncludeSubfolders',true);

%-------------------------------------------------------------------------%

%VERİ SETİ İLE İLGİLİ GENEL BİLGİLER

% Veri setinde kaç adet etiket bulunduğu yazdırılır.
tableLabels=countEachLabel(imds);

%Veri setinde bulunan en düşük veri
minLabels=min(tableLabels{:,2});
% imds=splitEachLabel(imds,1410,'randomized');

%Veri setini %80 oranında train-validate olarak böl
[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');

%Veri setindeki sınıf sayısı tanımlanır.
numClasses = numel(categories(imdsTrain.Labels));

%Eğitim Veri Sayısı
numTrainImages = numel(imdsTrain.Labels);

%Doğrulama Veri Sayısı
numValidationImages = numel(imdsValidation.Labels);

%Toplam Veri Sayısı
numTotalImages = numTrainImages + numValidationImages;
numImages=imdsTrain.countEachLabel{:,2};
numImages=sum(numImages,"all");
% idx = randperm(numTrainImages,16);
% I = imtile(imds, 'Frames', idx);
% figure
% imshow(I)

%-------------------------------------------------------------------------%

%YAPAY SİNİR AĞI OLUŞTURMA

%YSA Mimarisi seçilir
net = googlenet;

%YSA Giriş boyutu ağa göre belirlenir. 
inputSize = net.Layers(1).InputSize;

%Pre-trained YSA lgraph değişkenine atanır.
lgraph = layerGraph(net); 

%-------------------------------------------------------------------------%

%TRANSFER LEARNING OLUŞTURMA
%Transfer Learning yaparken yani pre-trained bir mimari ile tekrar bir
% eğitim yapmak için öğrenebilir katman ve sınıflandırma katmanları
% değiştirilir. Opsiyonel olarak ilk katmanlarındaki ağırlıklar dondurulur.


%findLayersTo Replace fonskiyonu kullanılarak, learnable ve class layer
%belirlenir.
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

% Ağın learnable layer'ı Fully Connected veya Conv2d Layer olabilir.
% Googlenet'te fullyconnectedlayer'dır ve bu ağı newLearnableLayer ile
% değiştirir. Eğitim hızlanması için LearnRate factorler 20 yapılır.
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',20, ...
        'BiasLearnRateFactor',20);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',20, ...
        'BiasLearnRateFactor',20);
end


%replaceLayer fonksiyonu ile newLearnableLayer eskisi yerine ağa verilir.
lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);

%Ağın sınıflandırma katmanı yeniden oluşturulur.
newClassLayer = classificationLayer('Name','new_classoutput');

%replaceLayer fonksiyonu ile newClassLayer eskisi yerine ağa verilir.
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%-------------------------------------------------------------------------%

% BAŞLANGIÇ KATMANLARINI DONDURMA

%Yeni eklenen katmanlar ile birlikte katmanlar layers değişkenine atanır
layers = lgraph.Layers;

%Bağlantılar ise connections değişkenine atanır
connections = lgraph.Connections;

%freezeWeights fonk ile ilk 10 katmandaki ağırlıklar sıfırlanır.
layers(1:10) = freezeWeights(layers(1:10));

%createLgraphUsingConnections fonk ile katmanlar arası bağlantılar
%tekrardan yapılır.
lgraph = createLgraphUsingConnections(layers,connections);
 
%-------------------------------------------------------------------------%

%EĞİTİM ÖZELLİKLERİNİN AYARLANMASI
%Eğitim için görsellerin mimarininin giriş katmanına uygun boyutlarda
%oluşturulması gerekmektedir. Bu sebeple;

%Eğitim ve doğrulama verileri ağın giriş boyutuna ayarlanır.
augimdsTrain = augmentedImageDatastore(inputSize,imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize,imdsValidation);

%mini yığın boyutu 
miniBatchSize = 32;

%doğrulama sıklıığı -> overfitting kontrolü yapılır
valFrequency = 125;
% valFrequency = floor(numel(augimdsTrain.Files)/miniBatchSize);

%Eğitim ayarları trainingOptions fonksiyonu ile yapılır.
options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',5, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',valFrequency, ...
    'Verbose',true, ...
    'Plots','training-progress',...
    'OutputNetwork','best-validation-loss',...
    'ResetInputNormalization',true,...
    'L2Regularization',0.0005,...
    'VerboseFrequency',valFrequency,...
    'ExecutionEnvironment','gpu');

%-------------------------------------------------------------------------%

%EĞİTİMİN YAPILMASI

% Ağın eğitimi trainNetwork fonk ile yapılır. 1. Argüman => Eğitim verisi,
% 2. Argüman => Network 3. Argüman => Eğitin özellikleri
netTransfer = trainNetwork(augimdsTrain,lgraph,options);

%-------------------------------------------------------------------------%

%SINIFLANDIRMA

% Sınıflandırma classfy fonk ile yapılır
[YPred,scores] = classify(netTransfer,augimdsValidation);

%Sonuçların Görselleştirilmesi - opsiyonel
% idx = randperm(numel(augimdsValidation.Files),4);
% figure
% for i = 1:4
%     subplot(2,2,i)
%     I = readimage(imdsValidation,idx(i));
%     imshow(I)
%     label = YPred(idx(i));
%     title(string(label));
% end

% Doğruluk Oranını bulma
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation);

%Karışıklık Matrixi oluşturma
figure
cm=confusionchart(YValidation,YPred,"ColumnSummary","absolute",...
    "RowSummary","absolute","Normalization","absolute");

%Modeli Kaydet
save('GN_VK_D3.mat');