%%% Make sure this file is in the same file path as 4_AnimalCategories
%% Get alexnet as our transfer learning network
net = alexnet;

% Initialise layers
alexnetLayers = net.Layers;

% Store images into a datastore
imagePath = '4_AnimalCategories';
imageDs = imageDatastore(imagePath,'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% Split into training and validation randomly
numTraining = 80;
[imageTrain,imageValidateL] = splitEachLabel(imageDs,numTraining,'randomize');

% Change all images to size 227x227 and convert to RGB so that it can be
% used with the AlexNet layers
imageTrain = augmentedImageDatastore([227, 227, 3], imageTrain,...
    'ColorPreprocessing', 'gray2rgb');
imageValidate = augmentedImageDatastore([227, 227, 3], imageValidateL,...
    'ColorPreprocessing', 'gray2rgb');

% AlexNet's layers are for 1000 classes, configure the layers to fit our
% dataset by removing last 3 layers
newLayer = alexnetLayers(1:end-3);

% Add more layers into the end of the array
numCategories = 4;
newLayer(end+1) = fullyConnectedLayer(numCategories,'WeightLearnRateFactor',10,...
    'WeightL2Factor', 1, 'BiasLearnRateFactor', 20, 'BiasL2Factor', 0);
newLayer(end+1) = softmaxLayer;
newLayer(end+1) = classificationLayer();

% Set training options
trainingOpts = trainingOptions('sgdm', ...
    'InitialLearnRate',0.0001, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imageValidate, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots','training-progress');

% Train the network
CNN = trainNetwork(imageTrain, newLayer, trainingOpts);

%% Testing the classifier visually
[predictY, validateScore] = classify(CNN, imageValidate);

idx = randperm(numel(imageValidate.Files), 9);
figure
for j = 1:9
    subplot(3,3,j)
    image = readimage(imageValidateL, idx(j));
    imshow(image)
    classifiedLabel = predictY(idx(j));
    title(string(classifiedLabel));
end
%% Test accuracy of classifier
validateY = imageValidateL.Labels;
accuracy = sum(predictY == validateY)/numel(validateY)