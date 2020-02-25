%% Load images
clear, clc;
path = '4_AnimalCategories';
imds = imageDatastore(path, 'IncludeSubfolders',true,'LabelSource','foldernames');

%% Split images into training/test
numTrainFiles = 80;
[imdsTrain, imdsValidation] = splitEachLabel(imds, numTrainFiles, 'randomize');
imdsTrainResized = augmentedImageDatastore([56, 56, 3], imdsTrain, 'ColorPreprocessing', 'gray2rgb');
imdsValidationResized = augmentedImageDatastore([56, 56, 3], imdsValidation, 'ColorPreprocessing', 'gray2rgb');

%% Store Validation data for later testing
validation = read(imdsValidationResized);

%% Defining Network Architecture
layers = [
    imageInputLayer([56 56 3])
	
	convolution2dLayer(3, 256)
    batchNormalizationLayer
    reluLayer
	
	convolution2dLayer(3, 256)
    batchNormalizationLayer
    reluLayer
	maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3, 256)
    batchNormalizationLayer
    reluLayer
	
	convolution2dLayer(3, 256)
    batchNormalizationLayer
    reluLayer
	maxPooling2dLayer(2,'Stride',2)
	
	convolution2dLayer(3, 512)
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3, 512)
    batchNormalizationLayer
    reluLayer
	
	convolution2dLayer(3, 512)
    batchNormalizationLayer
    reluLayer
	maxPooling2dLayer(2,'Stride',2)

	fullyConnectedLayer(4)
    softmaxLayer
    classificationLayer];

%% Training the network
options = trainingOptions('sgdm', ...	% SGD with momentum
    'InitialLearnRate',0.01, ...		% Initial learning rate
    'MaxEpochs', 10, ...				% Training cycles
    'Shuffle','every-epoch', ...		% Shuffle the data every training cycle
    'ValidationData',imdsValidationResized, ...% Specifying validation data
	'ExecutionEnvironment', 'gpu',...
	'ValidationFrequency', 5,...
    'Verbose',true, ...
    'Plots','training-progress');

net = trainNetwork(imdsTrainResized, layers, options);





