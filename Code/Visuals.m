%% Load data
clear, clc;
load('AnimalClassify.mat');

%% Predict batch 1
pred = predict(net, validation(:,1));
pred = pred == max(pred,[],2);

%% Label batch 1
actLabels = imdsValidation.Labels(:);
actLabels = grp2idx(actLabels);
predLabels = zeros(128, 1);
for i = 1:128
	predLabels(i) = pred(i, actLabels(i));
end

%% Plot validation images
for i = 1:64
	subplot(8,8,i);
	imshow(validation{i,1}{:,:,:});
	title(['Correct: ' num2str(predLabels(i))]);
end



