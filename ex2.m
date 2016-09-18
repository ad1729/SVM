%% adding the path with the SVM and LSSVM toolboxes
addpath('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/svm/');
addpath('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/LSSVMlab/');
%addpath('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/fixed-size/');
clear;
clc;
%% Function estimation and time series prediction
%% SVM for regression
uiregress

%% Sum of cosines

%% Demo
% code taken from demofun
%demofun
X = (-3:0.2:3)';
Y = sinc(X)+0.1.*randn(length(X),1);
gam = 10;
sig2 = 0.3;
type = 'function estimation';
[alpha,b] = trainlssvm({X,Y,type,gam,sig2,'RBF_kernel'});

% evaluate
Xt = 3.*randn(10,1);
Yt = simlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b},Xt);

plotlssvm({X,Y,type,gam,sig2,'RBF_kernel','preprocess'},{alpha,b});
hold on; plot(min(X):.1:max(X),sinc(min(X):.1:max(X)),'g-.', 'LineWidth', 3);

%% Synthetic Example
clear; clc;
X = (-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);
% training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

gam = 10; % 10, 100
sig2 = 0.1; % 0.1, 1
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

plotlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'},{alpha,b});

YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);

plot(Xtest,Ytest,'.', 'MarkerSize', 15);
hold on;
plot(Xtest,YtestEst,'r-+', 'LineWidth', 2);
legend('Ytest','YtestEst');
title('gam = 10, sig2 = 0.1');
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/cos-reg-2', '-dpng');

%% Tuning the hyperparameters
seq = (-4:0.1:5);
N = length(seq);

cost_crossval = zeros(N);
cost_loo = zeros(N);

for i = 1:N,
    gam = 10^seq(i);
    for j = 1:N,
        sig2 = 10^seq(j);
        cost_crossval(i,j) = crossvalidate({Xtrain,Ytrain,'f', gam, sig2},10);
        cost_loo(i,j) = leaveoneout({Xtrain,Ytrain,'f', gam, sig2});
    end
end

%% plot the results for 10 fold cv
figure;
contour(seq, seq, cost_crossval, 'LineWidth', 2)
xlabel('log(gam)');
ylabel('log(sig2)');
zlabel('10-fold CV Error Rate');
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/tune-grid-err-cv', '-dpng');

%% evaluating the performance of the model on the test set for 10 fold CV
[gam_idx,sig2_idx] = find(cost_crossval == min(cost_crossval(:)));
gam = 10.^seq(gam_idx);
sig2 = 10.^seq(sig2_idx);

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);

plot(Xtest,Ytest,'.', 'MarkerSize', 15);
hold on;
plot(Xtest,YtestEst,'r-+', 'LineWidth', 2);
legend('Ytest','YtestEst');
title(sprintf('10 fold CV: gam = %s, sig2 = %s', num2str(gam), num2str(sig2)));
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/tune-grid-err-cv-test', '-dpng');

%% plot the results for loocv
figure;
contour(seq, seq, cost_loo, 'LineWidth', 2)
xlabel('log(gam)');
ylabel('log(sig2)');
zlabel('LOOCV Error Rate');
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/tune-grid-err-loo', '-dpng');

%% evaluating the performance of the model on the test set for loocv
[gam_idx,sig2_idx] = find(cost_loo == min(cost_loo(:)));
gam = 10.^seq(gam_idx);
sig2 = 10.^seq(sig2_idx);

[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'});

YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);

plot(Xtest,Ytest,'.', 'MarkerSize', 15);
hold on;
plot(Xtest,YtestEst,'r-+', 'LineWidth', 2);
legend('Ytest','YtestEst');
title(sprintf('LOOCV: gam = %s, sig2 = %s', num2str(gam), num2str(sig2)));
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/tune-grid-err-loocv-test', '-dpng');

%% Using tunelssvm
optFun = 'gridsearch'; % gridsearch and simplex
globalOptFun = 'ds'; % csa and ds

tic
[gam,sig2,cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', ... 
    globalOptFun},optFun,'crossvalidatelssvm',{10,'mse'})
toc

gam
sig2
cost

%% Evaluate
[alpha,b] = trainlssvm({Xtrain,Ytrain,'f',gam,sig2});
%plotlssvm({Xtrain,Ytrain,'f',gam,sig2},{alpha,b});

YtestEst = simlssvm({Xtrain,Ytrain,'f',gam,sig2,'RBF_kernel'}, {alpha,b},Xtest);

plot(Xtest,Ytest,'.', 'MarkerSize', 15);
hold on;
plot(Xtest,YtestEst,'r-+', 'LineWidth', 2);
legend('Ytest','YtestEst');
%title(sprintf('LOOCV: gam = %s, sig2 = %s', num2str(gam), num2str(sig2)));

%% Bayes
clear; clc;
X = (-10:0.1:10)';
Y = cos(X) + cos(2*X) + 0.1.*randn(length(X),1);
% training/validation and test sets are created:
Xtrain = X(1:2:length(X));
Ytrain = Y(1:2:length(Y));
Xtest = X(2:2:length(X));
Ytest = Y(2:2:length(Y));

%%
%sig2 = 0.5; gam = 10;
sig2 = 0.001; gam = 100;
criterion_L1 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},1)
criterion_L2 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},2)
criterion_L3 = bay_lssvm({Xtrain,Ytrain,'f',gam,sig2},3)
[~,alpha,b] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},1);
[~,gam] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},2);
[~,sig2] = bay_optimize({Xtrain,Ytrain,'f',gam,sig2},3);

% compute error bars
sig2e = bay_errorbar({Xtrain,Ytrain,'f',gam,sig2},'figure');

Yest = simlssvm({Xtrain, Ytrain, 'f', gam, sig2, 'RBF_kernel'}, {alpha, b}, Xtest);
mse_test = mse(Yest-Ytest);
mse_test

%% Bayes to iris
clear;
load iris;
gam = 5; sig2 = 0.75;
%[gam, sig2] = tunelssvm({X,Y,'c',[],[],'RBF_kernel', 'csa'}, 'simplex', 'crossvalidatelssvm', {10, 'misclass'});
[~,alpha,b] = bay_optimize({X,Y,'c',gam,sig2},1);
[~,gam] = bay_optimize({X,Y,'c',gam,sig2},2);
[~,sig2] = bay_optimize({X,Y,'c',gam,sig2},3);
bay_modoutClass({X,Y,'c',gam,sig2},'figure');

Yest = simlssvm({X,Y,'c',gam,sig2}, {alpha, b}, Xt);
err = sum(Yest ~= Yt)/length(Yt)
roc(Yt, Yest)

%% Bayes regression
X = 10.*rand(100,3)-3;
Y = cos(X(:,1)) + cos(2*(X(:,1))) + 0.3.*randn(100,1);
[gam, sig2, cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel', 'csa'}, 'simplex', ...
    'crossvalidatelssvm', {10, 'mse'});
[selected, ranking] = bay_lssvmARD({X,Y,'f',gam,sig2});

%% Robust regression
clear;clc;
X = (-10:0.2:10)';
Y = cos(X) + cos(2*X) + 0.1.*rand(size(X));

% adding outliers
out = [15 17 19];
Y(out) = 0.7+0.3*rand(size(out));
out = [41 44 46];
Y(out) = 1.5+0.2*rand(size(out));

gam = 100; sig2 = 0.1;

[alpha,b] = trainlssvm({X,Y,'f',gam,sig2});
%[gam, sig2, cost] = tunelssvm({X,Y,'f',[],[],'RBF_kernel'},'simplex','crossvalidatelssvm', {10,'mse'});
plotlssvm({X,Y,'f',gam,sig2},{alpha,b});
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/non-rob-fit', '-dpng');

%% model robust
model = initlssvm(X,Y,'f',[],[],'RBF_kernel');
costFun = 'rcrossvalidatelssvm';
wFun = 'wmyriad'; % whampel, wlogistic, wmyriad
model = tunelssvm(model,'simplex',costFun,{10,'mae'},wFun);
model.costCV
model = robustlssvm(model);
plotlssvm(model);
%print(sprintf('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/rob-%s', wFun), '-dpng');

%% Homework Problem
%% Time series prediction
clear; clc;
load('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/Session 2/santafe.mat')

figure
subplot(1,2,1)
plot(Z, '-');
title('Training Set');
subplot(1,2,2)
plot(Ztest, 'r-');
title('Test Set');
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/laser', '-dpng');

%%
lag = 50; % lag of the series; lag 21 had the minimum MAPE
X = windowize(Z,1:(lag+1));
Y = X(:,end);
X = X(:,1:lag);
horizon = length(Ztest)-lag;

[gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa','original'}, ...
    'simplex','crossvalidatelssvm', {10,'mae'});
% model = bay_optimize({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'},1);
% model = bay_optimize({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'},2);
% model = bay_optimize({X,Y,'f',gam,sig2,'RBF_kernel','csa','original'},3);
model = trainlssvm({X,Y,'f',gam,sig2, 'RBF_kernel', 'csa', 'original'});
Zpt = predict(model,Ztest(1:lag),horizon);

mape = mean(abs(Zpt-Ztest(lag+1:end))./abs(Ztest(lag+1:end)));
mae = mean(abs(Zpt-Ztest(lag+1:end)));
figure;
plot([Ztest(lag+1:end) Zpt]);
xlabel('Time');
legend('Test Data','Prediction');
title(sprintf('Lag = %s; MAPE = %s; MAE = %s', num2str(lag), num2str(mape), num2str(mae)));
mape
mae
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/laser-50', '-dpng');

%% Can see which lag had the lowest error
lag_all = 5:1:80;
mae_all = ones(1, length(lag_all));
mape_all = ones(1, length(lag_all));

for i = 1:length(lag_all),
    
    lag = lag_all(i); % lag of the series
    X = windowize(Z,1:(lag+1));
    Y = X(:,end);
    X = X(:,1:lag);
    horizon = length(Ztest)-lag;
    
    [gam,sig2] = tunelssvm({X,Y,'f',[],[],'RBF_kernel','csa','original'}, ...
    'simplex','crossvalidatelssvm', {10,'mae'});

    model = trainlssvm({X,Y,'f',gam,sig2, 'RBF_kernel', 'csa', 'original'});
    Zpt = predict(model,Ztest(1:lag),horizon);
    
    mape_all(i) = mean(abs(Zpt-Ztest(lag+1:end))./abs(Ztest(lag+1:end)));
    mae_all(i) = mean(abs(Zpt-Ztest(lag+1:end)));
    
    figure;
    plot([Ztest(lag+1:end) Zpt]);
    xlabel('Time');
    legend('Test Data','Prediction');
    title(sprintf('Lag = %s; MAPE = %s; MAE = %s', num2str(lag), num2str(mape_all(i)), num2str(mae_all(i))));
    filename = sprintf('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/Time Series/lag-%s', num2str(lag));
    print(filename, '-dpng');
    close;

end

%% which lag had the smallest error
[lag_idx] = find(mape_all == min(mape_all(:)));
mape_all(lag_idx)
plot(lag_all(14:end), mape_all(14:end), 'b+-');
xlabel('Lag');
ylabel('Mean Absolute Percentage Error');
title('MAPE for Santa Fe');
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/lag-vs-error', '-dpng');
