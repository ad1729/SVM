%% adding the path with the SVM and LSSVM toolboxes
addpath('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/svm/');
addpath('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/LSSVMlab/');
addpath('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/fixed-size/');
clear;
clc;
%% Classification
%% (1.1) Classifying two Gaussians
X1 = 1 + randn(50,2);
X2 = -1 + randn(51,2);
Y1 = ones(50,1);
Y2 = -ones(51,1);
X  = [X1;X2];
Y = [Y1;Y2];

%% Plotting
figure;
hold on;
plot(X1(:,1), X1(:,2), 'ro', 'LineWidth', 3);
plot(X2(:,1), X2(:,2), 'bo', 'LineWidth', 3);
hold off;
title('Two class Gaussian RV');
xlabel('x');
ylabel('y');
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/plot1-1', '-dpng');

%% (1.2) Support Vector Machine
% http://cs.stanford.edu/people/karpathy/svmjs/demo/
%% (1.3) LS-SVMlab
%% Demo
democlass;
%%
help prelssvm
%%
clear;
clc;

%% Iris data
load('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/Session 1/iris.mat')
%% Fit the model
gam = 1.0;
type = 'lin_kernel';

[alpha, b] = trainlssvm({X, Y, 'c', gam, [], type});
plotlssvm({X, Y, 'c', gam, [], type, 'preprocess'}, {alpha, b});
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/iris-linear', '-dpng');

%% performance on test set
[Yht, Zt] = simlssvm({X,Y,type,gam,[],'lin_kernel'}, {alpha,b}, Xt);
err = sum(Yht ~= Yt); 
fprintf('\n on test: #misclass = %d, error rate = %.2f%%\n', err, err/length(Yt)*100)

%% 1.3.1 (tuning hyperparameters) LS-SVM
% set the parameters to some value
gam = 0.1;
sig2 = 20;

% generate random indices
idx = randperm(size(X,1));

% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

% train the model
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2, 'RBF_kernel'});

% evaluate it on Xval:
estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);

%% model evaluation (fixed validation set)
% generate random indices
idx = randperm(size(X,1));

% create the training and validation sets
% using the randomized indices
Xtrain = X(idx(1:80),:);
Ytrain = Y(idx(1:80));
Xval = X(idx(81:100),:);
Yval = Y(idx(81:100));

gamlist=[1, 5, 10, 25, 50, 100];
sig2list=[0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 100];

for gam=gamlist,
    errlist=[];
    
    for sig2=sig2list,
        % train the model
        [alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2, 'RBF_kernel'});
        % evaluate it on Xval:
        estYval = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'},{alpha,b},Xval);
        % calculate the error
        err = sum(estYval ~= Yval); errlist = [errlist; err];
        fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, err/length(Yval)*100)         
    end

    % make a plot of the % misclassification wrt. sig2 for each gam val
    figure;
    plot(log(sig2list), ((errlist)/(length(Yval))*100), '*-', 'LineWidth', 3, 'MarkerSize', 12), 
    ylim([0 max(((errlist)/(length(Yval))*100))+10]),
    xlabel(sprintf('log(sig2)-with-gam = %s', num2str(gam))), ylabel('% misclassified'),
    file_string = sprintf('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/iris-rbf-fixed-val-gam-%s', num2str(gam));
    print(file_string, '-dpng'); 
end

%%
clc;

%% model evaluation (10-fold CV)
gamlist=[1, 5, 10, 25, 50, 100];
sig2list=[0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 100];

for gam=gamlist,
    errlist=[];
    
    for sig2=sig2list,
        % calculate the prediction error
        err = crossvalidate({Xtrain, Ytrain, 'c', gam, sig2, 'RBF_kernel'}, 10, 'misclass', 'mean'); 
        errlist = [errlist; err];
        %fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, (err/10)*100)         
    end

    % make a plot of the % misclassification wrt. sig2 for each gam val
    figure;
    plot(log(sig2list), errlist, '*-', 'LineWidth', 3, 'MarkerSize', 12),
    xlabel(sprintf('log(sig2)-with-gam = %s', num2str(gam))), ylabel('avg. num. misclassified (in 8 folds)'),
    file_string = sprintf('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/iris-rbf-cv-gam-%s', num2str(gam));
    print(file_string, '-dpng');
end

%% model evaluation (LOO-CV)
gamlist=[1, 5, 10, 25, 50, 100];
sig2list=[0.001, 0.01, 0.1, 1, 5, 10, 25, 50, 100];

for gam=gamlist,
    errlist=[];
    
    for sig2=sig2list,
        % calculate the prediction error
        err = leaveoneout({Xtrain, Ytrain, 'c', gam, sig2, 'RBF_kernel'}, 'misclass', 'mean'); 
        errlist = [errlist; err];
        %fprintf('\n on test: #misclass = %d, error rate = %.2f%% \n', err, (err/10)*100)         
    end

    % make a plot of the % misclassification wrt. sig2 for each gam val
    figure;
    plot(log(sig2list), errlist, '*-', 'LineWidth', 3, 'MarkerSize', 12),
    xlabel(sprintf('log(sig2)-with-gam = %s', num2str(gam))), ylabel('avg. num. misclassified (in 8 folds)'),
    file_string = sprintf('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/iris-rbf-loocv-gam-%s', num2str(gam));
    print(file_string, '-dpng');
end

%% Using tunelssvm
clc;
model = {X,Y,'c',[],[],'RBF_kernel','csa'}; % csa vs ds
[gam,sig2,cost] = tunelssvm(model,'gridsearch', 'crossvalidatelssvm', {10,'misclass'}); 
gam
sig2
cost
% simplex vs gridsearch

%% ROC curve
gam = 0.1;
sig2 = 2;
[alpha,b] = trainlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'});
[Ysim,Ylatent] = simlssvm({Xtrain,Ytrain,'c',gam,sig2,'RBF_kernel'}, {alpha,b},Xval);
roc(Ylatent,Yval);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ripley dataset
clear;
clc;
load('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/Session 1/ripley.mat')

%% plot the data
figure;
hold on;
% plot the train set
gscatter(X(:,1), X(:,2), Y, 'br', 'xo');
% plot the test set
%gscatter(Xt(:,1), Xt(:,2), Yt, 'br', 'xo');
hold off;
title('Ripley Data: Training Set');
%title('Ripley Data: Test Set');
xlabel('x1');
ylabel('x2');
% plot the train set
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/ripley-train', '-dpng');
% plot the test set
%print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/ripley-test', '-dpng');

%% Model
kernel = 'RBF_kernel'; % 'RBF_kernel' or 'lin_kernel'
model = {X,Y,'c',[],[],kernel,'csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({X,Y,'c',gam,sig2, kernel});
plotlssvm({X, Y, 'c', gam, sig2, kernel, 'preprocess'}, {alpha, b});
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/ripley-class-rbf', '-dpng');
% evaluate it on Xval:
[Ysim, Ylatent] = simlssvm({X,Y,'c',gam,sig2,kernel},{alpha,b},Xt);
roc(Ylatent,Yt);
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/ripley-roc-rbf', '-dpng');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%--------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Breast Cancer dataset
clear;
clc;
load('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/Session 1/breast.mat')
X = trainset;
Y = labels_train;
Xt = testset;
Yt = labels_test;

%% plot the training data
figure;
hold on;
% plot the train set
gscatter(X(:,1), X(:,2), Y, 'br', 'xo');
hold off;
title('Breast Cancer Data: Training Set');
xlabel('x1');
ylabel('x2');
% plot the train set
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/breast-train', '-dpng');

%% plot the test data
figure;
hold on;
%plot the test set
gscatter(Xt(:,1), Xt(:,2), Yt, 'br', 'xo');
hold off;
title('Breast Cancer Data: Test Set');
xlabel('x1');
ylabel('x2');
% plot the test set
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/breast-test', '-dpng');

%% Model
kernel = 'lin_kernel'; % 'RBF_kernel' or 'lin_kernel'
model = {X,Y,'c',[],[],kernel,'csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({X,Y,'c',gam,sig2, kernel});
plotlssvm({X, Y, 'c', gam, sig2, kernel, 'preprocess'}, {alpha, b});
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/breast-class-linear', '-dpng');
% evaluate it on Xval:
[Ysim, Ylatent] = simlssvm({X,Y,'c',gam,sig2,kernel},{alpha,b},Xt);
roc(Ylatent,Yt);
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/breast-roc-linear', '-dpng');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%---------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Diabetes dataset
clear;
clc;
load('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/Session 1/diabetes.mat')
X = trainset;
Y = labels_train;
Xt = testset;
Yt = labels_test;

%% plot the training data
figure;
hold on;
% plot the train set
gscatter(X(:,1), X(:,2), Y, 'br', 'xo');
hold off;
title('Breast Cancer Data: Training Set');
xlabel('x1');
ylabel('x2');
% plot the train set
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/diabetes-train', '-dpng');

%% plot the test data
figure;
hold on;
%plot the test set
gscatter(Xt(:,1), Xt(:,2), Yt, 'br', 'xo');
hold off;
title('Breast Cancer Data: Test Set');
xlabel('x1');
ylabel('x2');
% plot the test set
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/diabetes-test', '-dpng');

%% Model
kernel = 'RBF_kernel'; % 'RBF_kernel' or 'lin_kernel'
model = {X,Y,'c',[],[],kernel,'csa'};
[gam,sig2,cost] = tunelssvm(model,'simplex', 'crossvalidatelssvm', {10,'misclass'});
[alpha,b] = trainlssvm({X,Y,'c',gam,sig2, kernel});
plotlssvm({X, Y, 'c', gam, sig2, kernel, 'preprocess'}, {alpha, b});
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/diabetes-class-rbf', '-dpng');
% evaluate it on Xval:
[Ysim, Ylatent] = simlssvm({X,Y,'c',gam,sig2,kernel},{alpha,b},Xt);
roc(Ylatent,Yt);
print('/home/ad/Desktop/KUL Course Material/SVM (support vector machines)/Exercise Session/images/diabetes-roc-rbf', '-dpng');