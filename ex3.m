%% Unsupervised Learning
clear;
clc;

%%
load shuttle.dat
tabulate(shuttle(:,10)) % last variable is the response

%%
load california.dat

%% 3.5 (from fslssvm_script.m)

% classification
X = shuttle(:,1:end-1);
Y = shuttle(:,end);
testX = [];
testY = [];

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 1; % 6
function_type = 'c'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);


%% Function estimation

X = california(:,1:end-1);
Y = california(:,end);
testX = [];
testY = [];

%Parameter for input space selection
%Please type >> help fsoperations; to get more information  

k = 1; % 6
function_type = 'f'; %'c' - classification, 'f' - regression  
kernel_type = 'RBF_kernel'; % or 'lin_kernel', 'poly_kernel'
global_opt = 'csa'; % 'csa' or 'ds'

%Process to be performed
user_process={'FS-LSSVM', 'SV_L0_norm'};
window = [15,20,25];

[e,s,t] = fslssvm(X,Y,k,function_type,kernel_type,global_opt,user_process,window,testX,testY);
