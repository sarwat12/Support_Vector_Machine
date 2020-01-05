%Name = Sarwat Shaheen %ID = 214677322
clear; clc;
%Loading file with input matrix
A = load('fg_inputs.txt');
%Matrix contaning all inputs with first feature = 1
X = ones(200,4); X(1:200, 2:4) = A(1:200, 1:3);
%Matrix created for labels corresponding to X
Y(1:100, 1) = 1; Y(101:200) = -1;
%Vector created for parameters w, yet to be estimated
W = zeros(1, 4);
%SGD implementation for this Soft SVM %With Paramters T = 500; initial lambda = 1; N = 200;
T = 500; lambda = 1;
for i = 1:T
  %random index between 1 and 200 for SGD and step size
  random_index = randi(200); Wprime(i, :) = (1/(lambda*i)) * W(1,:);
  %if misclassified
  if ((X(random_index)*W'.*Y(random_index)) < 1)
    %Calculating gradient with respect to each parameter
    J(1) = (lambda*norm(W)) +  (-1.*Y(random_index).*X(random_index,1)); J(2) = (lambda*norm(W)) +  (-1.*Y(random_index).*X(random_index,2));
    J(3) = (lambda*norm(W)) +  (-1.*Y(random_index).*X(random_index,3)); J(4) = (lambda*norm(W)) +  (-1.*Y(random_index).*X(random_index,4));
    %Updating parameters
    W(1) = W(1) - J(1); W(2) = W(2) - J(2); W(3) = W(3) - J(3); W(4) = W(4) - J(4);
  endif
endfor
%Averaged final weight vector and final hypothesis
fW = sum(Wprime)/T; hTrained = X * fW';
%keeping track of hinge loss, emphirical loss, and binary loss
slack = max(0, (1 - (hTrained.*Y))); L = (lambda * norm(W)^2) + ((1/200) * sum(slack)); B = (hTrained .* Y < 0);
%Plotting binary and hinge loss plots
figure(1); plot(slack, 'r-'); xlabel('No. of iterations'); ylabel('Hinge loss'); title('Q3b: Hinge Loss with lambda = 0.01');
figure(2); plot(B, '-s'); xlabel('No. of iterations'); ylabel('Binary Loss Function'); title('Q3b: Binary Loss with lambda = 0.01');
%Weight vector based on last iterate
disp(fW);