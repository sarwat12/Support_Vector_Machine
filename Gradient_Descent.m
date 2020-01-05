%Sarwat Shaheen #214677322
clear; clc;
%Load Data
Data = load('seeds_dataset.txt');
%Input data matrix without labels
X = ones(210, 8);
X(1:210,2:8) = Data(1:210,1:7);
%Labels for training W1
Y1(1:70,:) = 1; Y1(71:210,:) = -1;
%Labels for training W1
Y2(1:70,:) = -1; Y2(71:140,:) = 1; Y2(141:210,:) = -1;
%Labels for training W1
Y3(1:140,:) = -1; Y3(141:210,:) = 1;
%Parameter vector for W1
W1 = ones(1,8);
%Parameter vector for W2
W2 = zeros(1,8);
%Parameter vector for W3
W3 = zeros(1,8);

H1 = X * W1'; H2 = X * W2';  H3 = X * W3';

T = 500; lambda = 1;
for i = 1:T
  %random index between 1 and 200 for SGD and step size
  random_index = randi(200); W1prime(i, :) = (1/(lambda*i)) * W1(1,:);
  %if misclassified
  if ((X(random_index)*W1'.*Y1(random_index)) < 1)
    %Calculating gradient with respect to each parameter
    J(1) = (-1.*Y1(random_index).*X(random_index,1)); J(2) = (-1.*Y1(random_index).*X(random_index,2));
    J(3) = (-1.*Y1(random_index).*X(random_index,3)); J(4) = (-1.*Y1(random_index).*X(random_index,4));
    J(5) = (-1.*Y1(random_index).*X(random_index,5)); J(6) = (-1.*Y1(random_index).*X(random_index,6));
    J(7) = (-1.*Y1(random_index).*X(random_index,7)); J(8) = (-1.*Y1(random_index).*X(random_index,8));
    %Updating parameters
    W1(1) = W1(1) - J(1); W1(2) = W1(2) - J(2); W1(3) = W1(3) - J(3); W1(4) = W1(4) - J(4);
    W1(5) = W1(5) - J(5); W1(6) = W1(6) - J(6); W1(7) = W1(7) - J(7); W1(8) = W1(8) - J(8); 
  endif
  B(i) = dot(X(1,:), W1.*Y1(1)) < 1;
endfor
%Averaged final weight vector and final hypothesis
fW1 = sum(W1prime)/T; h1Trained = X * fW1';
figure(1); plot(B, '-s'), xlabel('Iterations'); ylabel('Binary Loss');
title('A2Q3e-W3 Binary Loss');