clear; close all; clc;

data = csvread('students.csv', 1, 0);
X = data(:, 1:2);
y = data(:, 3);

m = rows(X);
rng(7);
perm = randperm(m);
train_idx = perm(1:round(0.8*m));
test_idx  = perm(round(0.8*m)+1:end);

X_train = X(train_idx, :);
y_train = y(train_idx, :);
X_test  = X(test_idx,  :);
y_test  = y(test_idx,  :);

mu = mean(X_train);
sigma = std(X_train);
sigma(sigma == 0) = 1;

X_train_n = (X_train - mu) ./ sigma;
X_test_n  = (X_test  - mu) ./ sigma;

X_train_n = [ones(rows(X_train_n),1) X_train_n];
X_test_n  = [ones(rows(X_test_n),1)  X_test_n];

theta = zeros(columns(X_train_n), 1);
alpha = 0.1;
num_iters = 1500;

for i = 1:num_iters
  h = sigmoid(X_train_n * theta);
  grad = (1/rows(X_train_n)) * (X_train_n' * (h - y_train));
  theta = theta - alpha * grad;
  if mod(i, 200) == 0
    J = cost_logistic(X_train_n, y_train, theta);
    printf("iter %4d | cost = %.4f\n", i, J);
  end
end

printf("\nTrained theta:\n"); disp(theta');

pred_test = sigmoid(X_test_n * theta) >= 0.5;
acc = mean(double(pred_test == y_test)) * 100;
printf("Test Accuracy: %.2f%% (%d/%d)\n", acc, sum(pred_test==y_test), length(y_test));

sample = [72 68];
sample_n = ([sample] - mu) ./ sigma;
sample_n = [1 sample_n];
p = sigmoid(sample_n * theta);
printf("Sample probability of admission = %.3f (label=%d)\n", p, p>=0.5);

figure(1);
hold on; grid on; box on;
pos = find(y==1); neg = find(y==0);
plot(X(pos,1), X(pos,2), 'kx', 'markersize', 8, 'linewidth', 2);
plot(X(neg,1), X(neg,2), 'bo', 'markersize', 6, 'linewidth', 1.5);
xlabel('Exam 1'); ylabel('Exam 2'); title('Admission data');

x1 = linspace(min(X(:,1))-5, max(X(:,1))+5, 200);
x2 = linspace(min(X(:,2))-5, max(X(:,2))+5, 200);
[X1g, X2g] = meshgrid(x1, x2);
Xg = [X1g(:) X2g(:)];
Xg_n = (Xg - mu) ./ sigma;
Hg = sigmoid([ones(rows(Xg_n),1) Xg_n] * theta);
Hg = reshape(Hg, size(X1g));
contour(X1g, X2g, Hg, [0.5 0.5], 'r', 'linewidth', 2);
legend('Admitted (1)', 'Not admitted (0)', 'Decision boundary');
hold off;

function g = sigmoid(z)
  g = 1 ./ (1 + exp(-z));
end

function J = cost_logistic(X, y, theta)
  m = rows(X);
  h = 1 ./ (1 + exp(-(X*theta)));
  h = min(max(h, 1e-12), 1-1e-12);
  J = ( -1/m ) * ( y' * log(h) + (1 - y') * log(1 - h) );
end
