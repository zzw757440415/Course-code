close all;
clc;
%% 读取图像，并显示
img = imread('img2.jpg');
figure;
subplot(2,3,1);
imshow(img);
title('原始图像');
%% 定义kmeans参数
k = 3;
[m,n,p] = size(img);
A = reshape(img(:,:,1),m*n,1);
B = reshape(img(:,:,2),m*n,1);
C = reshape(img(:,:,3),m*n,1);
data = [A B C];
%% kmeans进行图像分割
res = kmeans(double(data),k);
result = reshape(res,m,n);
subplot(2,3,2);
imshow(label2rgb(result));
title(strcat('K=',num2str(k),'时RGB通道分割结果'));

res = kmeans(double(data),k+1);
result = reshape(res,m,n);
subplot(2,3,3);
imshow(label2rgb(result));
title(strcat('K=',num2str(k+1),'时RGB通道分割结果'));

res = kmeans(double(data),k+2);
result = reshape(res,m,n);
subplot(2,3,4);
imshow(label2rgb(result));
title(strcat('K=',num2str(k+2),'时RGB通道分割结果'));

res = kmeans(double(data),k+3);
result = reshape(res,m,n);
subplot(2,3,5);
imshow(label2rgb(result));
title(strcat('K=',num2str(k+3),'时RGB通道分割结果'));

res = kmeans(double(data),k+4);
result = reshape(res,m,n);
subplot(2,3,6);
imshow(label2rgb(result));
title(strcat('K=',num2str(k+4),'时RGB通道分割结果'));
%% 定义FCM参数
X = reshape(double(img),m*n,p);
k = 3;
b = 2;
%% FCM进行图像分割
figure;
subplot(2,3,1);
imshow(img);
title('原始图像');

[C,dist,~] = fcm(X,k,b);
[~, label] = min(dist,[],2);
subplot(2,3,2);
imshow(uint8(reshape(C(label, :), m, n, p)));
title(strcat('K=',num2str(k),'时RGB通道分割结果'));

[C,dist,~] = fcm(X,k+1,b);
[~, label] = min(dist,[],2);
subplot(2,3,3);
imshow(uint8(reshape(C(label, :), m, n, p)));
title(strcat('K=',num2str(k+1),'时RGB通道分割结果'));

[C,dist,~] = fcm(X,k+2,b);
[~, label] = min(dist,[],2);
subplot(2,3,4);
imshow(uint8(reshape(C(label, :), m, n, p)));
title(strcat('K=',num2str(k+2),'时RGB通道分割结果'));

[C,dist,~] = fcm(X,k+3,b);
[~, label] = min(dist,[],2);
subplot(2,3,5);
imshow(uint8(reshape(C(label, :), m, n, p)));
title(strcat('K=',num2str(k+3),'时RGB通道分割结果'));

[C,dist,~] = fcm(X,k+4,b);
[~, label] = min(dist,[],2);
subplot(2,3,6);
imshow(uint8(reshape(C(label, :), m, n, p)));
title(strcat('K=',num2str(k+4),'时RGB通道分割结果'));










