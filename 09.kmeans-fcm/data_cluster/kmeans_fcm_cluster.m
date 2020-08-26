clear,clc;
%% 产生第一组数据
mu1 = [0,0];  % 均值
s1 = [0.25 0;0 0.25];  % 协方差
data1 = mvnrnd(mu1,s1,100);  % 产生第一组数据
%% 产生第二组数据
mu2 = [-1.25,1.25];  % 均值
s2 = [0.25 0;0 0.25];  % 协方差
data2 = mvnrnd(mu2,s2,100);  % 产生第二组数据
%% 产生第三组数据
mu3 = [1.25,1.25];  % 均值
s3 = [0.25 0;0 0.25];  % 协方差
data3 = mvnrnd(mu3,s3,100);  % 产生第三组数据
%% 加载 Iris 数据,并归一化处理
load('iris.mat');
X = iris(1:150,1:4);
Y = iris(1:150,5);
[nr,nc] = size(X);
data_iris = zeros(nr,nc);
for i = 1:nc
    for j = 1:nr
        data_iris(j,i) = (X(j, i)-std(X(:, i)))/mean(X(:, i));
    end
end
%% 显示原始自定义数据
data = [data1;data2;data3];
[m,n] = size(data);
plot(data1(:,1),data1(:,2),'r*');
hold on;
plot(data2(:,1),data2(:,2),'g*');
plot(data3(:,1),data3(:,2),'b*');
title('原始数据分布');
xlabel('x data');
ylabel('y data');
%% 利用PCA对iris数据集降维，并可视化
[pc,score,latent,tsquare]=pca(data_iris);
res = cumsum(latent)./sum(latent);
disp(res);  % 显示每一维对原始数据的精度
data_iris = score(:,1:3);  % 利用PCA降到3维
figure;
scatter3(data_iris(:,1),data_iris(:,2),data_iris(:,3),40,Y,'*');
title('Iris数据原始分布');
xlabel('x data');
ylabel('y data');
zlabel('z data');
%% kmeans对原始自定义数据聚类,并可视化
k = input('输入聚类个数：');  % 聚类个数 
[pattern,center] = self_kmeans(k,data);
figure;
hold on;
for i=1:m
    if pattern(i,n+1) == 1
        plot(pattern(i,1),pattern(i,2),'r*');
        plot(center(1,1),center(1,2),'ko','MarkerFaceColor','k');
    elseif pattern(i,n+1) == 2
        plot(pattern(i,1),pattern(i,2),'g*');
        plot(center(2,1),center(2,2),'ko','MarkerFaceColor','k');
    elseif pattern(i,n+1) == 3
        plot(pattern(i,1),pattern(i,2),'b*');
        plot(center(3,1),center(3,2),'ko','MarkerFaceColor','k');
    elseif pattern(i,n+1) == 4
        plot(pattern(i,1),pattern(i,2),'m*');
        plot(center(4,1),center(4,2),'ko','MarkerFaceColor','k');
    else
        plot(pattern(i,1),pattern(i,2),'y*');
        plot(center(5,1),center(5,2),'ko','MarkerFaceColor','k');
    end
end
title('kmeans 原始数据集聚类结果');
xlabel('x data');
ylabel('y data');
%% FCM对原始自定义数据聚类,并可视化
Max = 1000;
f = 1.5;
tol = 1e-3;
[prediction] = self_fcm(k,f,Max,tol,data);
figure;
hold on;
for i=1:m
    if prediction(i) == 1
        plot(data(i,1),data(i,2),'r*');
    elseif prediction(i) == 2
        plot(data(i,1),data(i,2),'g*');
    elseif prediction(i) == 3
        plot(data(i,1),data(i,2),'b*');
    elseif prediction(i) == 4
        plot(data(i,1),data(i,2),'m*');
    else
        plot(data(i,1),data(i,2),'y*');
    end
end
title('FCM 原始数据集聚类结果');
xlabel('x data');
ylabel('y data');
%% kmeans 算法对iris数据集聚类，并可视化
[pattern_iris,center_iris] = self_kmeans(k,data_iris);
figure;
scatter3(center_iris(:,1),center_iris(:,2),center_iris(:,3),20,'MarkerFaceColor','k');
hold on;
scatter3(pattern_iris(:,1),pattern_iris(:,2),pattern_iris(:,3),20,pattern_iris(:,4))
title('kmeans Iris数据集聚类结果');
xlabel('x data');
ylabel('y data');
zlabel('z data');
%% FCM 算法对iris数据集聚类，并可视化
[prediction] = self_fcm(k,f,Max,tol,data_iris);
figure;
scatter3(data_iris(:,1),data_iris(:,2),data_iris(:,3),20,prediction);
title('FCM Iris数据集聚类结果');
xlabel('x data');
zlabel('z data');
ylabel('y data');