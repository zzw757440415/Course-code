clear,clc;
%% ������һ������
mu1 = [0,0];  % ��ֵ
s1 = [0.25 0;0 0.25];  % Э����
data1 = mvnrnd(mu1,s1,100);  % ������һ������
%% �����ڶ�������
mu2 = [-1.25,1.25];  % ��ֵ
s2 = [0.25 0;0 0.25];  % Э����
data2 = mvnrnd(mu2,s2,100);  % �����ڶ�������
%% ��������������
mu3 = [1.25,1.25];  % ��ֵ
s3 = [0.25 0;0 0.25];  % Э����
data3 = mvnrnd(mu3,s3,100);  % ��������������
%% ���� Iris ����,����һ������
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
%% ��ʾԭʼ�Զ�������
data = [data1;data2;data3];
[m,n] = size(data);
plot(data1(:,1),data1(:,2),'r*');
hold on;
plot(data2(:,1),data2(:,2),'g*');
plot(data3(:,1),data3(:,2),'b*');
title('ԭʼ���ݷֲ�');
xlabel('x data');
ylabel('y data');
%% ����PCA��iris���ݼ���ά�������ӻ�
[pc,score,latent,tsquare]=pca(data_iris);
res = cumsum(latent)./sum(latent);
disp(res);  % ��ʾÿһά��ԭʼ���ݵľ���
data_iris = score(:,1:3);  % ����PCA����3ά
figure;
scatter3(data_iris(:,1),data_iris(:,2),data_iris(:,3),40,Y,'*');
title('Iris����ԭʼ�ֲ�');
xlabel('x data');
ylabel('y data');
zlabel('z data');
%% kmeans��ԭʼ�Զ������ݾ���,�����ӻ�
k = input('������������');  % ������� 
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
title('kmeans ԭʼ���ݼ�������');
xlabel('x data');
ylabel('y data');
%% FCM��ԭʼ�Զ������ݾ���,�����ӻ�
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
title('FCM ԭʼ���ݼ�������');
xlabel('x data');
ylabel('y data');
%% kmeans �㷨��iris���ݼ����࣬�����ӻ�
[pattern_iris,center_iris] = self_kmeans(k,data_iris);
figure;
scatter3(center_iris(:,1),center_iris(:,2),center_iris(:,3),20,'MarkerFaceColor','k');
hold on;
scatter3(pattern_iris(:,1),pattern_iris(:,2),pattern_iris(:,3),20,pattern_iris(:,4))
title('kmeans Iris���ݼ�������');
xlabel('x data');
ylabel('y data');
zlabel('z data');
%% FCM �㷨��iris���ݼ����࣬�����ӻ�
[prediction] = self_fcm(k,f,Max,tol,data_iris);
figure;
scatter3(data_iris(:,1),data_iris(:,2),data_iris(:,3),20,prediction);
title('FCM Iris���ݼ�������');
xlabel('x data');
zlabel('z data');
ylabel('y data');