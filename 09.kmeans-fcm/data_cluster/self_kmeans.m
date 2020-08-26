function [pattern,center] = self_kmeans(k,data)
%% 初始化变量
[m,n] = size(data);
center = zeros(k,n);
pattern = data;
%% kmeans算法
for x=1:k
    center(x,:) = data(randi(m,1),:);  % 随机产生k个中心点
end
while true
	distance = zeros(1,k);
    num = zeros(1,k);
    new_center = zeros(k,n);  % 保存新的聚类中心
    for x=1:m
        for y=1:k
            distance(y) = norm(data(x,:)-center(y,:));  % 计算每个样本到三个聚类中心的距离
        end
        [dis,temp] = min(distance);  % 计算最小距离，dis为距离值，temp为第几个
        pattern(x,n+1) = temp;  % 保存每个样本点的类别
    end
    kk = 0;
	%% 将所有在同一个类中的点坐标全部相加，计算新的中心坐标
    for y=1:k
        for x=1:m
            if pattern(x,n+1)==y
                new_center(y,:) = new_center(y,:) + pattern(x,1:n);
                num(y) = num(y) + 1;
            end
        end
        new_center(y,:) = new_center(y,:) / num(y);
        if norm(new_center(y,:) - center(y,:)) < 0.1
            kk = kk + 1;
        end
    end
    if kk == k
        break;  % 新旧三个聚类中心的距离均小于0.1时，结束迭代
    else
        center = new_center;
    end
end
end


