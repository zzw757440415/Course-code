function [prediction] = self_fcm(k,f,Max,tol,data)
metric = @euclidean;
[m,n] = size(data);
v = repmat(max(data),k,1).*rand([k,n]);
U = rand([k,m]);
for j = 1:m
	U(:,j) = U(:,j)./sum(U(:,j));
end
for i = 1:k
	v(i,:) = sum((data(:,:).*repmat(U(i,:)'.^f,1,n)),1)./sum(U(i,:).^f);
end

v_old = v;
delta = 1e4;
ks = 0;
while (ks<Max && delta>tol)
	for i = 1:k
		for j = 1:m
			U(i,j) = 1/sum((metric(data(j,:),v(i,:))./metric(data(j,:),v)).^(2/(f-1)));
		end
	end
	for i = 1:k
		v(i,:) = sum((data(:,:).*repmat(U(i,:)'.^f,1,n)),1)./sum(U(i,:).^f);
	end
	v_new = v;
	delta = max(max(abs(v_new - v_old)));
	v_old = v;
	ks = ks + 1;
end
prediction = zeros([1, m]);
for i = 1:m
	[M, prediction(i)] = max(U(:,i));
end
end

