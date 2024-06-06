load('data/paviaU/paviaU_gt.mat');
load('pred_mat_1.mat');
pos = find(paviaU_gt==0);
pred_mat(pos)=0;
imshow(label2rgb(pred_mat));
figure(2)
imshow(label2rgb(paviaU_gt));