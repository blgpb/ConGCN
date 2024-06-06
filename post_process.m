% 2021年10月22日10:02:42
% 将可视化的pred_mat的边缘美化
load('data/paviaU/paviaU_gt.mat');
load('pred_mat_1.mat');
pos = find(paviaU_gt==0);
pred_mat(pos)=0;
imshow(label2rgb(pred_mat)); % predicted result
figure(2)
imshow(label2rgb(paviaU_gt)); % groundtruth