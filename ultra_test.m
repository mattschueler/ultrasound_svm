mfile = matfile("FILENAME.mat");

imdata = mfile.ImgData;
imdata = imdata{1}(:,:,1,:);
lgI = log(imdata);

B=reshape(lgI, [39680,2000]);
csvwrite("FILENAME.csv", B');
C = reshape(B(:,1),[310,128]);
