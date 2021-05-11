function [X,Y, y] = LoadBatch(file)
%LOADBATCH reads in the data from a CIFAR-10 batch file
%and returns the image and label data in separate files.
A = load(file);
X = double(A.data')/255;

%label vector of size n
y = A.labels';
y = y + uint8(ones(1,length(y))); %Simplifying indexing

%Create the image label matrix Y of size Kxn where K = #of labels.
Y= zeros(10, length(y));
for  i= 1:length(Y)
    Y(y(i),i) = 1;
end
end

