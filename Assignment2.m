%% Assignment 2
% Programmer: Kevin Karch
% Class: CS383 - Machine Learning

clc;
clear all;
close all;
fclose('all');
if exist('plots', 'dir')
   rmdir('plots','s')
end

%% Data Prep

rawData = csvread('diabetes.csv');
%rawData = sort(rawData,1); % Sort data ascending (-1 -> 1)
%split = find(rawData==1,1); % Find the first positive instance
[rows,cols] = size(rawData);
X = rawData(:,2:cols);
Y = rawData(:,1);

%% Standardize Data

data = double(X);
std_X = [];
[rows,cols] = size(X);
for i = 1:cols
    avg = mean(data(:,i));
    std_dev = std(data(:,i));
    for j = 1:rows
        std_X(j,i) = (data(j,i)-avg)/std_dev;
    end
end

clear i j avg std_dev

%% Seed RNG
rng(0)

%% Call myKMeans
myKMeans(std_X,7,Y); %Comment to Debug

%% K-Means Function 

function myKMeans(X,k,Y) %Comment to Debug
%X = std_X; % Uncomment to Debug
%Constants
%k = 7; %Uncomment to Debug
if k>7
    k = 7;
end

blue=[0,0.4470,0.7410];orange=[0.8500,0.3250,0.0980];
yellow=[0.9290,0.6940,0.1250];purple=[0.4940,0.1840,0.5560];
green=[0.4660, 0.6740, 0.1880];cyan=[0.3010, 0.7450, 0.9330];
magenta=[0.6350, 0.0780, 0.1840];	
[rows,cols] = size(X);
imlibrary = [];

%PCA
if cols > 3
cov = (transpose(X)*X)/rows; 
[eigenVec,eigenVal] = eig(cov);


[erows,ecols] = size(eigenVal); %Flatten EigenValues
flatEigVal = [];
for i=1:erows
    flatEigVal = [flatEigVal,eigenVal(i,i)];
end

[max1, ind1] = max(flatEigVal); %Take top 3 EigenMaxes
flatEigVal(ind1)      = -Inf;
[max2, ind2] = max(flatEigVal);
flatEigVal(ind2)      = -Inf;
[max3, ind3] = max(flatEigVal);
flatEigVal(ind3)      = -Inf;

X = X(:,[ind1,ind2,ind3]); %Extract top 3 contributors

clear erows ecols i  ind1 ind2 ind3 max1 max2 max3 flatEigVal
end % End PCA 

% Select Random Start 
%rng(0) %Seed RNG
startIndex = randperm(rows,k); %Vector of k random numbers
means = [];
for i = 1:length(startIndex) %Vectorize starting means
    means = [means;X(startIndex(i),:)];
end

DM = 1;
iteration = 0;
while abs(DM) > (2^-23)
iteration = iteration+1;
%Distance Calculations D = [D(X1,mean1),D(X1,mean2),...D(X1,meank)]
% Vector distance from X(j,:) to mean 
D = [];
for i = 1:k
    dCalc = [];
    for j = 1:rows
        dCalc = [dCalc;norm(means(i,:)-X(j,:))];
    end
    D = [D,dCalc];
end

%Cluster
C(1:rows,1:3,1:k) = 0; %Create MultiDimensional Matrix
for i = 1:rows 
    [v,index] = min(D(i,:)); %find index of min distance
    C(i,:,index) = X(i,:); %stick that observation in that cluster 
end

colors = [blue;orange;yellow;purple;green;cyan;magenta];

%Calculate Purity
clusterPurity = 0;
Purity = 0;
for i = 1:k %For every Cluster
    temp = C(:,:,i); % Copy cluster into temp
    temp = [temp,Y];
    temp2 = [];
    for j = 1:rows
        if temp(j,1) == 0 && temp(j,2) == 0 && temp(j,3) == 0 %Remove 0 vectors
            continue
        else
            temp2 = [temp2;temp(j,:)]; % Recombined X and Y
        end
    end
    pos = sum(temp2(:,4)==1); %Count classes
    neg = sum(temp2(:,4)==(-1));
    Purity = Purity + max(pos,neg); %Purity Calculations
end
Purity = Purity/rows; %Composite Purity

%Plot Clusters
tit = ['Iteration ',num2str(iteration),' -',' Purity = ',num2str(Purity)];  %Title 'Iteration n - Purity = ...'
clf %Clear Last Figure
figure1 = figure(1);
for i = 1:k %For every Cluster
    color = colors(i,1:3); %Select Color
    scatter3(C(:,1,i),C(:,2,i),C(:,3,i),36,color,'x') %Plot Cluster i
    if i == 1
        hold on
    end
    scatter3(means(i,1),means(i,2),means(i,3),75,'MarkerEdgeColor','k','MarkerFaceColor',color); %Plot Mean in Same Color
    title(tit);
end

%Save Plots 
if ~exist('plots', 'dir')   %Create Plots Folder and Add to Work Directory
   mkdir('plots')
   addpath('plots')
end
imname = ['plots/figure',num2str(iteration),'.jpg']; %Name the plots and add them to the library
imlibrary = [imlibrary;string(imname)];
saveas(figure1,imname) 

%pause(1)% Pause for viewer in debugging

%Calculate new means
oldmeans = means;       
means = [];
for i = 1:k                     %Move cluster to temp working space
    temp = C(:,:,i);
    temp = temp(any(temp,2),:);     %Strip 0 vectors
    means = [means;mean(temp)];     %Compute mean
end

%Calculate Manhattan distance D(oldmean,mean)
DM = 0;
for i = 1:k
    Di = (means(i,1)-oldmeans(i,1))+(means(i,2)-oldmeans(i,2))+(means(i,3)-oldmeans(i,3));
    DM = DM + Di;
end
        
end %End While

%Construct Video
vname = ['K_',num2str(k),'_F_all'];
video = VideoWriter(vname); %create the video object
video.FrameRate = 1;
%video.FileFormat = 'mp4';
open(video); %open the file for writing
[numImg,c] = size(imlibrary);
for ii=1:numImg %where N is the number of images
  I = imread(char(imlibrary(ii))); %read the next image
  writeVideo(video,I); %write the image to file
end
close(video); %close the file

if exist('plots', 'dir')
   rmdir('plots','s')
end
end %End Function % Comment to Debug


