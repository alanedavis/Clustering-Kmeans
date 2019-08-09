function myKMeans = myKMeans(X, k)
%% myKMeans
clf('reset');

videoTest = VideoWriter('testVideo.avi');
videoTest.FrameRate = 1;
purityEnabled = 0;

if exist('Y', 'var')
    purityEnabled = 1;
    X = [Y X];
    random_X = X(randperm(size(X, 1)), :);
    Y = X(:, 1);
    X(:, 1) = [];
else
    purityEnabled = 0;
    random_X = X(randperm(size(X, 1)), :);
end

covarianceMatrix = cov(random_X);

[eigValue, eigVector] = eigs(covarianceMatrix);
Z1 = X * eigValue(:, 1);
Z2 = X * eigValue(:, 2);

sizeOfNormX = size(X);

if sizeOfNormX(2) <= 2
    az =90;
    el =90;
    Z3 = zeros(sizeOfNormX(1), 1);
else
    az = 37.5;
    el = 30;
    Z3 = X * eigValue(:, 3);
end

view(az,el);

dataset = [Z1, Z2, Z3];
figure
%scatter3(Z1, Z2, Z3);
%title('3D PCA Projection of Data');

%title('2D PCA Projection of Data');

%% Create three random points for each centriod

centriods = {[dataset(1,1) dataset(1, 2) dataset(1, 3)], 'r', [], [], [], [0 0];...
    [dataset(2,1) dataset(2, 2) dataset(2, 3)], 'b', [], [], [], [0 0];...
    [dataset(3,1) dataset(3, 2) dataset(3, 3)], 'g', [], [], [], [0 0];...
    [dataset(4,1) dataset(4, 2) dataset(4, 3)], 'c', [], [], [], [0 0];...
    [dataset(5,1) dataset(5, 2) dataset(5, 3)], 'm', [], [], [], [0 0];...
    [dataset(6,1) dataset(6, 2) dataset(6, 3)], 'y', [], [], [], [0 0];...
    [dataset(7,1) dataset(7, 2) dataset(7, 3)], 'k', [], [], [], [0 0];};

%The 4 empty arrays at the end of the cell array above will have x y
%z and the purity stored in them respectively. The first zero is the number
%of postives, the second is the total number of points under the cluster.

%1 is red r
%2 is blue b
%3 is green g
%4 is cyan c
%5 is magenta m
%6 is yellow y
%7 is black k

figure(1)
hold on
view(az,el);
%% Look at each point within the dataset and see what color they are, and plot result

distances = [];
minDistCentriod = 0;
Time = 1;

for i = 1:length(Z1)
    for j = 1:k
        distances = [distances sqrt((centriods{j, 1}(1) - dataset(i, 1))^2 + (centriods{j, 1}(2) - dataset(i, 2))^2 + (centriods{j, 1}(3) - dataset(i, 3))^2 )];
    end
    
    minDistCentriod = find(distances == min(distances));
    distances = [];
    scatter3(dataset(i, 1), dataset(i, 2), dataset(i, 3), 'x','MarkerEdgeColor', centriods{minDistCentriod, 2}); 

    centriods{minDistCentriod, 3} = [centriods{minDistCentriod, 3} dataset(i, 1)];
    centriods{minDistCentriod, 4} = [centriods{minDistCentriod, 4} dataset(i, 2)];
    centriods{minDistCentriod, 5} = [centriods{minDistCentriod, 5} dataset(i, 3)];
    
    if purityEnabled
        centriods{minDistCentriod, 6}(2) = centriods{minDistCentriod, 6}(2) + 1;
        if Y(i) == 1
            centriods{minDistCentriod, 6}(1) = centriods{minDistCentriod, 6}(1) + 1;
        end
    end
            
end

for i = 1:k
scatter3(centriods{i, 1}(1), centriods{i, 1}(2), centriods{i, 1}(3), 'o',...
    'MarkerFaceColor',centriods{i, 2},'LineWidth',1.5,'MarkerEdgeColor','k');
% txt = centriods{i, 6}(1) / centriods{i, 6}(2)
% text(centriods{i, 1}(1),centriods{i, 1}(2),centriods{i, 1}(3), num2str(txt));
end

title('Iteration 1');

hold off

F(Time) = getframe(gcf);
Time = Time + 1;

view(az,el);


clf('reset')

%% Use the intial setup created earlier to reiterate this cluster process until the mean of all centriods doesn't change.

meanChanged = 1;

while meanChanged
    clf('reset')
    hold on
    for n = 1:k
        centriods{n, 1} = [mean(centriods{n, 3}), mean(centriods{n, 4}), mean(centriods{n, 5})];
        centriods{n, 3} = [];
        centriods{n, 4} = [];
        centriods{n, 5} = [];
        scatter3(centriods{n, 1}(1), centriods{n, 1}(2), centriods{n, 1}(3), 'o',...
        'MarkerFaceColor',centriods{n, 2},'LineWidth',1.5,'MarkerEdgeColor','k')
    end
    
    for i = 1:length(Z1)
        for j = 1:k
            distances = [distances sqrt((centriods{j, 1}(1) - dataset(i, 1))^2 + (centriods{j, 1}(2) - dataset(i, 2))^2 + (centriods{j, 1}(3) - dataset(i, 3))^2 )];
        end

        minDistCentriod = find(distances == min(distances));
        distances = [];
        scatter3(dataset(i, 1), dataset(i, 2), dataset(i, 3), 'x','MarkerEdgeColor', centriods{minDistCentriod, 2}); 

        centriods{minDistCentriod, 3} = [centriods{minDistCentriod, 3} dataset(i, 1)];
        centriods{minDistCentriod, 4} = [centriods{minDistCentriod, 4} dataset(i, 2)];
        centriods{minDistCentriod, 5} = [centriods{minDistCentriod, 5} dataset(i, 3)];

    end
    title(['Iteration ' num2str(Time)]);
    hold off
    view(az,el);
    F(Time) = getframe(gcf);
    Time = Time + 1;
    
    for m = 1:k
        if abs(norm(centriods{m, 1}) - norm([mean(centriods{m, 3}), mean(centriods{m, 4}), mean(centriods{m, 5})])) <= 2^(-23) 
            meanChanged = 0;
        else
            meanChanged = 1;
            break
        end
    end
end

open(videoTest);

for i = 1:length(F)
    frame = F(i);
    writeVideo(videoTest, frame);
end

close(videoTest);

end