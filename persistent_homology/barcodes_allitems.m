
% Iterates over all 20 objects. For each value of k, we computes object k's
% persistent homology. This is saved as a .png file in your working
% directory
for k = 1:20
    % Filename of image
    pic = strcat('obj',string(k),'__0','.png');
    X = imread(char(pic));
    % Takes first column of matrix and makes it a vector
    Xvect = X(:,1);
    %This loop goes through the remaining columns and concatenates them
    %onto Xvect
    for i = 2:128
        Xvect = [Xvect;X(:,i)];
    end
    % Has Xvect as first row of matrix Xmat
    Xmat = Xvect';
    % For all remaining 71 images, do the same process as above and add the
    % resulting vector as the i^th row of Xmat
    for i = 1:71
        picname = strcat('obj',string(k),'__',string(i),'.png');
        X = imread(char(picname));
        Xvect = X(:,1);
        for j = 2:128
            Xvect = [Xvect;X(:,j)];
        end
        Xmat = [Xmat;Xvect'];
    end
    
    %Normalizes data
    
    Xmat = Xmat/255
    % Computation of persistent homology.
    max_dimension = 2;
    max_filtration_value = 10000;
    num_divisions = 10000;
    stream = api.Plex4.createVietorisRipsStream(Xmat,max_dimension,max_filtration_value,num_divisions);
    persistence = api.Plex4.getModularSimplicialAlgorithm(max_dimension,2);
    intervals = persistence.computeIntervals(stream);
    name = char(strcat('obj',string(k),' Z2 coeffs'));
    options.filename = name;
    plot_barcodes(intervals,options);
end