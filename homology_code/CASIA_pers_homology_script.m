folders = dir;
folders = {folders.name};
for j = 81:length(folders)
    foldername = strcat('./',folders{j});
    cd(foldername);
    files = dir;
    names = {files.name};
    for i = 1:length(names)
        if length(names{i})==7
            X = imread(char(names{i}));
            X = rgb2gray(X);
            Xvect = X(:,1);
            for l = 2:length(X)
                Xvect = [Xvect ; X(:,1)];
            end
            break
        else
            continue
        end
    end
    Xmat = Xvect';
    for l =i+1:length(names)
        if length(names{l})==7
            X = imread(char(names{l}));
            X = rgb2gray(X);
            Xvect = X(:,1);
            for k =2:length(X)
                Xvect = [Xvect;X(:,k)];
            end
            Xmat = [Xmat;Xvect'];
        else
            continue
        end
    end
    max_dimension = 2;
    max_filtration_value = 30000;
    num_divisions = 10000;
    stream = api.Plex4.createVietorisRipsStream(Xmat,max_dimension,max_filtration_value,num_divisions);
    persistence = api.Plex4.getModularSimplicialAlgorithm(max_dimension,2);
    intervals = persistence.computeIntervals(stream);
    name = char(strcat(folders{j},' Z2 coeffs grayscale'));
    options.filename = name;
    plot_barcodes(intervals,options);
    cd ..
    clearvars -except folders j
end
