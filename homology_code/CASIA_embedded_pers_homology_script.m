folders = dir;
folders = {folders.name};
for j =86:length(folders)
    foldername = strcat('./',folders{j});
    cd(foldername);
    files = dir;
    names = {files.name};
    for i = 1:length(names)
        if length(names{i})==7
            fileID = fopen(names{i},'r');
            formatSpec = '%f';
            Xvect = fscanf(fileID,formatSpec);
            break
        else
            continue
        end
    end
    Xmat = Xvect';
    for l =i+1:length(names)
        if length(names{l})==7
            fileID = fopen(names{l},'r');
            Xvect = fscanf(fileID,formatSpec);
            Xmat = [Xmat;Xvect'];
        else
            continue
        end
    end
    max_dimension = 2;
    max_filtration_value = 50;
    num_divisions = 10000;
    stream = api.Plex4.createVietorisRipsStream(Xmat,max_dimension,max_filtration_value,num_divisions);
    persistence = api.Plex4.getModularSimplicialAlgorithm(max_dimension,2);
    intervals = persistence.computeIntervals(stream);
    name = char(strcat(folders{j},' Z2 coeffs embedded'));
    options.filename = name;
    plot_barcodes(intervals,options);
    cd ..
    clearvars -except folders j
end
