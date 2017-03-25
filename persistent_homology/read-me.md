To calculate the persistent homology of a data set, you first need to download Javaplex from the following site:
https://github.com/appliedtopology/javaplex/releases/tag/4.2.5

Note that for whatever reason, the script load_javaplex.m has the command "clear import", which cannot be run from a script.
So you need to enter the following directly in to Matlab:

javaaddpath('./lib/javaplex.jar');
import edu.stanford.math.plex4.*;

javaaddpath('./lib/plex-viewer.jar');
import edu.stanford.math.plex_viewer.*;

cd './utility';
addpath(pwd);
cd '..';

api.Plex4.createExplicitSimplexStream()

(end code)

The last line verifies that Javaplex is indeed working correctly. If no error appears, you're fine. Now you can go in to the
directory that has all your data and compute the persistent homology as given in the provided script.

The data that this script is applied to can be found at:
http://www.cs.columbia.edu/CAVE/software/softlib/coil-20.php

We used the processed data for our project.
