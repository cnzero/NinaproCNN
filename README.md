## Files structure and their explanation
-. `usefulFcns.py`
    1. `SplitOnColumn`. split a 2D numpy array on its Column
	    to a 3D numpy array
    2. `BuildNewlyDir`.
	    newly build an empty directory.
		if it exists, delete it, and then build a new one.
	    or, build a new one.

-. `FeaturesExtraction.py`
    many features extraction functions sets
	and an `extractSlidingWindow` methods to extract [feStr] with sliding window of [LI-LW]

-. `DataPreparation.py`
    1. `getRawDict` from [.mat] file
	2. `getRmsImagesLabels` from rawDict, to construct two generalized numpy arrays 3D[Images]mx16x30 and 2D[Labels]mx8


-. `scriptDataPreparation.py` a work-through script
    1. read from .mat file
	2. extract [RMS] feature
	3. write 3D[Images]mx16x30 and 2D[Labels]mx8 to file 'ninaRmsImagesLabels.pkl' with pickle module
	4. read  3D[Images]mx16x30 and 2D[Labels]mx8 from file 'ninaRmsImagesLabels.pkl' with pickle module
	5. write == read ? checking
	6. time duration computation for every part.

-. `classNinapro.py`
    1. read [Images]&[Labels] from .pkl file
	2. split [Images]&[Labels] to Train-Test-Validate parts with a proportion
	3. `next_batch` for usage during CNN training.
    
### [.mat] data file

