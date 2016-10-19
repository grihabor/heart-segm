run: dirs data/manual_seg data/mrimages 
	python parse.py data	

data/mrimages: dirs data/mrimages.tar.gz
	cd data/mrimages; tar -xvzf mrimages.tar.gz
	
data/mrimages.tar.gz: dirs
	wget http://www.cse.yorku.ca/~mridataset/mrimages.tar.gz
	mv mrimages.tar.gz data/mrimages/mrimages.tar.gz 

data/manual_seg: dirs data/manual_seg.tar.gz
	cd data/manual_seg; tar -xvzf manual_seg.tar.gz
	
data/manual_seg.tar.gz: dirs
	wget http://www.cse.yorku.ca/~mridataset/manual_seg.tar.gz 
	mv manual_seg.tar.gz data/manual_seg/manual_seg.tar.gz 

dirs: 
	mkdir -p data
	mkdir -p data/manual_seg
	mkdir -p data/mrimages
