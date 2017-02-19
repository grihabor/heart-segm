CUR_DIR = $(shell pwd)

all: dirs data/manual_seg data/mrimages data/input

data/input: parse.py
	python parse.py data
	

data/mrimages: data/mrimages/mrimages.tar.gz
	cd data/mrimages; tar -xvzf mrimages.tar.gz
	
data/mrimages/mrimages.tar.gz:
	wget http://www.cse.yorku.ca/~mridataset/mrimages.tar.gz
	touch mrimages.tar.gz
	mv mrimages.tar.gz data/mrimages/mrimages.tar.gz 

data/manual_seg: data/manual_seg/manual_seg.tar.gz
	cd data/manual_seg; tar -xvzf manual_seg.tar.gz
	
data/manual_seg/manual_seg.tar.gz: 
	wget http://www.cse.yorku.ca/~mridataset/manual_seg.tar.gz 
	touch manual_seg.tar.gz
	mv manual_seg.tar.gz data/manual_seg/manual_seg.tar.gz 

dirs: 
	mkdir -p data
	mkdir -p data/manual_seg
	mkdir -p data/mrimages

build:
	docker build -t grihabor/caffe . 

run:
	docker run -i -t -v $(CUR_DIR):/root/project bvlc/caffe:cpu /bin/bash -c "cd /root/project; bash"
