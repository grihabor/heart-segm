FROM kaixhin/caffe
COPY fcn.berkeleyvision.org ~/project/fcn
RUN ~/caffe/scripts/download_model_binary.py ~/project/fcn/voc-fcn8s
COPY requirements.txt ~/project/requirements.txt
RUN pip install -r ~/project/requirements.txt