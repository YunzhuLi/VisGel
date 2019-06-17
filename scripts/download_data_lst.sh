echo "Downloading data lists for vision2touch ..."

mkdir -p ./data_lst
DATA_PATH=./data_lst/train_vision2touch.txt
URL=http://visgel.csail.mit.edu/files/vision2touch/train_lst.txt

wget -N $URL -O $DATA_PATH

DATA_PATH=./data_lst/eval_vision2touch.txt
URL=http://visgel.csail.mit.edu/files/vision2touch/eval_lst.txt

wget -N $URL -O $DATA_PATH

DATA_PATH=./data_lst/weight_vision2touch.txt
URL=http://visgel.csail.mit.edu/files/vision2touch/weight_lst.txt

wget -N $URL -O $DATA_PATH


echo "Downloading data lists for touch2vision ..."

mkdir -p ./data_lst
DATA_PATH=./data_lst/train_touch2vision.txt
URL=http://visgel.csail.mit.edu/files/touch2vision/train_lst.txt

wget -N $URL -O $DATA_PATH

DATA_PATH=./data_lst/eval_touch2vision.txt
URL=http://visgel.csail.mit.edu/files/touch2vision/eval_lst.txt

wget -N $URL -O $DATA_PATH
