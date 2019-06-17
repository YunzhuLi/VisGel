echo "Downloading vision2touch demo data list ..."

mkdir -p ./data_lst
DATA_PATH=./data_lst/demo_vision2touch.txt
URL=http://visgel.csail.mit.edu/files/vision2touch/demo_lst.txt

wget -N $URL -O $DATA_PATH

echo "Downloading vision2touch demo data ..."

mkdir -p ./data/data_demo_vision2touch
DATA_PATH=./data/data_demo_vision2touch/demo.zip
URL=http://visgel.csail.mit.edu/files/vision2touch/demo.zip

wget -N $URL -O $DATA_PATH

echo "Unzipping ..."

unzip $DATA_PATH -d ./data/data_demo_vision2touch
rm $DATA_PATH


#######################


echo "Downloading touch2vision demo data list ..."

mkdir -p ./data_lst
DATA_PATH=./data_lst/demo_touch2vision.txt
URL=http://visgel.csail.mit.edu/files/touch2vision/demo_lst.txt

wget -N $URL -O $DATA_PATH

echo "Downloading touch2vision demo data ..."

mkdir -p ./data/data_demo_touch2vision
DATA_PATH=./data/data_demo_touch2vision/demo.zip
URL=http://visgel.csail.mit.edu/files/touch2vision/demo.zip

wget -N $URL -O $DATA_PATH

echo "Unzipping ..."

unzip $DATA_PATH -d ./data/data_demo_touch2vision
rm $DATA_PATH

