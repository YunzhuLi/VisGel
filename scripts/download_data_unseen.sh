echo "Downloading data for unseen objects ... (83.2 GB)"

mkdir -p ./data/data_unseen
DATA_PATH=./data/data_unseen.zip
URL=http://data.csail.mit.edu/visgel/data_unseen.zip

wget -N $URL -O $DATA_PATH

echo "Unzipping ..."

unzip $DATA_PATH -d ./data/data_unseen
rm $DATA_PATH

