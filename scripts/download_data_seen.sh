echo "Downloading data for seen objects ... (328 GB)"

mkdir -p ./data/data_seen
DATA_PATH=./data/data_seen.zip
URL=http://data.csail.mit.edu/visgel/data_seen.zip

wget -N $URL -O $DATA_PATH

echo "Unzipping ..."

unzip $DATA_PATH -d ./data/data_seen
rm $DATA_PATH

