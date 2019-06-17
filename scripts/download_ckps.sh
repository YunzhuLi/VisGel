echo "Downloading vision2touch ckp ..."

mkdir -p ./dump_vision2touch/ckps
MODEL_FILE=./dump_vision2touch/ckps/netG_epoch74.pth
URL=http://visgel.csail.mit.edu/files/vision2touch/ckps/netG_epoch74.pth

wget -N $URL -O $MODEL_FILE


echo "Downloading touch2vision ckp ..."

mkdir -p ./dump_touch2vision/ckps
MODEL_FILE=./dump_touch2vision/ckps/netG_epoch74.pth
URL=http://visgel.csail.mit.edu/files/touch2vision/ckps/netG_epoch74.pth

wget -N $URL -O $MODEL_FILE

