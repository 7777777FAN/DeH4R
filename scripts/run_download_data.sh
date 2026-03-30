# The data was neither collected, published, nor stored by us; we only provide a convenient way to download it.

# cityscale
echo " ============ Downloading CityScale dataset ============"
# the file id is copied from https://github.com/TonyXuQAQ/RNGDetPlusPlus/blob/master/cityscale/prepare_dataset/preprocessing.bash
gdown 1R8sI1RmFe3rUfWMQaOfsYlBDHpQxFH-H     
unzip data.zip
mkdir -p ./data/cityscale
mv ./data/20cities ./data/data_split.json ./data/cityscale

# spacenet
echo " ============ Downloading SpaceNet dataset ============"
# the file id is copied from https://github.com/TonyXuQAQ/RNGDetPlusPlus/blob/master/spacenet/prepare_dataset/preprocessing.bash
gdown 1FiZVkEEEVir_iUJpEH5NQunrtlG0Ff1W
unzip RGB_1.0_meter_full.zip
mkdir -p ./data/spacenet
mv ./RGB_1.0_meter ./data/spacenet
mv ./data/spacenet/RGB_1.0_meter/dataset.json ./data/spacenet/data_split.json
