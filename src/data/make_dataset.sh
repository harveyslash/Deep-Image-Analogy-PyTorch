#!/bin/bash

# set download directory
download_dir=${DATA_PATH}/raw

# create download directory if it does not exist
[[ ! -d ${download} ]] && mkdir -p ${download_dir}

# set links to download from
declare -a links=('http://dpegb9ebondhq.cloudfront.net/product_photos/36966876/Mona-Lisa_homepage.jpg' 'https://i.pinimg.com/736x/0d/2c/04/0d2c041de3d73541c252759d91a565f6--avatar-disney-princess.jpg')

# set shortname for the downloaded images
declare -a aliases=('mona.jpg' 'neytiri.jpg')

# loop over links and download them
for idx in "${!links[@]}"
do
    curl -o ${download_dir}/${aliases[${idx}]} ${links[${idx}]}
done

