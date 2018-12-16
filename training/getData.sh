# get COCO dataset
# mkdir data
# mkdir data/coco/
# cd data/coco/
# cd ../../

mkdir data/coco/images

wget http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip

unzip person_keypoints_trainval2014.zip -d data/coco/
unzip val2014.zip -d data/coco/images
unzip test2014.zip -d data/coco/images
unzip train2014.zip -d data/coco/images
unzip test2015.zip -d data/coco/images

rm -f person_keypoints_trainval2014.zip
rm -f test2015.zip
rm -f test2014.zip
rm -f train2015.zip
rm -f val2014.zip
