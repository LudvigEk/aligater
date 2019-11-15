#!/bin/bash

#$1 = src
#$2 = tar
#$3 = text file with sample images, one path per line
SRC=$1
TAR=$2
SAMPS=$3

echo $SRC
echo $TAR
echo $SAMPS

readarray -t FILES < $SAMPS
for f in "${FILES[@]}"
do
  cp "${f}" "${TAR}"
done
