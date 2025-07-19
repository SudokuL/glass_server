import wget
# -*- coding: utf-8 -*-
url = "https://storage.googleapis.com/kaggle-models-data/135843/159788/bundle/archive.tar.gz?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1752933219&Signature=feVndh71P0RqF0Y9%2FI6QL95AiFmICj%2Fz49eU%2Bxh2NkkHZr9lx2cJHvo2ORzMbW6py9AcolRirVCHKami9NhMIZFRbsAwTTAsWGa2ixbwsnGkBSGk566TKU9D0VMAHfj0qshHQ%2FijmXSzwAWKQmuZQ678xS8nrTnSaQmkhoENbgi7X5V9hBXekbA0TomrLy1dDPPmLovpsorQP6zn1Cvanqe26WVDj%2FBUEe3NAFeS3nPOgCKPMQNU2yjAX%2FstqsK0QXOiGChc7GIS87b0mmJxVWVx65oTz1DyZlzdPu9miOmR5SxzrGAzoIb6aOEbtEiD4td2lpqOA%2F9irn9Es4osGQ%3D%3D&response-content-disposition=attachment%3B+filename%3Dchinese-dish-nutrient-estimation-pytorch-default-v1.tar.gz"
output_filename = 'chinese-dish-nutrient-estimation-pytorch-default-v1.tar.gz' 
wget.download(url, out='/root/autodl-tmp/.autodl/iot/food_cv/'+output_filename)
 
