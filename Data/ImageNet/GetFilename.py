from urllib import request
IMG_LIST_URL="http://www.image-net.org/api/text/imagenet.synset.geturls.getmapping?wnid={}"

url = IMG_LIST_URL.format("n02113335")
with request.urlopen(url) as response:
    html = response.read() 
    print(html)