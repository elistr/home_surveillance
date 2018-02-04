import urllib2
import socket
import time

URL = "http://192.168.33.21/videostream.cgi?user=dali&pwd=dali"
TEMP_FILENAME = "img2_%09d.jpg"


def download():
    counter = 0
    passman = urllib2.HTTPPasswordMgrWithDefaultRealm()
    passman.add_password(None, URL, None, None)
    authhandler = urllib2.HTTPBasicAuthHandler(passman)
    opener = urllib2.build_opener(authhandler)
    urllib2.install_opener(opener)

    JPEG_MAGIC_START = 'FFD8'.decode('hex')
    JPEG_MAGIC_END = 'FFD9'.decode('hex')

    while True:
        try:
            url_open = urllib2.urlopen(URL)
        except (urllib2.HTTPError, socket.error):
            print ('Got error: Trying again..')
            continue

        grabbed = url_open.read(100000)
        # print 'Grabbed %s bytes of data' % len(grabbed)

        if JPEG_MAGIC_START in grabbed and JPEG_MAGIC_END in grabbed:
            jpeg_data = grabbed[grabbed.index(JPEG_MAGIC_START):grabbed.index(JPEG_MAGIC_END) + 2]
            f = open((TEMP_FILENAME % counter), 'wb')
            f.write(jpeg_data)
            f.close()
            counter += 1
            time.sleep(1)
            continue


if __name__ == "__main__":
    download()