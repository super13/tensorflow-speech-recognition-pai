f=`find -name '*.flac'|gawk -F. '{print $2}'`
for i in $f
do
    ffmpeg -i .$i.flac .$i.wav
    rm -f .$i.flac
done
