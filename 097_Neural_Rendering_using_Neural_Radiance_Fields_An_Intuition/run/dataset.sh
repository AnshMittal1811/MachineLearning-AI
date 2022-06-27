mkdir -p data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1e-OFd6kMtz1x1zO0RQh5hVYfD_yfdJii' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1e-OFd6kMtz1x1zO0RQh5hVYfD_yfdJii" -O data/replica.zip && rm -rf /tmp/cookies.txt
unzip data/replica.zip -d data/
rm data/replica.zip

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1nt1F-GxDwkQXOIyO-VZwy230fO7iqLam' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1nt1F-GxDwkQXOIyO-VZwy230fO7iqLam" -O data/TanksAndTemple.zip && rm -rf /tmp/cookies.txt
unzip data/TanksAndTemple.zip -d data/
mv data/data/TanksAndTemple data/
rm -r data/data
rm data/TanksAndTemple.zip