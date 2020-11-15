# Download and prepare the datasets

# text8
# source http://mattmahoney.net/dc/textdata.html
# wc -c text8
# 100000000 text8
# like Mikolov and others (cf. https://arxiv.org/pdf/1808.04444.pdf)
# we split the data into 90M characters for train,
# 5M characters for dev, and 5M characters for test

wget http://mattmahoney.net/dc/text8.zip
unzip text8.zip
mkdir text8_dir
mv text8.zip text8 text8_dir

{
  head -c  90000000 > text8_dir/text8.train.raw
  head -c   5000000 > text8_dir/text8.valid.raw
  head -c   5000000 > text8_dir/text8.test.raw
} < text8_dir/text8

# enwik8
# source http://mattmahoney.net/dc/textdata.html
wget http://mattmahoney.net/dc/enwik8.zip
unzip enwik8.zip
mkdir enwik8_dir
mv enwik8.zip enwik8 enwik8_dir

{
  head -c  90000000 > enwik8_dir/enwik8.train.raw
  head -c   5000000 > enwik8_dir/enwik8.valid.raw
  head -c   5000000 > enwik8_dir/enwik8.test.raw
} < enwik8_dir/enwik8

# enwik9
# source http://mattmahoney.net/dc/textdata.html
wget http://mattmahoney.net/dc/enwik9.zip
unzip enwik9.zip
mkdir enwik9_dir
mv enwik9.zip enwik9 enwik9_dir

{
  head -c  900000000 > enwik9_dir/enwik9.train.raw
  head -c   50000000 > enwik9_dir/enwik9.valid.raw
  head -c   50000000 > enwik9_dir/enwik9.test.raw
} < enwik9_dir/enwik9

# wikitext-2-raw
# source https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip
unzip wikitext-2-raw-v1.zip
mv wikitext-2-raw-v1.zip wikitext-2-raw

# wikitext-103-raw
# source https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
mv wikitext-103-raw-v1.zip wikitext-103-raw

ls -al *
