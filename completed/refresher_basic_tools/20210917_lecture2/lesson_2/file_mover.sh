# mv "$1/*.txt" "$2/*.txt"
for name in $(ls *.txt)
do
  mv "$1$name" "$2$name"
done
