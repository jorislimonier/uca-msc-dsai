for i in $(seq 1 $1)
do
  mv "$2/$3$i" "$2/$3$i.txt"
done
