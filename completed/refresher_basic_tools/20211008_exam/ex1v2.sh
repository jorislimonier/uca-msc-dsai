case $3 in
  "ls")
    ls $1*$2
  ;;
  "rm")
    rm $1*$2
  ;;
  "mv")
	for i in $1*$2
	do
		mv "$i" "${i%.*}_1$2"
	done
esac
