case $3 in
  "ls")
    ls $1*$2
  ;;
  "rm")
    rm $1*$2
  ;;
  "mv")
	for i in ls $1*$2
	do
		echo "------$i"
		#mv "$i" "${i}_1"
	done
esac
