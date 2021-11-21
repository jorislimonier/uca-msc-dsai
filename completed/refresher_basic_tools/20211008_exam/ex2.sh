(( answer = RANDOM % 101 ))

while (( guess != answer )); do
	read -p "Guess a number: " guess
	if (( guess < answer )); then
		echo "The right answer is higher"
	elif (( guess > answer )); then
		echo "The right answer is lower"
	fi
done
